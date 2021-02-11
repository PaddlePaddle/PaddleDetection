# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.modeling.ops import ConvNorm, DeformConvNorm, MaskMatrixNMS, DropBlock
from ppdet.core.workspace import register

from six.moves import zip
import numpy as np

__all__ = ['SOLOv2Head']


@register
class SOLOv2Head(object):
    """
    Head block for SOLOv2 network

    Args:
        num_classes (int): Number of output classes.
        seg_feat_channels (int): Num_filters of kernel & categroy branch convolution operation.
        stacked_convs (int): Times of convolution operation.
        num_grids (list[int]): List of feature map grids size.
        kernel_out_channels (int): Number of output channels in kernel branch.
        dcn_v2_stages (list): Which stage use dcn v2 in tower.
        segm_strides (list[int]): List of segmentation area stride.
        solov2_loss (object): SOLOv2Loss instance.
        score_threshold (float): Threshold of categroy score.
        mask_nms (object): MaskMatrixNMS instance.
        drop_block (bool): Whether use drop_block or not.
    """
    __inject__ = ['solov2_loss', 'mask_nms']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 num_grids=[40, 36, 24, 16, 12],
                 kernel_out_channels=256,
                 dcn_v2_stages=[],
                 segm_strides=[8, 8, 16, 32, 32],
                 solov2_loss=None,
                 score_threshold=0.1,
                 mask_threshold=0.5,
                 mask_nms=MaskMatrixNMS(
                     update_threshold=0.05,
                     pre_nms_top_n=500,
                     post_nms_top_n=100,
                     kernel='gaussian',
                     sigma=2.0).__dict__,
                 drop_block=False):
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels
        self.dcn_v2_stages = dcn_v2_stages
        self.segm_strides = segm_strides
        self.solov2_loss = solov2_loss
        self.mask_nms = mask_nms
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.drop_block = drop_block
        self.conv_type = [ConvNorm, DeformConvNorm]
        if isinstance(mask_nms, dict):
            self.mask_nms = MaskMatrixNMS(**mask_nms)

    def _conv_pred(self, conv_feat, num_filters, is_test, name, name_feat=None):
        for i in range(self.stacked_convs):
            if i in self.dcn_v2_stages:
                conv_func = self.conv_type[1]
            else:
                conv_func = self.conv_type[0]
            conv_feat = conv_func(
                input=conv_feat,
                num_filters=self.seg_feat_channels,
                filter_size=3,
                stride=1,
                norm_type='gn',
                norm_groups=32,
                freeze_norm=False,
                act='relu',
                initializer=fluid.initializer.NormalInitializer(scale=0.01),
                norm_name='{}.{}.gn'.format(name, i),
                name='{}.{}'.format(name, i))
        if name_feat == 'bbox_head.solo_cate':
            bias_init = float(-np.log((1 - 0.01) / 0.01))
            bias_attr = ParamAttr(
                name="{}.bias".format(name_feat),
                initializer=fluid.initializer.Constant(value=bias_init))
        else:
            bias_attr = ParamAttr(name="{}.bias".format(name_feat))

        if self.drop_block:
            conv_feat = DropBlock(
                conv_feat, block_size=3, keep_prob=0.9, is_test=is_test)

        conv_feat = fluid.layers.conv2d(
            input=conv_feat,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            padding=1,
            param_attr=ParamAttr(
                name="{}.weight".format(name_feat),
                initializer=fluid.initializer.NormalInitializer(scale=0.01)),
            bias_attr=bias_attr,
            name=name + '_feat_')
        return conv_feat

    def _points_nms(self, heat, kernel=2):
        hmax = fluid.layers.pool2d(
            input=heat, pool_size=kernel, pool_type='max', pool_padding=1)
        keep = fluid.layers.cast(
            paddle.equal(hmax[:, :, :-1, :-1], heat), 'float32')
        return paddle.multiply(heat, keep)

    def _split_feats(self, feats):
        return (paddle.nn.functional.interpolate(
            feats[0],
            scale_factor=0.5,
            align_corners=False,
            align_mode=0,
            mode='bilinear'), feats[1], feats[2], feats[3],
                paddle.nn.functional.interpolate(
                    feats[4],
                    size=fluid.layers.shape(feats[3])[-2:],
                    mode='bilinear',
                    align_corners=False,
                    align_mode=0))

    def get_outputs(self, input, is_eval=False):
        """
        Get SOLOv2 head output

        Args:
            input (list): List of Variables, output of backbone or neck stages
            is_eval (bool): whether in train or test mode
        Returns:
            cate_pred_list (list): Variables of each category branch layer
            kernel_pred_list (list): Variables of each kernel branch layer
        """
        feats = self._split_feats(input)
        cate_pred_list = []
        kernel_pred_list = []
        for idx in range(len(self.seg_num_grids)):
            cate_pred, kernel_pred = self._get_output_single(
                feats[idx], idx, is_eval=is_eval)
            cate_pred_list.append(cate_pred)
            kernel_pred_list.append(kernel_pred)

        return cate_pred_list, kernel_pred_list

    def _get_output_single(self, input, idx, is_eval=False):
        ins_kernel_feat = input
        # CoordConv
        x_range = paddle.linspace(
            -1, 1, fluid.layers.shape(ins_kernel_feat)[-1], dtype='float32')
        y_range = paddle.linspace(
            -1, 1, fluid.layers.shape(ins_kernel_feat)[-2], dtype='float32')
        y, x = paddle.tensor.meshgrid([y_range, x_range])
        x = fluid.layers.unsqueeze(x, [0, 1])
        y = fluid.layers.unsqueeze(y, [0, 1])
        y = fluid.layers.expand(
            y, expand_times=[fluid.layers.shape(ins_kernel_feat)[0], 1, 1, 1])
        x = fluid.layers.expand(
            x, expand_times=[fluid.layers.shape(ins_kernel_feat)[0], 1, 1, 1])
        coord_feat = fluid.layers.concat([x, y], axis=1)
        ins_kernel_feat = fluid.layers.concat(
            [ins_kernel_feat, coord_feat], axis=1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = paddle.nn.functional.interpolate(
            kernel_feat,
            size=[seg_num_grid, seg_num_grid],
            mode='bilinear',
            align_corners=False,
            align_mode=0)
        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_pred = self._conv_pred(
            kernel_feat,
            self.kernel_out_channels,
            is_eval,
            name='bbox_head.kernel_convs',
            name_feat='bbox_head.solo_kernel')

        # cate branch
        cate_pred = self._conv_pred(
            cate_feat,
            self.cate_out_channels,
            is_eval,
            name='bbox_head.cate_convs',
            name_feat='bbox_head.solo_cate')

        if is_eval:
            cate_pred = self._points_nms(
                fluid.layers.sigmoid(cate_pred), kernel=2)
            cate_pred = fluid.layers.transpose(cate_pred, [0, 2, 3, 1])
        return cate_pred, kernel_pred

    def get_loss(self, cate_preds, kernel_preds, ins_pred, ins_labels,
                 cate_labels, grid_order_list, fg_num):
        """
        Get loss of network of SOLOv2.

        Args:
            cate_preds (list): Variable list of categroy branch output.
            kernel_preds (list): Variable list of kernel branch output.
            ins_pred (list): Variable list of instance branch output.
            ins_labels (list): List of instance labels pre batch.
            cate_labels (list): List of categroy labels pre batch.
            grid_order_list (list): List of index in pre grid.
            fg_num (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Variable): The instance loss Variable of SOLOv2 network.
            loss_cate (Variable): The category loss Variable of SOLOv2 network.
        """
        new_kernel_preds = []
        pad_length_list = []
        for kernel_preds_level, grid_orders_level in zip(kernel_preds,
                                                         grid_order_list):
            reshape_pred = fluid.layers.reshape(
                kernel_preds_level,
                shape=(fluid.layers.shape(kernel_preds_level)[0],
                       fluid.layers.shape(kernel_preds_level)[1], -1))
            reshape_pred = fluid.layers.transpose(reshape_pred, [0, 2, 1])
            reshape_pred = fluid.layers.reshape(
                reshape_pred, shape=(-1, fluid.layers.shape(reshape_pred)[2]))
            gathered_pred = fluid.layers.gather(
                reshape_pred, index=grid_orders_level)
            gathered_pred = fluid.layers.lod_reset(gathered_pred,
                                                   grid_orders_level)
            pad_value = fluid.layers.assign(input=np.array(
                [0.0], dtype=np.float32))
            pad_pred, pad_length = fluid.layers.sequence_pad(
                gathered_pred, pad_value=pad_value)
            new_kernel_preds.append(pad_pred)
            pad_length_list.append(pad_length)

        # generate masks
        ins_pred_list = []
        for kernel_pred, pad_length in zip(new_kernel_preds, pad_length_list):
            cur_ins_pred = ins_pred
            cur_ins_pred = fluid.layers.reshape(
                cur_ins_pred,
                shape=(fluid.layers.shape(cur_ins_pred)[0],
                       fluid.layers.shape(cur_ins_pred)[1], -1))
            ins_pred_conv = paddle.matmul(kernel_pred, cur_ins_pred)
            cur_ins_pred = fluid.layers.reshape(
                ins_pred_conv,
                shape=(fluid.layers.shape(ins_pred_conv)[0],
                       fluid.layers.shape(ins_pred_conv)[1],
                       fluid.layers.shape(ins_pred)[-2],
                       fluid.layers.shape(ins_pred)[-1]))
            cur_ins_pred = fluid.layers.sequence_unpad(cur_ins_pred, pad_length)
            ins_pred_list.append(cur_ins_pred)

        num_ins = fluid.layers.reduce_sum(fg_num)
        cate_preds = [
            fluid.layers.reshape(
                fluid.layers.transpose(cate_pred, [0, 2, 3, 1]),
                shape=(-1, self.cate_out_channels)) for cate_pred in cate_preds
        ]
        flatten_cate_preds = fluid.layers.concat(cate_preds)
        new_cate_labels = []
        cate_labels = fluid.layers.concat(cate_labels)
        cate_labels = fluid.layers.unsqueeze(cate_labels, 1)
        loss_ins, loss_cate = self.solov2_loss(
            ins_pred_list, ins_labels, flatten_cate_preds, cate_labels, num_ins)

        return {'loss_ins': loss_ins, 'loss_cate': loss_cate}

    def get_prediction(self, cate_preds, kernel_preds, seg_pred, im_info):
        """
        Get prediction result of SOLOv2 network

        Args:
            cate_preds (list): List of Variables, output of categroy branch.
            kernel_preds (list): List of Variables, output of kernel branch.
            seg_pred (list): List of Variables, output of mask head stages.
            im_info(Variables): [h, w, scale] for input images.
        Returns:
            seg_masks (Variable): The prediction segmentation.
            cate_labels (Variable): The prediction categroy label of each segmentation.
            seg_masks (Variable): The prediction score of each segmentation.
        """
        num_levels = len(cate_preds)
        featmap_size = fluid.layers.shape(seg_pred)[-2:]
        seg_masks_list = []
        cate_labels_list = []
        cate_scores_list = []
        cate_preds = [cate_pred * 1.0 for cate_pred in cate_preds]
        kernel_preds = [kernel_pred * 1.0 for kernel_pred in kernel_preds]
        # Currently only supports batch size == 1
        for idx in range(1):
            cate_pred_list = [
                fluid.layers.reshape(
                    cate_preds[i][idx], shape=(-1, self.cate_out_channels))
                for i in range(num_levels)
            ]
            seg_pred_list = seg_pred
            kernel_pred_list = [
                fluid.layers.reshape(
                    fluid.layers.transpose(kernel_preds[i][idx], [1, 2, 0]),
                    shape=(-1, self.kernel_out_channels))
                for i in range(num_levels)
            ]
            cate_pred_list = fluid.layers.concat(cate_pred_list, axis=0)
            kernel_pred_list = fluid.layers.concat(kernel_pred_list, axis=0)

            seg_masks, cate_labels, cate_scores = self.get_seg_single(
                cate_pred_list, seg_pred_list, kernel_pred_list, featmap_size,
                im_info[idx])
        return {
            "segm": seg_masks,
            'cate_label': cate_labels,
            'cate_score': cate_scores
        }

    def get_seg_single(self, cate_preds, seg_preds, kernel_preds, featmap_size,
                       im_info):

        im_scale = im_info[2]
        h = fluid.layers.cast(im_info[0], 'int32')
        w = fluid.layers.cast(im_info[1], 'int32')
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        inds = fluid.layers.where(cate_preds > self.score_threshold)
        cate_preds = fluid.layers.reshape(cate_preds, shape=[-1])
        # Prevent empty and increase fake data
        ind_a = fluid.layers.cast(fluid.layers.shape(kernel_preds)[0], 'int64')
        ind_b = fluid.layers.zeros(shape=[1], dtype='int64')
        inds_end = fluid.layers.unsqueeze(
            fluid.layers.concat([ind_a, ind_b]), 0)
        inds = fluid.layers.concat([inds, inds_end])
        kernel_preds_end = fluid.layers.ones(
            shape=[1, self.kernel_out_channels], dtype='float32')
        kernel_preds = fluid.layers.concat([kernel_preds, kernel_preds_end])
        cate_preds = fluid.layers.concat(
            [cate_preds, fluid.layers.zeros(
                shape=[1], dtype='float32')])

        # cate_labels & kernel_preds
        cate_labels = inds[:, 1]
        kernel_preds = fluid.layers.gather(kernel_preds, index=inds[:, 0])
        cate_score_idx = fluid.layers.elementwise_add(inds[:, 0] * 80,
                                                      cate_labels)
        cate_scores = fluid.layers.gather(cate_preds, index=cate_score_idx)

        size_trans = np.power(self.seg_num_grids, 2)
        strides = []
        for _ind in range(len(self.segm_strides)):
            strides.append(
                fluid.layers.fill_constant(
                    shape=[int(size_trans[_ind])],
                    dtype="float32",
                    value=self.segm_strides[_ind]))
        strides = fluid.layers.concat(strides)
        strides = fluid.layers.gather(strides, index=inds[:, 0])

        # mask encoding.
        kernel_preds = fluid.layers.unsqueeze(kernel_preds, [2, 3])
        seg_preds = paddle.nn.functional.conv2d(seg_preds, kernel_preds)
        seg_preds = fluid.layers.sigmoid(fluid.layers.squeeze(seg_preds, [0]))
        seg_masks = seg_preds > self.mask_threshold
        seg_masks = fluid.layers.cast(seg_masks, 'float32')
        sum_masks = fluid.layers.reduce_sum(seg_masks, dim=[1, 2])

        keep = fluid.layers.where(paddle.greater_than(sum_masks, strides))
        keep = fluid.layers.squeeze(keep, axes=[1])
        # Prevent empty and increase fake data
        keep_other = fluid.layers.concat([
            keep, fluid.layers.cast(
                fluid.layers.shape(sum_masks)[0] - 1, 'int64')
        ])
        keep_scores = fluid.layers.concat([
            keep, fluid.layers.cast(fluid.layers.shape(sum_masks)[0], 'int64')
        ])
        cate_scores_end = fluid.layers.zeros(shape=[1], dtype='float32')
        cate_scores = fluid.layers.concat([cate_scores, cate_scores_end])

        seg_masks = fluid.layers.gather(seg_masks, index=keep_other)
        seg_preds = fluid.layers.gather(seg_preds, index=keep_other)
        sum_masks = fluid.layers.gather(sum_masks, index=keep_other)
        cate_labels = fluid.layers.gather(cate_labels, index=keep_other)
        cate_scores = fluid.layers.gather(cate_scores, index=keep_scores)

        # mask scoring.
        seg_mul = fluid.layers.cast(
            paddle.multiply(seg_preds, seg_masks), 'float32')
        seg_scores = paddle.divide(paddle.sum(seg_mul, axis=[1, 2]), sum_masks)
        cate_scores = paddle.multiply(cate_scores, seg_scores)

        # Matrix NMS
        seg_preds, cate_scores, cate_labels = self.mask_nms(
            seg_preds, seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)

        ori_shape = im_info[:2] / im_scale + 0.5
        ori_shape = fluid.layers.cast(ori_shape, 'int32')
        seg_preds = paddle.nn.functional.interpolate(
            fluid.layers.unsqueeze(seg_preds, 0),
            size=upsampled_size_out,
            mode='bilinear',
            align_corners=False,
            align_mode=0)[:, :, :h, :w]
        seg_masks = fluid.layers.squeeze(
            paddle.nn.functional.interpolate(
                seg_preds,
                size=ori_shape[:2],
                mode='bilinear',
                align_corners=False,
                align_mode=0),
            axes=[0])
        # TODO: convert uint8
        seg_masks = fluid.layers.cast(seg_masks > self.mask_threshold, 'int32')
        return seg_masks, cate_labels, cate_scores
