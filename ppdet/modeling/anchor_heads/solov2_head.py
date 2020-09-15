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

from ppdet.modeling.ops import ConvNorm, DeformConvNorm, MaskMatrixNMS
from ppdet.core.workspace import register

from ppdet.utils.check import check_version

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
        ins_loss_weight (float): Weight of instance loss.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
        dcn_v2_stages (list): Which stage use dcn v2 in tower.
        segm_strides (list[int]): List of segmentation area stride.
        score_threshold (float): Threshold of categroy score.
        update_threshold (float): Updated threshold of categroy score in second time.
        pre_nms_top_n (int): Number of total instance to be kept per image before NMS
        post_nms_top_n (int): Number of total instance to be kept per image after NMS.
        mask_nms (object): MaskMatrixNMS instance.
    """
    __inject__ = []
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 num_grids=[40, 36, 24, 16, 12],
                 kernel_out_channels=256,
                 ins_loss_weight=3.0,
                 focal_loss_gamma=2.0,
                 focal_loss_alpha=0.25,
                 dcn_v2_stages=[],
                 segm_strides=[8, 8, 16, 32, 32],
                 score_threshold=0.1,
                 mask_threshold=0.5,
                 update_threshold=0.05,
                 pre_nms_top_n=500,
                 post_nms_top_n=100,
                 mask_nms=MaskMatrixNMS(
                     kernel='gaussian', sigma=2.0).__dict__):
        check_version('2.0.0')
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels
        self.ins_loss_weight = ins_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.dcn_v2_stages = dcn_v2_stages
        self.segm_strides = segm_strides
        self.mask_nms = mask_nms
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.update_threshold = update_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.conv_type = [ConvNorm, DeformConvNorm]
        if isinstance(mask_nms, dict):
            self.mask_nms = MaskMatrixNMS(**mask_nms)

    def _conv_pred(self, conv_feat, num_filters, name, name_feat=None):
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
        keep = fluid.layers.cast((hmax[:, :, :-1, :-1] == heat), 'float32')
        return heat * keep

    def dice_loss(self, input, target):
        input = fluid.layers.reshape(
            input, shape=(fluid.layers.shape(input)[0], -1))
        target = fluid.layers.reshape(
            target, shape=(fluid.layers.shape(target)[0], -1))
        target = fluid.layers.cast(target, 'float32')
        a = fluid.layers.reduce_sum(input * target, dim=1)
        b = fluid.layers.reduce_sum(input * input, dim=1) + 0.001
        c = fluid.layers.reduce_sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

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

    def get_outputs(self, input, is_eval=False, batch_size=1):
        """
        Get SOLOv2 head output

        Args:
            input (list): List of Variables, output of backbone or neck stages
            is_eval (bool): whether in train or test mode
            batch_size (int): batch size
        Returns:
            cate_pred_list (list): Variables of each category branch layer
            kernel_pred_list (list): Variables of each kernel branch layer
        """
        feats = self._split_feats(input)
        cate_pred_list = []
        kernel_pred_list = []
        for idx in range(len(self.seg_num_grids)):
            cate_pred, kernel_pred = self._get_output_single(
                feats[idx], idx, is_eval=is_eval, batch_size=batch_size)
            cate_pred_list.append(cate_pred)
            kernel_pred_list.append(kernel_pred)

        return cate_pred_list, kernel_pred_list

    def _get_output_single(self, input, idx, is_eval=False, batch_size=1):
        ins_kernel_feat = input
        # CoordConv
        x_range = paddle.linspace(
            -1, 1, fluid.layers.shape(ins_kernel_feat)[-1], dtype='float32')
        y_range = paddle.linspace(
            -1, 1, fluid.layers.shape(ins_kernel_feat)[-2], dtype='float32')
        y, x = paddle.tensor.meshgrid([y_range, x_range])
        x = fluid.layers.unsqueeze(x, [0, 1])
        y = fluid.layers.unsqueeze(y, [0, 1])
        y = fluid.layers.expand(y, expand_times=[batch_size, 1, 1, 1])
        x = fluid.layers.expand(x, expand_times=[batch_size, 1, 1, 1])
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
            name='bbox_head.kernel_convs',
            name_feat='bbox_head.solo_kernel')

        # cate branch
        cate_pred = self._conv_pred(
            cate_feat,
            self.cate_out_channels,
            name='bbox_head.cate_convs',
            name_feat='bbox_head.solo_cate')

        if is_eval:
            cate_pred = self._points_nms(
                fluid.layers.sigmoid(cate_pred), kernel=2)
            cate_pred = fluid.layers.transpose(cate_pred, [0, 2, 3, 1])
        return cate_pred, kernel_pred

    def get_loss(self,
                 cate_preds,
                 kernel_preds,
                 ins_pred,
                 ins_labels,
                 cate_labels,
                 grid_order_list,
                 fg_num,
                 grid_offset,
                 batch_size=1):
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
            grid_offset (list): List of offset of pre grid.
            batch_size: Batch size.
        Returns:
            loss_ins (Variable): The instance loss Variable of SOLOv2 network.
            loss_cate (Variable): The category loss Variable of SOLOv2 network.
        """
        new_kernel_preds = []
        grid_offset_list = fluid.layers.split(
            grid_offset, num_or_sections=len(grid_order_list), dim=1)
        pred_weight_list = []
        for kernel_preds_level, grid_orders_level, grid_offset_level in zip(
                kernel_preds, grid_order_list, grid_offset_list):
            tmp_list = []
            kernel_pred_weight = []
            start_order_num = fluid.layers.zeros(shape=[1], dtype='int32')
            for i in range(batch_size):
                reshape_pred = fluid.layers.reshape(
                    kernel_preds_level[i],
                    shape=(int(kernel_preds_level[i].shape[0]), -1))
                end_order_num = start_order_num + grid_offset_level[i]
                grid_order_img = fluid.layers.slice(
                    grid_orders_level,
                    axes=[0],
                    starts=[start_order_num],
                    ends=[end_order_num])
                start_order_num = end_order_num
                reshape_pred = fluid.layers.transpose(reshape_pred, [1, 0])
                reshape_pred = fluid.layers.gather(
                    reshape_pred, index=grid_order_img)
                reshape_pred = fluid.layers.transpose(reshape_pred, [1, 0])
                tmp_list.append(reshape_pred)
            new_kernel_preds.append(tmp_list)

        # generate masks
        ins_pred_list = []
        for b_kernel_pred in new_kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                cur_ins_pred = ins_pred[idx]
                cur_ins_pred = fluid.layers.unsqueeze(cur_ins_pred, 0)
                kernel_pred = fluid.layers.transpose(kernel_pred, [1, 0])
                kernel_pred = fluid.layers.unsqueeze(kernel_pred, [2, 3])

                ins_pred_conv = paddle.nn.functional.conv2d(cur_ins_pred,
                                                            kernel_pred)
                cur_ins_pred = ins_pred_conv[0]
                b_mask_pred.append(cur_ins_pred)

            b_mask_pred = fluid.layers.concat(b_mask_pred, axis=0)
            ins_pred_list.append(b_mask_pred)

        num_ins = fluid.layers.reduce_sum(fg_num)

        # Ues dice_loss to calculate instance loss
        loss_ins = []
        total_weights = fluid.layers.zeros(shape=[1], dtype='float32')
        for input, target in zip(ins_pred_list, ins_labels):
            weights = fluid.layers.cast(
                fluid.layers.reduce_sum(
                    target, dim=[1, 2]) > 0, 'float32')
            input = fluid.layers.sigmoid(input)
            dice_out = fluid.layers.elementwise_mul(
                self.dice_loss(input, target), weights)
            total_weights += fluid.layers.reduce_sum(weights)
            loss_ins.append(dice_out)
        loss_ins = fluid.layers.reduce_sum(fluid.layers.concat(
            loss_ins)) / total_weights
        loss_ins = loss_ins * self.ins_loss_weight

        # Ues sigmoid_focal_loss to calculate category loss
        cate_preds = [
            fluid.layers.reshape(
                fluid.layers.transpose(cate_pred, [0, 2, 3, 1]),
                shape=(-1, self.cate_out_channels)) for cate_pred in cate_preds
        ]
        flatten_cate_preds = fluid.layers.concat(cate_preds)
        new_cate_labels = []
        cate_labels = fluid.layers.concat(cate_labels)
        cate_labels = fluid.layers.unsqueeze(cate_labels, 1)
        loss_cate = fluid.layers.sigmoid_focal_loss(
            x=flatten_cate_preds,
            label=cate_labels,
            fg_num=num_ins + 1,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha)
        loss_cate = fluid.layers.reduce_sum(loss_cate)

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

    def sort_score(self, scores, top_num):
        self.case_scores = scores

        def fn_1():
            return fluid.layers.topk(self.case_scores, top_num)

        def fn_2():
            return fluid.layers.argsort(self.case_scores, descending=True)

        sort_inds = fluid.layers.case(
            pred_fn_pairs=[(fluid.layers.shape(scores)[0] > top_num, fn_1)],
            default=fn_2)
        return sort_inds

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
                    dtype="int32",
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

        keep = fluid.layers.where(sum_masks > strides)
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
        seg_mul = fluid.layers.cast(seg_preds * seg_masks, 'float32')
        seg_scores = fluid.layers.reduce_sum(seg_mul, dim=[1, 2]) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = self.sort_score(cate_scores, self.pre_nms_top_n)

        seg_masks = fluid.layers.gather(seg_masks, index=sort_inds[1])
        seg_preds = fluid.layers.gather(seg_preds, index=sort_inds[1])
        sum_masks = fluid.layers.gather(sum_masks, index=sort_inds[1])
        cate_scores = sort_inds[0]
        cate_labels = fluid.layers.gather(cate_labels, index=sort_inds[1])

        # Matrix NMS
        cate_scores = self.mask_nms(
            seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)

        keep = fluid.layers.where(cate_scores >= self.update_threshold)
        keep = fluid.layers.squeeze(keep, axes=[1])
        # Prevent empty and increase fake data
        keep = fluid.layers.concat([
            keep, fluid.layers.cast(
                fluid.layers.shape(cate_scores)[0] - 1, 'int64')
        ])

        seg_preds = fluid.layers.gather(seg_preds, index=keep)
        cate_scores = fluid.layers.gather(cate_scores, index=keep)
        cate_labels = fluid.layers.gather(cate_labels, index=keep)

        # sort and keep top_k
        sort_inds = self.sort_score(cate_scores, self.post_nms_top_n)

        seg_preds = fluid.layers.gather(seg_preds, index=sort_inds[1])
        cate_scores = sort_inds[0]
        cate_labels = fluid.layers.gather(cate_labels, index=sort_inds[1])
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
        seg_masks = fluid.layers.cast(seg_masks > self.mask_threshold, 'int32')
        return seg_masks, cate_labels, cate_scores
