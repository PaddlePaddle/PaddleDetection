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
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant

from ppdet.modeling.layers import ConvNormLayer, MaskMatrixNMS
from ppdet.core.workspace import register

from six.moves import zip
import numpy as np

__all__ = ['SOLOv2Head']


@register
class SOLOv2MaskHead(nn.Layer):
    """
    MaskHead of SOLOv2

    Args:
        in_channels (int): The channel number of input Tensor.
        out_channels (int): The channel number of output Tensor.
        start_level (int): The position where the input starts.
        end_level (int): The position where the input ends.
        use_dcn_in_tower (bool): Whether to use dcn in tower or not.
    """

    def __init__(self,
                 in_channels=256,
                 mid_channels=128,
                 out_channels=256,
                 start_level=0,
                 end_level=3,
                 use_dcn_in_tower=False):
        super(SOLOv2MaskHead, self).__init__()
        assert start_level >= 0 and end_level >= start_level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.use_dcn_in_tower = use_dcn_in_tower
        self.range_level = end_level - start_level + 1
        # TODO: add DeformConvNorm
        conv_type = [ConvNormLayer]
        self.conv_func = conv_type[0]
        if self.use_dcn_in_tower:
            self.conv_func = conv_type[1]
        self.convs_all_levels = []
        for i in range(start_level, end_level + 1):
            conv_feat_name = 'mask_feat_head.convs_all_levels.{}'.format(i)
            conv_pre_feat = nn.Sequential()
            if i == start_level:
                conv_pre_feat.add_sublayer(
                    conv_feat_name + '.conv' + str(i),
                    self.conv_func(
                        ch_in=self.in_channels,
                        ch_out=self.mid_channels,
                        filter_size=3,
                        stride=1,
                        norm_type='gn',
                        norm_name=conv_feat_name + '.conv' + str(i) + '.gn',
                        name=conv_feat_name + '.conv' + str(i)))
                self.add_sublayer('conv_pre_feat' + str(i), conv_pre_feat)
                self.convs_all_levels.append(conv_pre_feat)
            else:
                for j in range(i):
                    ch_in = 0
                    if j == 0:
                        ch_in = self.in_channels + 2 if i == end_level else self.in_channels
                    else:
                        ch_in = self.mid_channels
                    conv_pre_feat.add_sublayer(
                        conv_feat_name + '.conv' + str(j),
                        self.conv_func(
                            ch_in=ch_in,
                            ch_out=self.mid_channels,
                            filter_size=3,
                            stride=1,
                            norm_type='gn',
                            norm_name=conv_feat_name + '.conv' + str(j) + '.gn',
                            name=conv_feat_name + '.conv' + str(j)))
                    conv_pre_feat.add_sublayer(
                        conv_feat_name + '.conv' + str(j) + 'act', nn.ReLU())
                    conv_pre_feat.add_sublayer(
                        'upsample' + str(i) + str(j),
                        nn.Upsample(
                            scale_factor=2, mode='bilinear'))
                self.add_sublayer('conv_pre_feat' + str(i), conv_pre_feat)
                self.convs_all_levels.append(conv_pre_feat)

        conv_pred_name = 'mask_feat_head.conv_pred.0'
        self.conv_pred = self.add_sublayer(
            conv_pred_name,
            self.conv_func(
                ch_in=self.mid_channels,
                ch_out=self.out_channels,
                filter_size=1,
                stride=1,
                norm_type='gn',
                norm_name=conv_pred_name + '.gn',
                name=conv_pred_name))

    def forward(self, inputs):
        """
        Get SOLOv2MaskHead output.

        Args:
            inputs(list[Tensor]): feature map from each necks with shape of [N, C, H, W]
        Returns:
            ins_pred(Tensor): Output of SOLOv2MaskHead head
        """
        feat_all_level = F.relu(self.convs_all_levels[0](inputs[0]))
        for i in range(1, self.range_level):
            input_p = inputs[i]
            if i == (self.range_level - 1):
                input_feat = input_p
                x_range = paddle.linspace(
                    -1, 1, paddle.shape(input_feat)[-1], dtype='float32')
                y_range = paddle.linspace(
                    -1, 1, paddle.shape(input_feat)[-2], dtype='float32')
                y, x = paddle.meshgrid([y_range, x_range])
                x = paddle.unsqueeze(x, [0, 1])
                y = paddle.unsqueeze(y, [0, 1])
                y = paddle.expand(
                    y, shape=[paddle.shape(input_feat)[0], 1, -1, -1])
                x = paddle.expand(
                    x, shape=[paddle.shape(input_feat)[0], 1, -1, -1])
                coord_feat = paddle.concat([x, y], axis=1)
                input_p = paddle.concat([input_p, coord_feat], axis=1)
            feat_all_level = paddle.add(feat_all_level,
                                        self.convs_all_levels[i](input_p))
        ins_pred = F.relu(self.conv_pred(feat_all_level))

        return ins_pred


@register
class SOLOv2Head(nn.Layer):
    """
    Head block for SOLOv2 network

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        seg_feat_channels (int): Num_filters of kernel & categroy branch convolution operation.
        stacked_convs (int): Times of convolution operation.
        num_grids (list[int]): List of feature map grids size.
        kernel_out_channels (int): Number of output channels in kernel branch.
        dcn_v2_stages (list): Which stage use dcn v2 in tower. It is between [0, stacked_convs).
        segm_strides (list[int]): List of segmentation area stride.
        solov2_loss (object): SOLOv2Loss instance.
        score_threshold (float): Threshold of categroy score.
        mask_nms (object): MaskMatrixNMS instance.
    """
    __inject__ = ['solov2_loss', 'mask_nms']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 num_grids=[40, 36, 24, 16, 12],
                 kernel_out_channels=256,
                 dcn_v2_stages=[],
                 segm_strides=[8, 8, 16, 32, 32],
                 solov2_loss=None,
                 score_threshold=0.1,
                 mask_threshold=0.5,
                 mask_nms=None):
        super(SOLOv2Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels
        self.dcn_v2_stages = dcn_v2_stages
        self.segm_strides = segm_strides
        self.solov2_loss = solov2_loss
        self.mask_nms = mask_nms
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

        conv_type = [ConvNormLayer]
        self.conv_func = conv_type[0]
        self.kernel_pred_convs = []
        self.cate_pred_convs = []
        for i in range(self.stacked_convs):
            if i in self.dcn_v2_stages:
                self.conv_func = conv_type[1]
            ch_in = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            kernel_conv = self.add_sublayer(
                'bbox_head.kernel_convs.' + str(i),
                self.conv_func(
                    ch_in=ch_in,
                    ch_out=self.seg_feat_channels,
                    filter_size=3,
                    stride=1,
                    norm_type='gn',
                    norm_name='bbox_head.kernel_convs.{}.gn'.format(i),
                    name='bbox_head.kernel_convs.{}'.format(i)))
            self.kernel_pred_convs.append(kernel_conv)
            ch_in = self.in_channels if i == 0 else self.seg_feat_channels
            cate_conv = self.add_sublayer(
                'bbox_head.cate_convs.' + str(i),
                self.conv_func(
                    ch_in=ch_in,
                    ch_out=self.seg_feat_channels,
                    filter_size=3,
                    stride=1,
                    norm_type='gn',
                    norm_name='bbox_head.cate_convs.{}.gn'.format(i),
                    name='bbox_head.cate_convs.{}'.format(i)))
            self.cate_pred_convs.append(cate_conv)

        self.solo_kernel = self.add_sublayer(
            'bbox_head.solo_kernel',
            nn.Conv2D(
                self.seg_feat_channels,
                self.kernel_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(
                    name="bbox_head.solo_kernel.weight",
                    initializer=Normal(
                        mean=0., std=0.01)),
                bias_attr=ParamAttr(name="bbox_head.solo_kernel.bias")))
        self.solo_cate = self.add_sublayer(
            'bbox_head.solo_cate',
            nn.Conv2D(
                self.seg_feat_channels,
                self.cate_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(
                    name="bbox_head.solo_cate.weight",
                    initializer=Normal(
                        mean=0., std=0.01)),
                bias_attr=ParamAttr(
                    name="bbox_head.solo_cate.bias",
                    initializer=Constant(
                        value=float(-np.log((1 - 0.01) / 0.01))))))

    def _points_nms(self, heat, kernel_size=2):
        hmax = F.max_pool2d(heat, kernel_size=kernel_size, stride=1, padding=1)
        keep = paddle.cast((hmax[:, :, :-1, :-1] == heat), 'float32')
        return heat * keep

    def _split_feats(self, feats):
        return (F.interpolate(
            feats[0],
            scale_factor=0.5,
            align_corners=False,
            align_mode=0,
            mode='bilinear'), feats[1], feats[2], feats[3], F.interpolate(
                feats[4],
                size=paddle.shape(feats[3])[-2:],
                mode='bilinear',
                align_corners=False,
                align_mode=0))

    def forward(self, input):
        """
        Get SOLOv2 head output

        Args:
            input (list): List of Tensors, output of backbone or neck stages
        Returns:
            cate_pred_list (list): Tensors of each category branch layer
            kernel_pred_list (list): Tensors of each kernel branch layer
        """
        feats = self._split_feats(input)
        cate_pred_list = []
        kernel_pred_list = []
        for idx in range(len(self.seg_num_grids)):
            cate_pred, kernel_pred = self._get_output_single(feats[idx], idx)
            cate_pred_list.append(cate_pred)
            kernel_pred_list.append(kernel_pred)

        return cate_pred_list, kernel_pred_list

    def _get_output_single(self, input, idx):
        ins_kernel_feat = input
        # CoordConv
        x_range = paddle.linspace(
            -1, 1, paddle.shape(ins_kernel_feat)[-1], dtype='float32')
        y_range = paddle.linspace(
            -1, 1, paddle.shape(ins_kernel_feat)[-2], dtype='float32')
        y, x = paddle.meshgrid([y_range, x_range])
        x = paddle.unsqueeze(x, [0, 1])
        y = paddle.unsqueeze(y, [0, 1])
        y = paddle.expand(
            y, shape=[paddle.shape(ins_kernel_feat)[0], 1, -1, -1])
        x = paddle.expand(
            x, shape=[paddle.shape(ins_kernel_feat)[0], 1, -1, -1])
        coord_feat = paddle.concat([x, y], axis=1)
        ins_kernel_feat = paddle.concat([ins_kernel_feat, coord_feat], axis=1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(
            kernel_feat,
            size=[seg_num_grid, seg_num_grid],
            mode='bilinear',
            align_corners=False,
            align_mode=0)
        cate_feat = kernel_feat[:, :-2, :, :]

        for kernel_layer in self.kernel_pred_convs:
            kernel_feat = F.relu(kernel_layer(kernel_feat))
        kernel_pred = self.solo_kernel(kernel_feat)
        # cate branch
        for cate_layer in self.cate_pred_convs:
            cate_feat = F.relu(cate_layer(cate_feat))
        cate_pred = self.solo_cate(cate_feat)

        if not self.training:
            cate_pred = self._points_nms(F.sigmoid(cate_pred), kernel_size=2)
            cate_pred = paddle.transpose(cate_pred, [0, 2, 3, 1])
        return cate_pred, kernel_pred

    def get_loss(self, cate_preds, kernel_preds, ins_pred, ins_labels,
                 cate_labels, grid_order_list, fg_num):
        """
        Get loss of network of SOLOv2.

        Args:
            cate_preds (list): Tensor list of categroy branch output.
            kernel_preds (list): Tensor list of kernel branch output.
            ins_pred (list): Tensor list of instance branch output.
            ins_labels (list): List of instance labels pre batch.
            cate_labels (list): List of categroy labels pre batch.
            grid_order_list (list): List of index in pre grid.
            fg_num (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Tensor): The instance loss Tensor of SOLOv2 network.
            loss_cate (Tensor): The category loss Tensor of SOLOv2 network.
        """
        batch_size = paddle.shape(grid_order_list[0])[0]
        ins_pred_list = []
        for kernel_preds_level, grid_orders_level in zip(kernel_preds,
                                                         grid_order_list):
            if grid_orders_level.shape[1] == 0:
                ins_pred_list.append(None)
                continue
            grid_orders_level = paddle.reshape(grid_orders_level, [-1])
            reshape_pred = paddle.reshape(
                kernel_preds_level,
                shape=(paddle.shape(kernel_preds_level)[0],
                       paddle.shape(kernel_preds_level)[1], -1))
            reshape_pred = paddle.transpose(reshape_pred, [0, 2, 1])
            reshape_pred = paddle.reshape(
                reshape_pred, shape=(-1, paddle.shape(reshape_pred)[2]))
            gathered_pred = paddle.gather(reshape_pred, index=grid_orders_level)
            gathered_pred = paddle.reshape(
                gathered_pred,
                shape=[batch_size, -1, paddle.shape(gathered_pred)[1]])
            cur_ins_pred = ins_pred
            cur_ins_pred = paddle.reshape(
                cur_ins_pred,
                shape=(paddle.shape(cur_ins_pred)[0],
                       paddle.shape(cur_ins_pred)[1], -1))
            ins_pred_conv = paddle.matmul(gathered_pred, cur_ins_pred)
            cur_ins_pred = paddle.reshape(
                ins_pred_conv,
                shape=(-1, paddle.shape(ins_pred)[-2],
                       paddle.shape(ins_pred)[-1]))
            ins_pred_list.append(cur_ins_pred)

        num_ins = paddle.sum(fg_num)
        cate_preds = [
            paddle.reshape(
                paddle.transpose(cate_pred, [0, 2, 3, 1]),
                shape=(-1, self.cate_out_channels)) for cate_pred in cate_preds
        ]
        flatten_cate_preds = paddle.concat(cate_preds)
        new_cate_labels = []
        for cate_label in cate_labels:
            new_cate_labels.append(paddle.reshape(cate_label, shape=[-1]))
        cate_labels = paddle.concat(new_cate_labels)

        loss_ins, loss_cate = self.solov2_loss(
            ins_pred_list, ins_labels, flatten_cate_preds, cate_labels, num_ins)

        return {'loss_ins': loss_ins, 'loss_cate': loss_cate}

    def get_prediction(self, cate_preds, kernel_preds, seg_pred, im_shape,
                       scale_factor):
        """
        Get prediction result of SOLOv2 network

        Args:
            cate_preds (list): List of Variables, output of categroy branch.
            kernel_preds (list): List of Variables, output of kernel branch.
            seg_pred (list): List of Variables, output of mask head stages.
            im_shape (Variables): [h, w] for input images.
            scale_factor (Variables): [scale, scale] for input images.
        Returns:
            seg_masks (Tensor): The prediction segmentation.
            cate_labels (Tensor): The prediction categroy label of each segmentation.
            seg_masks (Tensor): The prediction score of each segmentation.
        """
        num_levels = len(cate_preds)
        featmap_size = paddle.shape(seg_pred)[-2:]
        seg_masks_list = []
        cate_labels_list = []
        cate_scores_list = []
        cate_preds = [cate_pred * 1.0 for cate_pred in cate_preds]
        kernel_preds = [kernel_pred * 1.0 for kernel_pred in kernel_preds]
        # Currently only supports batch size == 1
        for idx in range(1):
            cate_pred_list = [
                paddle.reshape(
                    cate_preds[i][idx], shape=(-1, self.cate_out_channels))
                for i in range(num_levels)
            ]
            seg_pred_list = seg_pred
            kernel_pred_list = [
                paddle.reshape(
                    paddle.transpose(kernel_preds[i][idx], [1, 2, 0]),
                    shape=(-1, self.kernel_out_channels))
                for i in range(num_levels)
            ]
            cate_pred_list = paddle.concat(cate_pred_list, axis=0)
            kernel_pred_list = paddle.concat(kernel_pred_list, axis=0)

            seg_masks, cate_labels, cate_scores = self.get_seg_single(
                cate_pred_list, seg_pred_list, kernel_pred_list, featmap_size,
                im_shape[idx], scale_factor[idx][0])
            bbox_num = paddle.shape(cate_labels)[0]
        return seg_masks, cate_labels, cate_scores, bbox_num

    def get_seg_single(self, cate_preds, seg_preds, kernel_preds, featmap_size,
                       im_shape, scale_factor):
        h = paddle.cast(im_shape[0], 'int32')[0]
        w = paddle.cast(im_shape[1], 'int32')[0]
        upsampled_size_out = [featmap_size[0] * 4, featmap_size[1] * 4]

        y = paddle.zeros(shape=paddle.shape(cate_preds), dtype='float32')
        inds = paddle.where(cate_preds > self.score_threshold, cate_preds, y)
        inds = paddle.nonzero(inds)
        if paddle.shape(inds)[0] == 0:
            out = paddle.full(shape=[1], fill_value=-1)
            return out, out, out
        cate_preds = paddle.reshape(cate_preds, shape=[-1])
        # Prevent empty and increase fake data
        ind_a = paddle.cast(paddle.shape(kernel_preds)[0], 'int64')
        ind_b = paddle.zeros(shape=[1], dtype='int64')
        inds_end = paddle.unsqueeze(paddle.concat([ind_a, ind_b]), 0)
        inds = paddle.concat([inds, inds_end])
        kernel_preds_end = paddle.ones(
            shape=[1, self.kernel_out_channels], dtype='float32')
        kernel_preds = paddle.concat([kernel_preds, kernel_preds_end])
        cate_preds = paddle.concat(
            [cate_preds, paddle.zeros(
                shape=[1], dtype='float32')])

        # cate_labels & kernel_preds
        cate_labels = inds[:, 1]
        kernel_preds = paddle.gather(kernel_preds, index=inds[:, 0])
        cate_score_idx = paddle.add(inds[:, 0] * 80, cate_labels)
        cate_scores = paddle.gather(cate_preds, index=cate_score_idx)

        size_trans = np.power(self.seg_num_grids, 2)
        strides = []
        for _ind in range(len(self.segm_strides)):
            strides.append(
                paddle.full(
                    shape=[int(size_trans[_ind])],
                    fill_value=self.segm_strides[_ind],
                    dtype="int32"))
        strides = paddle.concat(strides)
        strides = paddle.gather(strides, index=inds[:, 0])

        # mask encoding.
        kernel_preds = paddle.unsqueeze(kernel_preds, [2, 3])
        seg_preds = F.conv2d(seg_preds, kernel_preds)
        seg_preds = F.sigmoid(paddle.squeeze(seg_preds, [0]))
        seg_masks = seg_preds > self.mask_threshold
        seg_masks = paddle.cast(seg_masks, 'float32')
        sum_masks = paddle.sum(seg_masks, axis=[1, 2])

        y = paddle.zeros(shape=paddle.shape(sum_masks), dtype='float32')
        keep = paddle.where(sum_masks > strides, sum_masks, y)
        keep = paddle.nonzero(keep)
        keep = paddle.squeeze(keep, axis=[1])
        # Prevent empty and increase fake data
        keep_other = paddle.concat(
            [keep, paddle.cast(paddle.shape(sum_masks)[0] - 1, 'int64')])
        keep_scores = paddle.concat(
            [keep, paddle.cast(paddle.shape(sum_masks)[0], 'int64')])
        cate_scores_end = paddle.zeros(shape=[1], dtype='float32')
        cate_scores = paddle.concat([cate_scores, cate_scores_end])

        seg_masks = paddle.gather(seg_masks, index=keep_other)
        seg_preds = paddle.gather(seg_preds, index=keep_other)
        sum_masks = paddle.gather(sum_masks, index=keep_other)
        cate_labels = paddle.gather(cate_labels, index=keep_other)
        cate_scores = paddle.gather(cate_scores, index=keep_scores)

        # mask scoring.
        seg_mul = paddle.cast(seg_preds * seg_masks, 'float32')
        seg_scores = paddle.sum(seg_mul, axis=[1, 2]) / sum_masks
        cate_scores *= seg_scores
        # Matrix NMS
        seg_preds, cate_scores, cate_labels = self.mask_nms(
            seg_preds, seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)
        ori_shape = im_shape[:2] / scale_factor + 0.5
        ori_shape = paddle.cast(ori_shape, 'int32')
        seg_preds = F.interpolate(
            paddle.unsqueeze(seg_preds, 0),
            size=upsampled_size_out,
            mode='bilinear',
            align_corners=False,
            align_mode=0)
        seg_preds = paddle.slice(
            seg_preds, axes=[2, 3], starts=[0, 0], ends=[h, w])
        seg_masks = paddle.squeeze(
            F.interpolate(
                seg_preds,
                size=ori_shape[:2],
                mode='bilinear',
                align_corners=False,
                align_mode=0),
            axis=[0])
        # TODO: support bool type
        seg_masks = paddle.cast(seg_masks > self.mask_threshold, 'int32')
        return seg_masks, cate_labels, cate_scores
