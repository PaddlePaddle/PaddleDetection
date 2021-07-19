# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer
from ppdet.modeling.bbox_utils import distance2bbox, bbox2distance
from ppdet.data.transform.atss_assigner import bbox_overlaps
from .gfl_head import GFLHead


@register
class PicoFeat(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False):
        super(PicoFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.cls_convs = []
        self.reg_convs = []
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                cls_conv_dw = self.add_sublayer(
                    'cls_conv_dw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=3,
                        stride=1,
                        groups=feat_out,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.))
                cls_subnet_convs.append(cls_conv_dw)
                cls_conv_pw = self.add_sublayer(
                    'cls_conv_pw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.))
                cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    reg_conv_dw = self.add_sublayer(
                        'reg_conv_dw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=3,
                            stride=1,
                            groups=feat_out,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.))
                    reg_subnet_convs.append(reg_conv_dw)
                    reg_conv_pw = self.add_sublayer(
                        'reg_conv_pw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.))
                    reg_subnet_convs.append(reg_conv_pw)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = F.leaky_relu(self.cls_convs[stage_idx][i](cls_feat), 0.1)
            if not self.share_cls_reg:
                reg_feat = F.leaky_relu(self.reg_convs[stage_idx][i](reg_feat),
                                        0.1)
        return cls_feat, reg_feat


@register
class PicoHead(GFLHead):
    """
    PicoHead
    Args:
        conv_feat (object): Instance of 'LiteGFLFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_qfl (object):
        loss_dfl (object):
        loss_bbox (object):
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_qfl', 'loss_dfl', 'loss_bbox', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='PicoFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32],
                 prior_prob=0.01,
                 loss_qfl='QualityFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 reg_max=16,
                 feat_in_chan=96,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(PicoHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_qfl=loss_qfl,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_qfl = loss_qfl
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_qfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # Clear the super class initialization
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None

        self.head_cls_list = []
        self.head_reg_list = []
        for i in range(len(fpn_stride)):
            head_cls = self.add_sublayer(
                "head_cls" + str(i),
                nn.Conv2D(
                    in_channels=self.feat_in_chan,
                    out_channels=self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.conv_feat.share_cls_reg else self.cls_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=ParamAttr(
                        initializer=Constant(value=bias_init_value))))
            self.head_cls_list.append(head_cls)
            if not self.conv_feat.share_cls_reg:
                head_reg = self.add_sublayer(
                    "head_reg" + str(i),
                    nn.Conv2D(
                        in_channels=self.feat_in_chan,
                        out_channels=4 * (self.reg_max + 1),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        weight_attr=ParamAttr(initializer=Normal(
                            mean=0., std=0.01)),
                        bias_attr=ParamAttr(initializer=Constant(value=0))))
                self.head_reg_list.append(head_reg)

    def forward(self, fpn_feats):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        cls_logits_list = []
        bboxes_reg_list = []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = paddle.split(
                    cls_logits,
                    [self.cls_out_channels, 4 * (self.reg_max + 1)],
                    axis=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)
            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score

            if not self.training:
                cls_score = F.sigmoid(cls_score.transpose([0, 2, 3, 1]))
                bbox_pred = self.distribution_project(
                    bbox_pred.transpose([0, 2, 3, 1])) * self.fpn_stride[i]

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return (cls_logits_list, bboxes_reg_list)

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          img_shape,
                          scale_factor,
                          rescale=True,
                          cell_offset=0):
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred in zip(self.fpn_stride, cls_scores,
                                                bbox_preds):
            featmap_size = cls_score.shape[0:2]
            y, x = self.get_single_level_center_point(
                featmap_size, stride, cell_offset=cell_offset)
            center_points = paddle.stack([x, y], axis=-1)
            scores = cls_score.reshape([-1, self.cls_out_channels])

            if scores.shape[0] > self.nms_pre:
                max_scores = scores.max(axis=1)
                _, topk_inds = max_scores.topk(self.nms_pre)
                center_points = center_points.gather(topk_inds)
                bbox_pred = bbox_pred.gather(topk_inds)
                scores = scores.gather(topk_inds)

            bboxes = distance2bbox(
                center_points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        if rescale:
            # [h_scale, w_scale] to [w_scale, h_scale, w_scale, h_scale]
            im_scale = paddle.concat([scale_factor[::-1], scale_factor[::-1]])
            mlvl_bboxes /= im_scale
        mlvl_scores = paddle.concat(mlvl_scores)
        mlvl_scores = mlvl_scores.transpose([1, 0])
        return mlvl_bboxes, mlvl_scores

    def decode(self, cls_scores, bbox_preds, im_shape, scale_factor,
               cell_offset):
        batch_bboxes = []
        batch_scores = []
        batch_size = cls_scores[0].shape[0]
        for img_id in range(batch_size):
            num_levels = len(cls_scores)
            cls_score_list = [cls_scores[i][img_id] for i in range(num_levels)]
            bbox_pred_list = [
                bbox_preds[i].reshape([batch_size, -1, 4])[img_id]
                for i in range(num_levels)
            ]
            bboxes, scores = self.get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                im_shape[img_id],
                scale_factor[img_id],
                cell_offset=cell_offset)
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = paddle.stack(batch_bboxes, axis=0)
        batch_scores = paddle.stack(batch_scores, axis=0)

        return batch_bboxes, batch_scores

    def post_process(self, gfl_head_outs, im_shape, scale_factor):
        cls_scores, bboxes_reg = gfl_head_outs
        bboxes, score = self.decode(cls_scores, bboxes_reg, im_shape,
                                    scale_factor, self.cell_offset)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score)
        return bbox_pred, bbox_num
