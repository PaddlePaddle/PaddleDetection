# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from .fcos_head import ScaleReg
from ..initializer import bias_init_with_prob, constant_, normal_
from ..ops import get_act_fn, anchor_generator
from ..rbox_utils import box2corners
from ..losses import ProbIoULoss
import numpy as np

__all__ = ['FCOSRHead']


def trunc_div(a, b):
    ipt = paddle.divide(a, b)
    sign_ipt = paddle.sign(ipt)
    abs_ipt = paddle.abs(ipt)
    abs_ipt = paddle.floor(abs_ipt)
    out = paddle.multiply(sign_ipt, abs_ipt)
    return out


def fmod(a, b):
    return a - trunc_div(a, b) * b


def fmod_eval(a, b):
    return a - a.divide(b).cast(paddle.int32).cast(paddle.float32) * b


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 norm_cfg={'name': 'gn',
                           'num_groups': 32},
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        norm_type = norm_cfg['name']
        if norm_type in ['sync_bn', 'bn']:
            self.norm = nn.BatchNorm2D(
                ch_out,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        else:
            groups = norm_cfg.get('num_groups', 1)
            self.norm = nn.GroupNorm(
                num_groups=groups,
                num_channels=ch_out,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


@register
class FCOSRHead(nn.Layer):
    """ FCOSR Head, refer to https://arxiv.org/abs/2111.10780 for details """

    __shared__ = ['num_classes', 'trt']
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 num_classes=15,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=4,
                 act='relu',
                 fpn_strides=[4, 8, 16, 32, 64],
                 trt=False,
                 loss_weight={'class': 1.0,
                              'probiou': 1.0},
                 norm_cfg={'name': 'gn',
                           'num_groups': 32},
                 assigner='FCOSRAssigner',
                 nms='MultiClassNMS'):

        super(FCOSRHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.stacked_convs = stacked_convs
        self.loss_weight = loss_weight
        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.probiou_loss = ProbIoULoss(mode='l1')
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.trt = trt
        self.loss_weight = loss_weight
        self.assigner = assigner
        self.nms = nms
        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        for i in range(self.stacked_convs):
            self.stem_cls.append(
                ConvBNLayer(
                    self.in_channels[i],
                    feat_channels,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act=act))
            self.stem_reg.append(
                ConvBNLayer(
                    self.in_channels[i],
                    feat_channels,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act=act))

        self.scales = nn.LayerList(
            [ScaleReg() for _ in range(len(fpn_strides))])

        # prediction
        self.pred_cls = nn.Conv2D(feat_channels, self.num_classes, 3, padding=1)

        self.pred_xy = nn.Conv2D(feat_channels, 2, 3, padding=1)

        self.pred_wh = nn.Conv2D(feat_channels, 2, 3, padding=1)

        self.pred_angle = nn.Conv2D(feat_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for cls_, reg_ in zip(self.stem_cls, self.stem_reg):
            normal_(cls_.conv.weight, std=0.01)
            normal_(reg_.conv.weight, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_(self.pred_cls.weight, std=0.01)
        constant_(self.pred_cls.bias, bias_cls)
        normal_(self.pred_xy.weight, std=0.01)
        normal_(self.pred_wh.weight, std=0.01)
        normal_(self.pred_angle.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _generate_anchors(self, feats):
        if self.trt:
            anchor_points = []
            for feat, stride in zip(feats, self.fpn_strides):
                _, _, h, w = paddle.shape(feat)
                anchor, _ = anchor_generator(
                    feat,
                    stride * 4,
                    1.0, [1.0, 1.0, 1.0, 1.0], [stride, stride],
                    offset=0.5)
                x1, y1, x2, y2 = paddle.split(anchor, 4, axis=-1)
                xc = (x1 + x2 + 1) / 2
                yc = (y1 + y2 + 1) / 2
                anchor_point = paddle.concat(
                    [xc, yc], axis=-1).reshape((1, h * w, 2))
                anchor_points.append(anchor_point)
            anchor_points = paddle.concat(anchor_points, axis=1)
            return anchor_points, None, None
        else:
            anchor_points = []
            stride_tensor = []
            num_anchors_list = []
            for feat, stride in zip(feats, self.fpn_strides):
                _, _, h, w = paddle.shape(feat)
                shift_x = (paddle.arange(end=w) + 0.5) * stride
                shift_y = (paddle.arange(end=h) + 0.5) * stride
                shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
                anchor_point = paddle.cast(
                    paddle.stack(
                        [shift_x, shift_y], axis=-1), dtype='float32')
                anchor_points.append(anchor_point.reshape([1, -1, 2]))
                stride_tensor.append(
                    paddle.full(
                        [1, h * w, 1], stride, dtype='float32'))
                num_anchors_list.append(h * w)
            anchor_points = paddle.concat(anchor_points, axis=1)
            stride_tensor = paddle.concat(stride_tensor, axis=1)
            return anchor_points, stride_tensor, num_anchors_list

    def forward(self, feats, target=None):
        if self.training:
            return self.forward_train(feats, target)
        else:
            return self.forward_eval(feats, target)

    def forward_train(self, feats, target=None):
        anchor_points, stride_tensor, num_anchors_list = self._generate_anchors(
            feats)
        cls_pred_list, reg_pred_list = [], []
        for stride, feat, scale in zip(self.fpn_strides, feats, self.scales):
            # cls
            cls_feat = feat
            for cls_layer in self.stem_cls:
                cls_feat = cls_layer(cls_feat)
            cls_pred = F.sigmoid(self.pred_cls(cls_feat))
            cls_pred_list.append(cls_pred.flatten(2).transpose((0, 2, 1)))
            # reg
            reg_feat = feat
            for reg_layer in self.stem_reg:
                reg_feat = reg_layer(reg_feat)

            reg_xy = scale(self.pred_xy(reg_feat)) * stride
            reg_wh = F.elu(scale(self.pred_wh(reg_feat)) + 1.) * stride
            reg_angle = self.pred_angle(reg_feat)
            reg_angle = fmod(reg_angle, self.half_pi)
            reg_pred = paddle.concat([reg_xy, reg_wh, reg_angle], axis=1)
            reg_pred_list.append(reg_pred.flatten(2).transpose((0, 2, 1)))

        cls_pred_list = paddle.concat(cls_pred_list, axis=1)
        reg_pred_list = paddle.concat(reg_pred_list, axis=1)

        return self.get_loss([
            cls_pred_list, reg_pred_list, anchor_points, stride_tensor,
            num_anchors_list
        ], target)

    def forward_eval(self, feats, target=None):
        cls_pred_list, reg_pred_list = [], []
        anchor_points, _, _ = self._generate_anchors(feats)
        for stride, feat, scale in zip(self.fpn_strides, feats, self.scales):
            b, _, h, w = paddle.shape(feat)
            # cls
            cls_feat = feat
            for cls_layer in self.stem_cls:
                cls_feat = cls_layer(cls_feat)
            cls_pred = F.sigmoid(self.pred_cls(cls_feat))
            cls_pred_list.append(cls_pred.reshape([b, self.num_classes, h * w]))
            # reg
            reg_feat = feat
            for reg_layer in self.stem_reg:
                reg_feat = reg_layer(reg_feat)

            reg_xy = scale(self.pred_xy(reg_feat)) * stride
            reg_wh = F.elu(scale(self.pred_wh(reg_feat)) + 1.) * stride
            reg_angle = self.pred_angle(reg_feat)
            reg_angle = fmod_eval(reg_angle, self.half_pi)
            reg_pred = paddle.concat([reg_xy, reg_wh, reg_angle], axis=1)
            reg_pred = reg_pred.reshape([b, 5, h * w]).transpose((0, 2, 1))
            reg_pred_list.append(reg_pred)

        cls_pred_list = paddle.concat(cls_pred_list, axis=2)
        reg_pred_list = paddle.concat(reg_pred_list, axis=1)
        reg_pred_list = self._bbox_decode(anchor_points, reg_pred_list)
        return cls_pred_list, reg_pred_list

    def _bbox_decode(self, points, reg_pred_list):
        xy, wha = paddle.split(reg_pred_list, [2, 3], axis=-1)
        xy = xy + points
        return paddle.concat([xy, wha], axis=-1)

    def _box2corners(self, pred_bboxes):
        """ convert (x, y, w, h, angle) to (x1, y1, x2, y2, x3, y3, x4, y4)

        Args:
            pred_bboxes (Tensor): [B, N, 5]
        
        Returns:
            polys (Tensor): [B, N, 8]
        """
        x, y, w, h, angle = paddle.split(pred_bboxes, 5, axis=-1)
        cos_a_half = paddle.cos(angle) * 0.5
        sin_a_half = paddle.sin(angle) * 0.5
        w_x = cos_a_half * w
        w_y = sin_a_half * w
        h_x = -sin_a_half * h
        h_y = cos_a_half * h
        return paddle.concat(
            [
                x + w_x + h_x, y + w_y + h_y, x - w_x + h_x, y - w_y + h_y,
                x - w_x - h_x, y - w_y - h_y, x + w_x - h_x, y + w_y - h_y
            ],
            axis=-1)

    def get_loss(self, head_outs, gt_meta):
        cls_pred_list, reg_pred_list, anchor_points, stride_tensor, num_anchors_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_rboxes = gt_meta['gt_rbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # decode
        pred_rboxes = self._bbox_decode(anchor_points, reg_pred_list)
        # label assignment
        assigned_labels, assigned_rboxes, assigned_scores = \
            self.assigner(
                anchor_points,
                stride_tensor,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                gt_rboxes,
                pad_gt_mask,
                self.num_classes,
                pred_rboxes
            )

        # reg_loss
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum().item()
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 5])
            pred_rboxes_pos = paddle.masked_select(pred_rboxes,
                                                   bbox_mask).reshape([-1, 5])
            assigned_rboxes_pos = paddle.masked_select(
                assigned_rboxes, bbox_mask).reshape([-1, 5])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).reshape([-1])
            avg_factor = bbox_weight.sum()
            loss_probiou = self.probiou_loss(pred_rboxes_pos,
                                             assigned_rboxes_pos)
            loss_probiou = paddle.sum(loss_probiou * bbox_weight) / avg_factor
        else:
            loss_probiou = pred_rboxes.sum() * 0.

        avg_factor = max(num_pos, 1.0)
        # cls_loss
        loss_cls = self._qfocal_loss(
            cls_pred_list, assigned_scores, reduction='sum')
        loss_cls = loss_cls / avg_factor

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['probiou'] * loss_probiou
        out_dict = {
            'loss': loss,
            'loss_probiou': loss_probiou,
            'loss_cls': loss_cls
        }
        return out_dict

    @staticmethod
    def _qfocal_loss(score, label, gamma=2.0, reduction='sum'):
        weight = (score - label).pow(gamma)
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction=reduction)
        return loss

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_rboxes = head_outs
        # [B, N, 5] -> [B, N, 4, 2] -> [B, N, 8]
        pred_rboxes = self._box2corners(pred_rboxes)
        # scale bbox to origin
        scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
        scale_factor = paddle.concat(
            [
                scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x,
                scale_y
            ],
            axis=-1).reshape([-1, 1, 8])
        pred_rboxes /= scale_factor
        bbox_pred, bbox_num, before_nms_indexes = self.nms(pred_rboxes,
                                                           pred_scores)
        return bbox_pred, bbox_num, before_nms_indexes
