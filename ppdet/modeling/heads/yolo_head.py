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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

import math
import numpy as np
from ..initializer import bias_init_with_prob, constant_
from ..backbones.csp_darknet import BaseConv, DWConv
from ..losses import IouLoss
from ppdet.modeling.assigners.simota_assigner import SimOTAAssigner
from ppdet.modeling.bbox_utils import bbox_overlaps
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['YOLOv3Head', 'YOLOXHead']


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


@register
class YOLOv3Head(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 data_format='NCHW'):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format

        self.yolo_outputs = []
        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            if self.iou_aware:
                y = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.reshape((b, na, no, h * w))
                    ioup = ioup.reshape((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = F.sigmoid(ioup)
                    obj = F.sigmoid(obj)
                    obj_t = (obj**(1 - self.iou_aware_factor)) * (
                        ioup**self.iou_aware_factor)
                    obj_t = _de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                    y_t = y_t.reshape((b, c, h, w))
                    y.append(y_t)
                return y
            else:
                return yolo_outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }


@register
class YOLOXHead(nn.Layer):
    __shared__ = ['num_classes', 'width_mult', 'act', 'trt', 'exclude_nms']
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 num_classes=80,
                 width_mult=1.0,
                 depthwise=False,
                 in_channels=[256, 512, 1024],
                 feat_channels=256,
                 fpn_strides=(8, 16, 32),
                 l1_epoch=285,
                 act='silu',
                 assigner=SimOTAAssigner(use_vfl=False),
                 nms='MultiClassNMS',
                 loss_weight={
                     'cls': 1.0,
                     'obj': 1.0,
                     'iou': 5.0,
                     'l1': 1.0,
                 },
                 trt=False,
                 exclude_nms=False):
        super(YOLOXHead, self).__init__()
        self._dtype = paddle.framework.get_default_dtype()
        self.num_classes = num_classes
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        feat_channels = int(feat_channels * width_mult)
        self.fpn_strides = fpn_strides
        self.l1_epoch = l1_epoch
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.loss_weight = loss_weight
        self.iou_loss = IouLoss(loss_weight=1.0)  # default loss_weight 2.5

        ConvBlock = DWConv if depthwise else BaseConv

        self.stem_conv = nn.LayerList()
        self.conv_cls = nn.LayerList()
        self.conv_reg = nn.LayerList()  # reg [x,y,w,h] + obj
        for in_c in self.in_channels:
            self.stem_conv.append(BaseConv(in_c, feat_channels, 1, 1, act=act))

            self.conv_cls.append(
                nn.Sequential(* [
                    ConvBlock(
                        feat_channels, feat_channels, 3, 1, act=act), ConvBlock(
                            feat_channels, feat_channels, 3, 1, act=act),
                    nn.Conv2D(
                        feat_channels,
                        self.num_classes,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
                ]))

            self.conv_reg.append(
                nn.Sequential(* [
                    ConvBlock(
                        feat_channels, feat_channels, 3, 1, act=act),
                    ConvBlock(
                        feat_channels, feat_channels, 3, 1, act=act),
                    nn.Conv2D(
                        feat_channels,
                        4 + 1,  # reg [x,y,w,h] + obj
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
                ]))

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = paddle.full([5], math.log(5.), dtype=self._dtype)
        bias_reg[:2] = 0.
        bias_reg[-1] = bias_cls
        for cls_, reg_ in zip(self.conv_cls, self.conv_reg):
            constant_(cls_[-1].weight)
            constant_(cls_[-1].bias, bias_cls)
            constant_(reg_[-1].weight)
            reg_[-1].bias.set_value(bias_reg)

    def _generate_anchor_point(self, feat_sizes, strides, offset=0.):
        anchor_points, stride_tensor = [], []
        num_anchors_list = []
        for feat_size, stride in zip(feat_sizes, strides):
            h, w = feat_size
            x = (paddle.arange(w) + offset) * stride
            y = (paddle.arange(h) + offset) * stride
            y, x = paddle.meshgrid(y, x)
            anchor_points.append(paddle.stack([x, y], axis=-1).reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [len(anchor_points[-1]), 1], stride, dtype=self._dtype))
            num_anchors_list.append(len(anchor_points[-1]))
        anchor_points = paddle.concat(anchor_points).astype(self._dtype)
        anchor_points.stop_gradient = True
        stride_tensor = paddle.concat(stride_tensor)
        stride_tensor.stop_gradient = True
        return anchor_points, stride_tensor, num_anchors_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        feat_sizes = [[f.shape[-2], f.shape[-1]] for f in feats]
        cls_score_list, reg_pred_list = [], []
        obj_score_list = []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_logit = self.conv_cls[i](feat)
            reg_pred = self.conv_reg[i](feat)
            # cls prediction
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            # reg prediction
            reg_xywh, obj_logit = paddle.split(reg_pred, [4, 1], axis=1)
            reg_xywh = reg_xywh.flatten(2).transpose([0, 2, 1])
            reg_pred_list.append(reg_xywh)
            # obj prediction
            obj_score = F.sigmoid(obj_logit)
            obj_score_list.append(obj_score.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_pred_list = paddle.concat(reg_pred_list, axis=1)
        obj_score_list = paddle.concat(obj_score_list, axis=1)

        # bbox decode
        anchor_points, stride_tensor, _ =\
            self._generate_anchor_point(feat_sizes, self.fpn_strides)
        reg_xy, reg_wh = paddle.split(reg_pred_list, 2, axis=-1)
        reg_xy += (anchor_points / stride_tensor)
        reg_wh = paddle.exp(reg_wh) * 0.5
        bbox_pred_list = paddle.concat(
            [reg_xy - reg_wh, reg_xy + reg_wh], axis=-1)

        if self.training:
            anchor_points, stride_tensor, num_anchors_list =\
                self._generate_anchor_point(feat_sizes, self.fpn_strides, 0.5)
            yolox_losses = self.get_loss([
                cls_score_list, bbox_pred_list, obj_score_list, anchor_points,
                stride_tensor, num_anchors_list
            ], targets)
            return yolox_losses
        else:
            pred_scores = (cls_score_list * obj_score_list).sqrt()
            return pred_scores, bbox_pred_list, stride_tensor

    def get_loss(self, head_outs, targets):
        pred_cls, pred_bboxes, pred_obj,\
        anchor_points, stride_tensor, num_anchors_list = head_outs
        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']
        pred_scores = (pred_cls * pred_obj).sqrt()
        # label assignment
        center_and_strides = paddle.concat(
            [anchor_points, stride_tensor, stride_tensor], axis=-1)
        pos_num_list, label_list, bbox_target_list = [], [], []
        for pred_score, pred_bbox, gt_box, gt_label in zip(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor, gt_bboxes, gt_labels):
            pos_num, label, _, bbox_target = self.assigner(
                pred_score, center_and_strides, pred_bbox, gt_box, gt_label)
            pos_num_list.append(pos_num)
            label_list.append(label)
            bbox_target_list.append(bbox_target)
        labels = paddle.to_tensor(np.stack(label_list, axis=0))
        bbox_targets = paddle.to_tensor(np.stack(bbox_target_list, axis=0))
        bbox_targets /= stride_tensor  # rescale bbox

        # 1. obj score loss
        mask_positive = (labels != self.num_classes)
        loss_obj = F.binary_cross_entropy(
            pred_obj,
            mask_positive.astype(pred_obj.dtype).unsqueeze(-1),
            reduction='sum')

        num_pos = sum(pos_num_list)

        if num_pos > 0:
            num_pos = paddle.to_tensor(num_pos, dtype=self._dtype).clip(min=1)
            loss_obj /= num_pos

            # 2. iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                bbox_targets, bbox_mask).reshape([-1, 4])
            bbox_iou = bbox_overlaps(pred_bboxes_pos, assigned_bboxes_pos)
            bbox_iou = paddle.diag(bbox_iou)

            loss_iou = self.iou_loss(
                pred_bboxes_pos.split(
                    4, axis=-1),
                assigned_bboxes_pos.split(
                    4, axis=-1))
            loss_iou = loss_iou.sum() / num_pos

            # 3. cls loss
            cls_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.num_classes])
            pred_cls_pos = paddle.masked_select(
                pred_cls, cls_mask).reshape([-1, self.num_classes])
            assigned_cls_pos = paddle.masked_select(labels, mask_positive)
            assigned_cls_pos = F.one_hot(assigned_cls_pos,
                                         self.num_classes + 1)[..., :-1]
            assigned_cls_pos *= bbox_iou.unsqueeze(-1)
            loss_cls = F.binary_cross_entropy(
                pred_cls_pos, assigned_cls_pos, reduction='sum')
            loss_cls /= num_pos

            # 4. l1 loss
            if targets['epoch_id'] >= self.l1_epoch:
                loss_l1 = F.l1_loss(
                    pred_bboxes_pos, assigned_bboxes_pos, reduction='sum')
                loss_l1 /= num_pos
            else:
                loss_l1 = paddle.zeros([])
                loss_l1.stop_gradient = False
        else:
            loss_cls = paddle.zeros([])
            loss_iou = paddle.zeros([])
            loss_l1 = paddle.zeros([])
            loss_cls.stop_gradient = False
            loss_iou.stop_gradient = False
            loss_l1.stop_gradient = False

        loss = self.loss_weight['obj'] * loss_obj + \
               self.loss_weight['cls'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou

        if targets['epoch_id'] >= self.l1_epoch:
            loss += (self.loss_weight['l1'] * loss_l1)

        yolox_losses = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_obj': loss_obj,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
        }
        return yolox_losses

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_bboxes, stride_tensor = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])
        pred_bboxes *= stride_tensor
        # scale bbox to origin image
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
