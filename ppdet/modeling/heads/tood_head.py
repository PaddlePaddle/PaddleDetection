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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant

from ppdet.core.workspace import register
from ..initializer import normal_, constant_, bias_init_with_prob
from ppdet.modeling.bbox_utils import bbox_center
from ..losses import GIoULoss
from paddle.vision.ops import deform_conv2d
from ppdet.modeling.layers import ConvNormLayer


class ScaleReg(nn.Layer):
    """
    Parameter for scaling the regression outputs.
    """

    def __init__(self, init_scale=1.):
        super(ScaleReg, self).__init__()
        self.scale_reg = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=init_scale)),
            dtype="float32")

    def forward(self, inputs):
        out = inputs * self.scale_reg
        return out


class TaskDecomposition(nn.Layer):
    """This code is based on
        https://github.com/fcjian/TOOD/blob/master/mmdet/models/dense_heads/tood_head.py
    """

    def __init__(
            self,
            feat_channels,
            stacked_convs,
            la_down_rate=8,
            norm_type='gn',
            norm_groups=32, ):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.norm_type = norm_type
        self.norm_groups = norm_groups
        self.in_channels = self.feat_channels * self.stacked_convs
        self.la_conv1 = nn.Conv2D(self.in_channels,
                                  self.in_channels // la_down_rate, 1)
        self.la_conv2 = nn.Conv2D(self.in_channels // la_down_rate,
                                  self.stacked_convs, 1)

        self.reduction_conv = ConvNormLayer(
            self.in_channels,
            self.feat_channels,
            filter_size=1,
            stride=1,
            norm_type=self.norm_type,
            norm_groups=self.norm_groups)

        self._init_weights()

    def _init_weights(self):
        normal_(self.la_conv1.weight, std=0.001)
        normal_(self.la_conv2.weight, std=0.001)

    def forward(self, feat, avg_feat=None):
        b, _, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = F.relu(self.la_conv1(avg_feat))
        weight = F.sigmoid(self.la_conv2(weight))

        # here new_conv_weight = layer_attention_weight * conv_weight
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape([b, 1, self.stacked_convs, 1]) * \
            self.reduction_conv.conv.weight.reshape(
            [1, self.feat_channels, self.stacked_convs, self.feat_channels])
        conv_weight = conv_weight.reshape(
            [b, self.feat_channels, self.in_channels])
        feat = feat.reshape([b, self.in_channels, h * w])
        feat = paddle.bmm(conv_weight, feat).reshape(
            [b, self.feat_channels, h, w])
        if self.norm_type is not None:
            feat = self.reduction_conv.norm(feat)
        feat = F.relu(feat)
        return feat


@register
class TOODHead(nn.Layer):
    """This code is based on
        https://github.com/fcjian/TOOD/blob/master/mmdet/models/dense_heads/tood_head.py
    """
    __inject__ = ['nms', 'static_assigner', 'assigner']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 feat_channels=256,
                 stacked_convs=6,
                 fpn_strides=(8, 16, 32, 64, 128),
                 grid_cell_scale=8,
                 grid_cell_offset=0.5,
                 norm_type='gn',
                 norm_groups=32,
                 static_assigner_epoch=4,
                 use_align_head=True,
                 loss_weight={
                     'class': 1.0,
                     'bbox': 1.0,
                     'iou': 2.0,
                 },
                 nms='MultiClassNMS',
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner'):
        super(TOODHead, self).__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.static_assigner_epoch = static_assigner_epoch
        self.use_align_head = use_align_head
        self.nms = nms
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.loss_weight = loss_weight
        self.giou_loss = GIoULoss()

        self.inter_convs = nn.LayerList()
        for i in range(self.stacked_convs):
            self.inter_convs.append(
                ConvNormLayer(
                    self.feat_channels,
                    self.feat_channels,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    norm_groups=norm_groups))

        self.cls_decomp = TaskDecomposition(
            self.feat_channels,
            self.stacked_convs,
            self.stacked_convs * 8,
            norm_type=norm_type,
            norm_groups=norm_groups)
        self.reg_decomp = TaskDecomposition(
            self.feat_channels,
            self.stacked_convs,
            self.stacked_convs * 8,
            norm_type=norm_type,
            norm_groups=norm_groups)

        self.tood_cls = nn.Conv2D(
            self.feat_channels, self.num_classes, 3, padding=1)
        self.tood_reg = nn.Conv2D(self.feat_channels, 4, 3, padding=1)

        if self.use_align_head:
            self.cls_prob_conv1 = nn.Conv2D(self.feat_channels *
                                            self.stacked_convs,
                                            self.feat_channels // 4, 1)
            self.cls_prob_conv2 = nn.Conv2D(
                self.feat_channels // 4, 1, 3, padding=1)
            self.reg_offset_conv1 = nn.Conv2D(self.feat_channels *
                                              self.stacked_convs,
                                              self.feat_channels // 4, 1)
            self.reg_offset_conv2 = nn.Conv2D(
                self.feat_channels // 4, 4 * 2, 3, padding=1)

        self.scales_regs = nn.LayerList([ScaleReg() for _ in self.fpn_strides])

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'feat_channels': input_shape[0].channels,
            'fpn_strides': [i.stride for i in input_shape],
        }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_(self.tood_cls.weight, std=0.01)
        constant_(self.tood_cls.bias, bias_cls)
        normal_(self.tood_reg.weight, std=0.01)

        if self.use_align_head:
            normal_(self.cls_prob_conv1.weight, std=0.01)
            normal_(self.cls_prob_conv2.weight, std=0.01)
            constant_(self.cls_prob_conv2.bias, bias_cls)
            normal_(self.reg_offset_conv1.weight, std=0.001)
            normal_(self.reg_offset_conv2.weight, std=0.001)
            constant_(self.reg_offset_conv2.bias)

    def _generate_anchors(self, feats):
        anchors, num_anchors_list = [], []
        stride_tensor_list = []
        for feat, stride in zip(feats, self.fpn_strides):
            _, _, h, w = feat.shape
            cell_half_size = self.grid_cell_scale * stride * 0.5
            shift_x = (paddle.arange(end=w) + self.grid_cell_offset) * stride
            shift_y = (paddle.arange(end=h) + self.grid_cell_offset) * stride
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor = paddle.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1)
            anchors.append(anchor.reshape([-1, 4]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor_list.append(
                paddle.full([num_anchors_list[-1], 1], stride))
        return anchors, num_anchors_list, stride_tensor_list

    @staticmethod
    def _batch_distance2bbox(points, distance, max_shapes=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): [B, l, 2]
            distance (Tensor): [B, l, 4]
            max_shapes (tuple): [B, 2], "h w" format, Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, :, 0] - distance[:, :, 0]
        y1 = points[:, :, 1] - distance[:, :, 1]
        x2 = points[:, :, 0] + distance[:, :, 2]
        y2 = points[:, :, 1] + distance[:, :, 3]
        bboxes = paddle.stack([x1, y1, x2, y2], -1)
        if max_shapes is not None:
            out_bboxes = []
            for bbox, max_shape in zip(bboxes, max_shapes):
                bbox[:, 0] = bbox[:, 0].clip(min=0, max=max_shape[1])
                bbox[:, 1] = bbox[:, 1].clip(min=0, max=max_shape[0])
                bbox[:, 2] = bbox[:, 2].clip(min=0, max=max_shape[1])
                bbox[:, 3] = bbox[:, 3].clip(min=0, max=max_shape[0])
                out_bboxes.append(bbox)
            out_bboxes = paddle.stack(out_bboxes)
            return out_bboxes
        return bboxes

    @staticmethod
    def _deform_sampling(feat, offset):
        """ Sampling the feature according to offset.
        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for for feature sampliing
        """
        # it is an equivalent implementation of bilinear interpolation
        # you can also use F.grid_sample instead
        c = feat.shape[1]
        weight = paddle.ones([c, 1, 1, 1])
        y = deform_conv2d(feat, offset, weight, deformable_groups=c, groups=c)
        return y

    def forward(self, feats):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        anchors, num_anchors_list, stride_tensor_list = self._generate_anchors(
            feats)
        cls_score_list, bbox_pred_list = [], []
        for feat, scale_reg, anchor, stride in zip(feats, self.scales_regs,
                                                   anchors, self.fpn_strides):
            b, _, h, w = feat.shape
            inter_feats = []
            for inter_conv in self.inter_convs:
                feat = F.relu(inter_conv(feat))
                inter_feats.append(feat)
            feat = paddle.concat(inter_feats, axis=1)

            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)

            # cls prediction and alignment
            cls_logits = self.tood_cls(cls_feat)
            if self.use_align_head:
                cls_prob = F.relu(self.cls_prob_conv1(feat))
                cls_prob = F.sigmoid(self.cls_prob_conv2(cls_prob))
                cls_score = (F.sigmoid(cls_logits) * cls_prob).sqrt()
            else:
                cls_score = F.sigmoid(cls_logits)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))

            # reg prediction and alignment
            reg_dist = scale_reg(self.tood_reg(reg_feat).exp())
            reg_dist = reg_dist.transpose([0, 2, 3, 1]).reshape([b, -1, 4])
            anchor_centers = bbox_center(anchor).unsqueeze(0) / stride
            reg_bbox = self._batch_distance2bbox(
                anchor_centers.tile([b, 1, 1]), reg_dist)
            if self.use_align_head:
                reg_bbox = reg_bbox.reshape([b, h, w, 4]).transpose(
                    [0, 3, 1, 2])
                reg_offset = F.relu(self.reg_offset_conv1(feat))
                reg_offset = self.reg_offset_conv2(reg_offset)
                bbox_pred = self._deform_sampling(reg_bbox, reg_offset)
                bbox_pred = bbox_pred.flatten(2).transpose([0, 2, 1])
            else:
                bbox_pred = reg_bbox

            if not self.training:
                bbox_pred *= stride
            bbox_pred_list.append(bbox_pred)
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        bbox_pred_list = paddle.concat(bbox_pred_list, axis=1)
        anchors = paddle.concat(anchors)
        anchors.stop_gradient = True
        stride_tensor_list = paddle.concat(stride_tensor_list).unsqueeze(0)
        stride_tensor_list.stop_gradient = True

        return cls_score_list, bbox_pred_list, anchors, num_anchors_list, stride_tensor_list

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_bboxes, anchors, num_anchors_list, stride_tensor_list = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor_list,
                bbox_center(anchors),
                gt_labels,
                gt_bboxes,
                bg_index=self.num_classes)
            alpha_l = -1

        # rescale bbox
        assigned_bboxes /= stride_tensor_list
        # classification loss
        loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha=alpha_l)
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.astype(paddle.float32).sum()
        # bbox regression loss
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            # iou loss
            loss_iou = self.giou_loss(pred_bboxes_pos,
                                      assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / bbox_weight.sum()
            # l1 loss
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
        else:
            loss_iou = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])

        loss_cls /= assigned_scores.sum().clip(min=1)
        loss = self.loss_weight['class'] * loss_cls + self.loss_weight[
            'iou'] * loss_iou

        return {
            'loss': loss,
            'loss_class': loss_cls,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1
        }

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_bboxes, _, _, _ = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])

        for i in range(len(pred_bboxes)):
            pred_bboxes[i, :, 0] = pred_bboxes[i, :, 0].clip(
                min=0, max=img_shape[i, 1])
            pred_bboxes[i, :, 1] = pred_bboxes[i, :, 1].clip(
                min=0, max=img_shape[i, 0])
            pred_bboxes[i, :, 2] = pred_bboxes[i, :, 2].clip(
                min=0, max=img_shape[i, 1])
            pred_bboxes[i, :, 3] = pred_bboxes[i, :, 3].clip(
                min=0, max=img_shape[i, 0])
        # scale bbox to origin
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num
