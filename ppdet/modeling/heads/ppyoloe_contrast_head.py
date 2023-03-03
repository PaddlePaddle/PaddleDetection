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

from ..initializer import bias_init_with_prob, constant_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.heads.ppyoloe_head import PPYOLOEHead

__all__ = ['PPYOLOEContrastHead']


@register
class PPYOLOEContrastHead(PPYOLOEHead):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms', 'contrast_loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 contrast_loss='SupContrast',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 for_distill=False):
        super().__init__(in_channels, num_classes, act, fpn_strides,
                         grid_cell_scale, grid_cell_offset, reg_max, reg_range,
                         static_assigner_epoch, use_varifocal_loss,
                         static_assigner, assigner, nms, eval_size, loss_weight,
                         trt, attn_conv, exclude_nms, exclude_post_process,
                         use_shared_conv, for_distill)

        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.contrast_loss = contrast_loss
        self.contrast_encoder = nn.LayerList()
        for in_c in self.in_channels:
            self.contrast_encoder.append(nn.Conv2D(in_c, 128, 3, padding=1))
        self._init_contrast_encoder()

    def _init_contrast_encoder(self):
        bias_en = bias_init_with_prob(0.01)
        for en_ in self.contrast_encoder:
            constant_(en_.weight)
            constant_(en_.bias, bias_en)

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        contrast_encoder_list = []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            contrast_logit = self.contrast_encoder[i](self.stem_cls[i](
                feat, avg_feat) + feat)
            contrast_encoder_list.append(
                contrast_logit.flatten(2).transpose([0, 2, 1]))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        contrast_encoder_list = paddle.concat(contrast_encoder_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, contrast_encoder_list, anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_contrast_encoder, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            if self.sm_use:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            else:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        # contrast loss
        loss_contrast = self.contrast_loss(pred_contrast_encoder.reshape([-1, pred_contrast_encoder.shape[-1]]), \
            assigned_labels.reshape([-1]), assigned_scores.max(-1).reshape([-1]))

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['contrast'] * loss_contrast

        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
            'loss_contrast': loss_contrast
        }
        return out_dict
