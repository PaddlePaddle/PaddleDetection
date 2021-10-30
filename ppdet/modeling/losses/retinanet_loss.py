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
from ppdet.core.workspace import register

from ppdet.modeling.proposal_generator.target import label_box
from ppdet.modeling.bbox_utils import bbox2delta

__all__ = ["RetinaNetLoss"]

@register
class RetinaNetLoss(nn.Layer):
    def __init__(
        self,
        focal_loss_alpha,
        focal_loss_gamma,
        smoothl1_loss_delta,
        positive_thresh,
        negative_thresh,
        allow_low_quality=True,
        num_classes=80,
        weights=[1.0, 1.0, 1.0, 1.0]
    ):
        super(RetinaNetLoss, self).__init__()
        
        self.num_classes = num_classes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smoothl1_loss_delta = smoothl1_loss_delta
        self.positive_thresh = positive_thresh
        self.negative_thresh = negative_thresh
        self.allow_low_quality = allow_low_quality
        self.weights = weights

        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

    def label_anchors(self, anchors, gt):
        batch_gt_box = gt["gt_bbox"]
        batch_gt_class = gt["gt_class"]

        gt_labels_list = []
        gt_boxes_list = []

        for i in range(len(batch_gt_box)):
            gt_boxes = batch_gt_box[i]
            gt_classes = batch_gt_class[i].flatten()
            matches_idxs, match_labels = label_box(anchors,
                                                   gt_boxes,
                                                   self.positive_thresh,
                                                   self.negative_thresh,
                                                   self.allow_low_quality,
                                                   -1)

            if len(gt_boxes) > 0:
                matched_boxes_i = paddle.gather(gt_boxes, matches_idxs)
                matched_classes_i = paddle.gather(gt_classes, matches_idxs)
                matched_classes_i = paddle.where(match_labels == 0,
                                                 paddle.full_like(matched_classes_i, self.num_classes),
                                                 matched_classes_i)
                matched_classes_i = paddle.where(match_labels == -1,
                                                 paddle.full_like(matched_classes_i, -1),
                                                 matched_classes_i)
            else:
                matched_boxes_i = paddle.zeros_like(anchors)
                matched_classes_i = paddle.zeros_like(matches_idxs) + self.num_classes

            gt_boxes_list.append(matched_boxes_i)
            gt_labels_list.append(matched_classes_i)
        
        return gt_boxes_list, gt_labels_list

    def forward(self, anchors, preds, inputs):

        pred_scores_list, pred_boxes_list = preds

        p_s = paddle.concat(pred_scores_list, axis=1)
        p_b = paddle.concat(pred_boxes_list, axis=1)  # [N, R, 4]

        gt_boxes, gt_classes = self.label_anchors(anchors, inputs)
        bs = len(gt_classes)
        gt_labels = paddle.stack(gt_classes).reshape([-1])  # [N * R]

        valid_idx = paddle.nonzero(gt_labels >= 0)
        pos_mask = paddle.logical_and(gt_labels >= 0, gt_labels != self.num_classes)
        pos_idx = paddle.nonzero(pos_mask).flatten()
        num_pos = pos_idx.shape[0]

        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        p_s = paddle.reshape(p_s, [-1, self.num_classes])
        pred_logits = paddle.gather(p_s, valid_idx)

        gt_labels = F.one_hot(paddle.gather(gt_labels, valid_idx), num_classes=self.num_classes + 1)[
            :, :-1
        ]
        
        gt_labels.stop_gradient = True

        cls_loss = F.sigmoid_focal_loss(pred_logits,
                                        gt_labels,
                                        alpha=self.focal_loss_alpha, 
                                        gamma=self.focal_loss_gamma, 
                                        reduction='sum')

        gt_deltas_list = [
            bbox2delta(anchors, gt_boxes[i], self.weights) for i in range(len(gt_boxes))
        ]

        gt_deltas = paddle.concat(gt_deltas_list)
        gt_deltas = paddle.gather(gt_deltas, pos_idx)
        gt_deltas.stop_gradient = True

        p_b = paddle.reshape(p_b, [-1, 4])
        pred_deltas = paddle.gather(p_b, pos_idx)

        if self.smoothl1_loss_delta > 0:
            reg_loss = F.smooth_l1_loss(pred_deltas, gt_deltas, reduction="sum",  delta=self.smoothl1_loss_delta)
        else:
            reg_loss = F.l1_loss(pred_deltas, gt_deltas, reduction="sum")

        return {
            "cls_loss": cls_loss / self.loss_normalizer,
            "reg_loss": reg_loss / self.loss_normalizer
        }


def sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="sum"):

    assert reduction in ["sum", "mean"
                         ], f'do not support this {reduction} reduction?'

    p = F.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss