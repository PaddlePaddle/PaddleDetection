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
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ..ops import iou_similarity
from ..bbox_utils import bbox2delta

__all__ = ['SSDLoss']


@register
class SSDLoss(nn.Layer):
    """
    SSDLoss

    Args:
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage.
        overlap_threshold (float32, optional): IoU threshold for negative bboxes
            and positive bboxes, 0.5 by default.
        neg_pos_ratio (float): The ratio of negative samples / positive samples.
        loc_loss_weight (float): The weight of loc_loss.
        conf_loss_weight (float): The weight of conf_loss.
        prior_box_var (list): Variances corresponding to prior box coord, [0.1,
            0.1, 0.2, 0.2] by default.
    """

    def __init__(self,
                 match_low_quality=True,
                 overlap_threshold=0.5,
                 neg_pos_ratio=3.0,
                 loc_loss_weight=1.0,
                 conf_loss_weight=1.0,
                 prior_box_var=[0.1, 0.1, 0.2, 0.2]):
        super(SSDLoss, self).__init__()
        self.match_low_quality = match_low_quality
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_loss_weight = loc_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.prior_box_var = [1. / a for a in prior_box_var]

    def _bipartite_match_for_batch(self, gt_bbox, gt_label, prior_boxes,
                                   mismatch_value):
        """
        Args:
            gt_bbox (Tensor): [B, N, 4]
            gt_label (Tensor): [B, N, 1]
            prior_boxes (Tensor): [A, 4]
            mismatch_value (int): Background class
        """
        batch_size, num_priors = gt_bbox.shape[0], prior_boxes.shape[0]
        ious = iou_similarity(gt_bbox.reshape((-1, 4)), prior_boxes).reshape(
            (batch_size, -1, num_priors))

        # Calculate the number of object per sample.
        num_object = (ious.sum(axis=-1) > 0).astype('int64').sum(axis=-1)

        # For each anchor, the max iou of all gts.
        anchor_max_iou, anchor_argmax_iou = ious.max(axis=1), ious.argmax(
            axis=1)
        # For each gt, the max iou of all anchors.
        gt_max_iou, gt_argmax_iou = ious.max(axis=2), ious.argmax(axis=2)

        # Gather target bbox and label according to 'anchor_argmax_iou' index.
        batch_ind = paddle.arange(
            0, batch_size, dtype='int64').unsqueeze(-1).tile([1, num_priors])
        anchor_argmax_iou = paddle.stack(
            [batch_ind, anchor_argmax_iou], axis=-1)
        targets_bbox = paddle.gather_nd(gt_bbox, anchor_argmax_iou)
        targets_label = paddle.gather_nd(gt_label, anchor_argmax_iou)
        # assign negative
        mismatch_value_tensor = paddle.full([batch_size, num_priors, 1],
                                            mismatch_value, 'int64')
        targets_label = paddle.where(
            anchor_max_iou.unsqueeze(-1) < self.overlap_threshold,
            mismatch_value_tensor, targets_label)

        # Low-quality matching will overwrite the assigned_gt_inds assigned
        # in last step. Thus, the assigned gt might not be the best one for
        # prediction.
        # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
        # bbox 1 will be assigned as the best target for bbox A in last step.
        # However, if GT bbox 2's gt_argmax_iou = A, bbox A's
        # targets_label will be overwritten to be bbox B.
        # This might be the reason that it is not used in ROI Heads.
        if self.match_low_quality:
            for i in range(batch_size):
                targets_bbox[i] = paddle.scatter(
                    targets_bbox[i], gt_argmax_iou[i, :int(num_object[i])],
                    gt_bbox[i, :int(num_object[i])])
                targets_label[i] = paddle.scatter(
                    targets_label[i], gt_argmax_iou[i, :int(num_object[i])],
                    gt_label[i, :int(num_object[i])])

        # encode box
        prior_boxes = prior_boxes.unsqueeze(0).tile([batch_size, 1, 1])
        targets_bbox = bbox2delta(
            prior_boxes.reshape([-1, 4]),
            targets_bbox.reshape([-1, 4]), self.prior_box_var)
        targets_bbox = targets_bbox.reshape([batch_size, -1, 4])

        return targets_bbox, targets_label

    def _mine_hard_example(self, conf_loss, targets_label, mismatch_value):
        pos = (targets_label != mismatch_value).astype(conf_loss.dtype)
        num_pos = pos.sum(axis=1, keepdim=True)
        neg = (targets_label == mismatch_value).astype(conf_loss.dtype)

        conf_loss = conf_loss.clone() * neg
        loss_idx = conf_loss.argsort(axis=1, descending=True)
        idx_rank = loss_idx.argsort(axis=1)
        num_negs = []
        for i in range(conf_loss.shape[0]):
            cur_num_pos = num_pos[i]
            num_neg = paddle.clip(
                cur_num_pos * self.neg_pos_ratio, max=pos.shape[1])
            num_negs.append(num_neg)
        num_neg = paddle.stack(num_negs).expand_as(idx_rank)
        neg_mask = (idx_rank < num_neg).astype(conf_loss.dtype)

        return (neg_mask + pos).astype('bool')

    def forward(self, boxes, scores, gt_bbox, gt_label, prior_boxes):
        boxes = paddle.concat(boxes, axis=1)
        scores = paddle.concat(scores, axis=1)
        gt_label = gt_label.unsqueeze(-1).astype('int64')
        prior_boxes = paddle.concat(prior_boxes, axis=0)
        num_classes = scores.shape[-1] - 1

        # Match bbox.
        # Index 'num_classes' is background.
        targets_bbox, targets_label = \
            self._bipartite_match_for_batch(gt_bbox, gt_label, prior_boxes, num_classes)
        targets_bbox.stop_gradient = True
        targets_label.stop_gradient = True

        # Compute regression loss.
        # Select positive samples.
        bbox_mask = paddle.tile(targets_label != num_classes, [1, 1, 4])
        location = paddle.masked_select(boxes, bbox_mask)
        targets_bbox = paddle.masked_select(targets_bbox, bbox_mask)
        loc_loss = F.smooth_l1_loss(location, targets_bbox, reduction='sum')
        loc_loss = loc_loss * self.loc_loss_weight

        # Compute confidence loss.
        conf_loss = F.softmax_with_cross_entropy(scores, targets_label)
        # Mining hard examples
        label_mask = self._mine_hard_example(
            conf_loss.squeeze(-1), targets_label.squeeze(-1), num_classes)
        conf_loss = paddle.masked_select(conf_loss, label_mask.unsqueeze(-1))
        conf_loss = conf_loss.sum() * self.conf_loss_weight

        # Compute overall weighted loss.
        normalizer = (targets_label != num_classes).astype('float32').sum()
        loss = (conf_loss + loc_loss) / normalizer

        return loss
