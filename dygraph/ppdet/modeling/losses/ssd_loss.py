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
import numpy as np
from ppdet.core.workspace import register
from ..ops import bipartite_match, box_coder, iou_similarity

__all__ = ['SSDLoss']


@register
class SSDLoss(nn.Layer):
    def __init__(self,
                 match_type='per_prediction',
                 overlap_threshold=0.5,
                 neg_pos_ratio=3.0,
                 neg_overlap=0.5,
                 loc_loss_weight=1.0,
                 conf_loss_weight=1.0):
        super(SSDLoss, self).__init__()
        self.match_type = match_type
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_overlap = neg_overlap
        self.loc_loss_weight = loc_loss_weight
        self.conf_loss_weight = conf_loss_weight

    def _label_target_assign(self,
                             gt_label,
                             matched_indices,
                             neg_mask=None,
                             mismatch_value=0):
        gt_label = gt_label.numpy()
        matched_indices = matched_indices.numpy()
        if neg_mask is not None:
            neg_mask = neg_mask.numpy()

        batch_size, num_priors = matched_indices.shape
        trg_lbl = np.ones((batch_size, num_priors, 1)).astype('int32')
        trg_lbl *= mismatch_value
        trg_lbl_wt = np.zeros((batch_size, num_priors, 1)).astype('float32')

        for i in range(batch_size):
            col_ids = np.where(matched_indices[i] > -1)
            col_val = matched_indices[i][col_ids]
            trg_lbl[i][col_ids] = gt_label[i][col_val]
            trg_lbl_wt[i][col_ids] = 1.0

        if neg_mask is not None:
            trg_lbl_wt += neg_mask[:, :, np.newaxis]

        return paddle.to_tensor(trg_lbl), paddle.to_tensor(trg_lbl_wt)

    def _bbox_target_assign(self, encoded_box, matched_indices):
        encoded_box = encoded_box.numpy()
        matched_indices = matched_indices.numpy()

        batch_size, num_priors = matched_indices.shape
        trg_bbox = np.zeros((batch_size, num_priors, 4)).astype('float32')
        trg_bbox_wt = np.zeros((batch_size, num_priors, 1)).astype('float32')

        for i in range(batch_size):
            col_ids = np.where(matched_indices[i] > -1)
            col_val = matched_indices[i][col_ids]
            for v, c in zip(col_val.tolist(), col_ids[0]):
                trg_bbox[i][c] = encoded_box[i][v][c]
            trg_bbox_wt[i][col_ids] = 1.0

        return paddle.to_tensor(trg_bbox), paddle.to_tensor(trg_bbox_wt)

    def _mine_hard_example(self,
                           conf_loss,
                           matched_indices,
                           matched_dist,
                           neg_pos_ratio=3.0,
                           neg_overlap=0.5):
        pos = (matched_indices > -1).astype(conf_loss.dtype)
        num_pos = pos.sum(axis=1, keepdim=True)
        neg = (matched_dist < neg_overlap).astype(conf_loss.dtype)

        conf_loss = conf_loss * (1.0 - pos) * neg
        loss_idx = conf_loss.argsort(axis=1, descending=True)
        idx_rank = loss_idx.argsort(axis=1)
        num_negs = []
        for i in range(matched_indices.shape[0]):
            cur_idx = loss_idx[i]
            cur_num_pos = num_pos[i]
            num_neg = paddle.clip(cur_num_pos * neg_pos_ratio, max=pos.shape[1])
            num_negs.append(num_neg)
        num_neg = paddle.stack(num_negs, axis=0).expand_as(idx_rank)
        neg_mask = (idx_rank < num_neg).astype(conf_loss.dtype)
        return neg_mask

    def forward(self, boxes, scores, gt_box, gt_class, anchors):
        boxes = paddle.concat(boxes, axis=1)
        scores = paddle.concat(scores, axis=1)
        prior_boxes = paddle.concat(anchors, axis=0)
        gt_label = gt_class.unsqueeze(-1)
        batch_size, num_priors = scores.shape[:2]
        num_classes = scores.shape[-1] - 1

        def _reshape_to_2d(x):
            return paddle.flatten(x, start_axis=2)

        # 1. Find matched bounding box by prior box.
        #   1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
        #   1.2 Compute matched bounding box by bipartite matching algorithm.
        matched_indices = []
        matched_dist = []
        for i in range(gt_box.shape[0]):
            iou = iou_similarity(gt_box[i], prior_boxes)
            matched_indice, matched_d = bipartite_match(iou, self.match_type,
                                                        self.overlap_threshold)
            matched_indices.append(matched_indice)
            matched_dist.append(matched_d)
        matched_indices = paddle.concat(matched_indices, axis=0)
        matched_indices.stop_gradient = True
        matched_dist = paddle.concat(matched_dist, axis=0)
        matched_dist.stop_gradient = True

        # 2. Compute confidence for mining hard examples
        # 2.1. Get the target label based on matched indices
        target_label, _ = self._label_target_assign(
            gt_label, matched_indices, mismatch_value=num_classes)
        confidence = _reshape_to_2d(scores)
        # 2.2. Compute confidence loss.
        # Reshape confidence to 2D tensor.
        target_label = _reshape_to_2d(target_label).astype('int64')
        conf_loss = F.softmax_with_cross_entropy(confidence, target_label)
        conf_loss = paddle.reshape(conf_loss, [batch_size, num_priors])

        # 3. Mining hard examples
        neg_mask = self._mine_hard_example(
            conf_loss,
            matched_indices,
            matched_dist,
            neg_pos_ratio=self.neg_pos_ratio,
            neg_overlap=self.neg_overlap)

        # 4. Assign classification and regression targets
        # 4.1. Encoded bbox according to the prior boxes.
        prior_box_var = paddle.to_tensor(
            np.array(
                [0.1, 0.1, 0.2, 0.2], dtype='float32')).reshape(
                    [1, 4]).expand_as(prior_boxes)
        encoded_bbox = []
        for i in range(gt_box.shape[0]):
            encoded_bbox.append(
                box_coder(
                    prior_box=prior_boxes,
                    prior_box_var=prior_box_var,
                    target_box=gt_box[i],
                    code_type='encode_center_size'))
        encoded_bbox = paddle.stack(encoded_bbox, axis=0)
        # 4.2. Assign regression targets
        target_bbox, target_loc_weight = self._bbox_target_assign(
            encoded_bbox, matched_indices)
        # 4.3. Assign classification targets
        target_label, target_conf_weight = self._label_target_assign(
            gt_label,
            matched_indices,
            neg_mask=neg_mask,
            mismatch_value=num_classes)

        # 5. Compute loss.
        # 5.1 Compute confidence loss.
        target_label = _reshape_to_2d(target_label).astype('int64')
        conf_loss = F.softmax_with_cross_entropy(confidence, target_label)

        target_conf_weight = _reshape_to_2d(target_conf_weight)
        conf_loss = conf_loss * target_conf_weight * self.conf_loss_weight

        # 5.2 Compute regression loss.
        location = _reshape_to_2d(boxes)
        target_bbox = _reshape_to_2d(target_bbox)

        loc_loss = F.smooth_l1_loss(location, target_bbox, reduction='none')
        loc_loss = paddle.sum(loc_loss, axis=-1, keepdim=True)
        target_loc_weight = _reshape_to_2d(target_loc_weight)
        loc_loss = loc_loss * target_loc_weight * self.loc_loss_weight

        # 5.3 Compute overall weighted loss.
        loss = conf_loss + loc_loss
        loss = paddle.reshape(loss, [batch_size, num_priors])
        loss = paddle.sum(loss, axis=1, keepdim=True)
        normalizer = paddle.sum(target_loc_weight)
        loss = paddle.sum(loss / normalizer)

        return loss
