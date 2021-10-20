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

import paddle
import numpy as np
import paddle.nn.functional as F
import paddle.nn as nn

from ppdet.modeling.losses.varifocal_loss import varifocal_loss
from ppdet.modeling.bbox_utils import batch_bbox_overlaps
from ppdet.core.workspace import register


@register
class SimOTAAssigner(object):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        num_classes (int): The num_classes of dataset.
        use_vfl (int): Whether to use varifocal_loss when calculating the cost matrix.
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 cls_weight=1.0,
                 num_classes=80,
                 use_vfl=True):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.use_vfl = use_vfl

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.shape[0]

        repeated_x = priors[:, 0].unsqueeze(1).tile([1, num_gt])
        repeated_y = priors[:, 1].unsqueeze(1).tile([1, num_gt])
        repeated_stride_x = priors[:, 2].unsqueeze(1).tile([1, num_gt])
        repeated_stride_y = priors[:, 3].unsqueeze(1).tile([1, num_gt])

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = paddle.stack([l_, t_, r_, b_], axis=1)
        is_in_gts = deltas.min(axis=1) > 0
        is_in_gts_all = is_in_gts.sum(axis=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = paddle.stack([cl_, ct_, cr_, cb_], axis=1)
        is_in_cts = ct_deltas.min(axis=1) > 0
        is_in_cts_all = is_in_cts.sum(axis=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = paddle.logical_or(is_in_gts_all, is_in_cts_all)

        is_in_gts_or_centers_inds = paddle.nonzero(
            is_in_gts_or_centers).squeeze(1)

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = paddle.logical_and(
            paddle.gather(
                is_in_gts.cast('int'), is_in_gts_or_centers_inds,
                axis=0).cast('bool'),
            paddle.gather(
                is_in_cts.cast('int'), is_in_gts_or_centers_inds,
                axis=0).cast('bool'))
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = np.zeros_like(cost.numpy())
        # select candidate topk ious for dynamic-k calculation
        topk_ious, _ = paddle.topk(pairwise_ious, self.candidate_topk, axis=0)
        # calculate dynamic k for each gt
        dynamic_ks = paddle.clip(topk_ious.sum(0).cast('int'), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = paddle.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx.numpy()] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost = cost.numpy()
            cost_argmin = np.argmin(cost[prior_match_gt_mask, :], axis=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.copy()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious.numpy()).sum(1)[fg_mask_inboxes]

        matched_pred_ious = paddle.to_tensor(
            matched_pred_ious, place=pairwise_ious.place)
        matched_gt_inds = paddle.to_tensor(
            matched_gt_inds, place=pairwise_ious.place)

        return matched_pred_ious, matched_gt_inds, valid_mask

    def get_sample(self, assign_gt_inds, gt_bboxes):
        pos_inds = np.unique(np.nonzero(assign_gt_inds > 0)[0])
        neg_inds = np.unique(np.nonzero(assign_gt_inds == 0)[0])
        pos_assigned_gt_inds = assign_gt_inds[pos_inds] - 1

        if gt_bboxes.size == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.size == 0
            pos_gt_bboxes = np.empty_like(gt_bboxes).reshape(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.resize(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def __call__(self,
                 pred_scores,
                 priors,
                 decoded_bboxes,
                 gt_bboxes,
                 gt_labels,
                 eps=1e-7):
        """Assign gt to priors using SimOTA.
        TODO: add comment.
        Returns:
            assign_result: The assigned result.
        """

        INF = 100000000
        num_gt = gt_bboxes.shape[0]
        num_bboxes = decoded_bboxes.shape[0]

        # assign 0 by default
        assigned_gt_inds = paddle.full(
            (num_bboxes, ), 0, dtype=paddle.int64).numpy()
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = paddle.full(
                    (num_bboxes, ), -1, dtype=paddle.int64)
            return

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)

        valid_mask_inds = paddle.nonzero(valid_mask).squeeze(1)
        valid_decoded_bbox = decoded_bboxes[valid_mask_inds]
        valid_pred_scores = pred_scores[valid_mask_inds]
        num_valid = valid_decoded_bbox.shape[0]

        pairwise_ious = batch_bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        if self.use_vfl:
            gt_vfl_labels = gt_labels.squeeze(-1).unsqueeze(0).tile(
                [num_valid, 1]).reshape([-1])
            valid_pred_scores = valid_pred_scores.unsqueeze(1).tile(
                [1, num_gt, 1]).reshape([-1, self.num_classes])
            vfl_score = np.zeros(valid_pred_scores.shape)
            vfl_score[np.arange(0, vfl_score.shape[0]), gt_vfl_labels.numpy(
            )] = pairwise_ious.reshape([-1])
            vfl_score = paddle.to_tensor(vfl_score)
            losses_vfl = varifocal_loss(
                valid_pred_scores, vfl_score,
                use_sigmoid=False).reshape([num_valid, num_gt])
            losses_giou = batch_bbox_overlaps(
                valid_decoded_bbox, gt_bboxes, mode='giou')
            cost_matrix = (
                losses_vfl * self.cls_weight + losses_giou * self.iou_weight +
                paddle.logical_not(is_in_boxes_and_center).cast('float32') * INF
            )
        else:
            iou_cost = -paddle.log(pairwise_ious + eps)
            gt_onehot_label = (F.one_hot(
                gt_labels.squeeze(-1).cast(paddle.int64),
                pred_scores.shape[-1]).cast('float32').unsqueeze(0).tile(
                    [num_valid, 1, 1]))

            valid_pred_scores = valid_pred_scores.unsqueeze(1).tile(
                [1, num_gt, 1])
            cls_cost = F.binary_cross_entropy(
                valid_pred_scores, gt_onehot_label, reduction='none').sum(-1)

            cost_matrix = (
                cls_cost * self.cls_weight + iou_cost * self.iou_weight +
                paddle.logical_not(is_in_boxes_and_center).cast('float32') * INF
            )

        matched_pred_ious, matched_gt_inds, valid_mask = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask.numpy())

        # assign results
        gt_labels = gt_labels.numpy()
        priors = priors.numpy()
        matched_gt_inds = matched_gt_inds.numpy()
        gt_bboxes = gt_bboxes.numpy()

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = np.full((num_bboxes, ), self.num_classes)
        assigned_labels[valid_mask] = gt_labels.squeeze(-1)[matched_gt_inds]

        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds \
            = self.get_sample(assigned_gt_inds, gt_bboxes)

        num_cells = priors.shape[0]
        bbox_targets = np.zeros_like(priors)
        bbox_weights = np.zeros_like(priors)
        labels = np.ones([num_cells], dtype=np.int64) * self.num_classes
        label_weights = np.zeros([num_cells], dtype=np.float32)

        if len(pos_inds) > 0:
            pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if not np.any(gt_labels):
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels.squeeze(-1)[pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        pos_num = max(pos_inds.size, 1)

        return priors, labels, label_weights, bbox_targets, pos_num
