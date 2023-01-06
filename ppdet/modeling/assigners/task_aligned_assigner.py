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
#from ..bbox_utils import batch_iou_similarity
from ..bbox_utils import batch_iou_similarity, batch_dist, is_close_gt
from .utils import (gather_topk_anchors, check_points_inside_bboxes,
                    compute_max_iou_anchor)

__all__ = ['TaskAlignedAssigner']


def loss_weight_dw(bbox_cls_scores, gt_labels, center_prior_weights, ious,
                   inside_gt_bbox_mask, mask_positive):
    """
    return [B, L]
    """
    B, N, L = bbox_cls_scores.shape
    num_gts = gt_labels.shape[1]
    joint_conf = bbox_cls_scores
    #To more precisely estimate the consistency degree between cls and reg heads, we represent IoU score as an expentional function of the reg loss.
    # p_loc = paddle.exp(-reg_loss*5)
    #replace exp(-reg_loss*5) to iou
    p_loc = ious
    # p_cls = bbox_cls_scores[gt_labels] 
    p_cls = bbox_cls_scores
    p_pos = p_cls * p_loc

    p_neg_weight = paddle.ones_like(joint_conf)
    neg_metrics = paddle.zeros_like(ious).fill_(-1)
    alpha = 2
    t = lambda x: 1 / (0.5**alpha - 1) * x**alpha - 1 / (0.5**alpha - 1)
    if num_gts > 0:

        def normalize(x):
            x_ = t(x)
            t1 = x_.min()
            t2 = min(1., x_.max())
            y = (x_ - t1 + 1e-12) / (t2 - t1 + 1e-12)
            y[x < 0.5] = 1
            return y

        # import pdb;pdb.set_trace()
        p_neg_weight = normalize(ious * mask_positive).min(
            axis=1) * inside_gt_bbox_mask.max(axis=1)

    p_neg_weight = p_neg_weight.detach()
    # neg_avg_factor = (1 - p_neg_weight).sum()
    score = (joint_conf * (gt_labels > 0)).max(axis=1)
    p_neg_weight = p_neg_weight * score**2

    p_pos_weight = (paddle.exp(5 * p_pos) * p_pos * center_prior_weights) / (
        paddle.exp(3 * p_pos) * p_pos * center_prior_weights).sum(
            0, keepdim=True).clip(min=1e-12)
    p_pos_weight = (p_pos_weight * mask_positive).max(axis=1).detach()
    return p_pos_weight, p_neg_weight


@register
class TaskAlignedAssigner(nn.Layer):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @paddle.no_grad()
    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = paddle.full(
                [batch_size, num_anchors], bg_index, dtype='int32')
            assigned_bboxes = paddle.zeros([batch_size, num_anchors, 4])
            assigned_scores = paddle.zeros(
                [batch_size, num_anchors, num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = batch_iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.transpose([0, 2, 1])
        batch_ind = paddle.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        gt_labels_ind = paddle.stack(
            [batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)],
            axis=-1)
        bbox_cls_scores = paddle.gather_nd(pred_scores, gt_labels_ind)
        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(
            self.beta)

        # check the positive sample's center in gt, [B, n, L]
        #is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        is_in_gts = is_close_gt(anchor_points, gt_bboxes, num_anchors_list)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = paddle.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = paddle.gather(
            gt_labels.flatten(), assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = paddle.where(
            mask_positive_sum > 0, assigned_labels,
            paddle.full_like(assigned_labels, bg_index))

        assigned_bboxes = paddle.gather(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten(), axis=0)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = paddle.index_select(
            assigned_scores, paddle.to_tensor(ind), axis=-1)
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(axis=-1, keepdim=True)
        max_ious_per_instance = (ious * mask_positive).max(axis=-1,
                                                           keepdim=True)
        alignment_metrics = alignment_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(-2).unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores
