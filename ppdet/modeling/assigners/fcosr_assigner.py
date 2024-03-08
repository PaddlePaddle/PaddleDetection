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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.rbox_utils import box2corners, check_points_in_polys, paddle_gather

__all__ = ['FCOSRAssigner']

EPS = 1e-9


@register
class FCOSRAssigner(nn.Layer):
    """ FCOSR Assigner, refer to https://arxiv.org/abs/2111.10780 for details

    1. compute normalized gaussian distribution score and refined gaussian distribution score
    2. refer to ellipse center sampling, sample points whose normalized gaussian distribution score is greater than threshold
    3. refer to multi-level sampling, assign ground truth to feature map which follows two conditions.
        i). first, the ratio between the short edge of the target and the stride of the feature map is less than 2.
        ii). second, the long edge of minimum bounding rectangle of the target is larger than the acceptance range of feature map
    4. refer to fuzzy sample label assignment, the points satisfying 2 and 3 will be assigned to the ground truth according to gaussian distribution score
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 factor=12,
                 threshold=0.23,
                 boundary=[[-1, 128], [128, 320], [320, 10000]],
                 score_type='iou'):
        super(FCOSRAssigner, self).__init__()
        self.num_classes = num_classes
        self.factor = factor
        self.threshold = threshold
        self.boundary = [
            paddle.to_tensor(
                l, dtype=paddle.float32).reshape([1, 1, 2]) for l in boundary
        ]
        self.score_type = score_type

    def get_gaussian_distribution_score(self, points, gt_rboxes, gt_polys):
        # projecting points to coordinate system defined by each rbox
        # [B, N, 4, 2] -> 4 * [B, N, 1, 2]
        a, b, c, d = gt_polys.split(4, axis=2)
        # [1, L, 2] -> [1, 1, L, 2]
        points = points.unsqueeze(0)
        ab = b - a
        ad = d - a
        # [B, N, 5] -> [B, N, 2], [B, N, 2], [B, N, 1]
        xy, wh, angle = gt_rboxes.split([2, 2, 1], axis=-1)
        # [B, N, 2] -> [B, N, 1, 2]
        xy = xy.unsqueeze(2)
        # vector of points to center [B, N, L, 2]
        vec = points - xy
        # <ab, vec> = |ab| * |vec| * cos(theta) [B, N, L]
        vec_dot_ab = paddle.sum(vec * ab, axis=-1)
        # <ad, vec> = |ad| * |vec| * cos(theta) [B, N, L]
        vec_dot_ad = paddle.sum(vec * ad, axis=-1)
        # norm_ab [B, N, L]
        norm_ab = paddle.sum(ab * ab, axis=-1).sqrt()
        # norm_ad [B, N, L]
        norm_ad = paddle.sum(ad * ad, axis=-1).sqrt()
        # min(h, w), [B, N, 1]
        min_edge = paddle.min(wh, axis=-1, keepdim=True)
        # delta_x, delta_y [B, N, L]
        delta_x = vec_dot_ab.pow(2) / (norm_ab.pow(3) * min_edge + EPS)
        delta_y = vec_dot_ad.pow(2) / (norm_ad.pow(3) * min_edge + EPS)
        # score [B, N, L]
        norm_score = paddle.exp(-0.5 * self.factor * (delta_x + delta_y))

        # simplified calculation
        sigma = min_edge / self.factor
        refined_score = norm_score / (2 * np.pi * sigma + EPS)
        return norm_score, refined_score

    def get_rotated_inside_mask(self, points, gt_polys, scores):
        inside_mask = check_points_in_polys(points, gt_polys)
        center_mask = scores >= self.threshold
        return (inside_mask & center_mask).cast(paddle.float32)

    def get_inside_range_mask(self, points, gt_bboxes, gt_rboxes, stride_tensor,
                              regress_range):
        # [1, L, 2] -> [1, 1, L, 2]
        points = points.unsqueeze(0)
        # [B, n, 4] -> [B, n, 1, 4]
        x1y1, x2y2 = gt_bboxes.unsqueeze(2).split(2, axis=-1)
        # [B, n, L, 2]
        lt = points - x1y1
        rb = x2y2 - points
        # [B, n, L, 4]
        ltrb = paddle.concat([lt, rb], axis=-1)
        # [B, n, L, 4] -> [B, n, L]
        inside_mask = paddle.min(ltrb, axis=-1) > EPS
        # regress_range [1, L, 2] -> [1, 1, L, 2]
        regress_range = regress_range.unsqueeze(0)
        # stride_tensor [1, L, 1] -> [1, 1, L]
        stride_tensor = stride_tensor.transpose((0, 2, 1))
        # fcos range
        # [B, n, L, 4] -> [B, n, L]
        ltrb_max = paddle.max(ltrb, axis=-1)
        # [1, 1, L, 2] -> [1, 1, L]
        low, high = regress_range[..., 0], regress_range[..., 1]
        # [B, n, L]
        regress_mask = (ltrb_max >= low) & (ltrb_max <= high)
        # mask for rotated
        # [B, n, 1]
        min_edge = paddle.min(gt_rboxes[..., 2:4], axis=-1, keepdim=True)
        # [B, n , L]
        rotated_mask = ((min_edge / stride_tensor) < 2.0) & (ltrb_max > high)
        mask = inside_mask & (regress_mask | rotated_mask)
        return mask.cast(paddle.float32)

    @paddle.no_grad()
    def forward(self,
                anchor_points,
                stride_tensor,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                gt_rboxes,
                pad_gt_mask,
                bg_index,
                pred_rboxes=None):
        r"""

        Args:
            anchor_points (Tensor, float32): pre-defined anchor points, shape(1, L, 2),
                    "x, y" format
            stride_tensor (Tensor, float32): stride tensor, shape (1, L, 1)
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            gt_rboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 5)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            pred_rboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 5)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_rboxes (Tensor): (B, L, 5)
            assigned_scores (Tensor): (B, L, C), if pred_rboxes is not None, then output ious
        """

        _, num_anchors, _ = anchor_points.shape
        batch_size, num_max_boxes, _ = gt_rboxes.shape
        if num_max_boxes == 0:
            assigned_labels = paddle.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_rboxes = paddle.zeros([batch_size, num_anchors, 5])
            assigned_scores = paddle.zeros(
                [batch_size, num_anchors, self.num_classes])
            return assigned_labels, assigned_rboxes, assigned_scores

        # get normalized gaussian distribution score and refined distribution score
        gt_polys = box2corners(gt_rboxes)
        score, refined_score = self.get_gaussian_distribution_score(
            anchor_points, gt_rboxes, gt_polys)
        inside_mask = self.get_rotated_inside_mask(anchor_points, gt_polys,
                                                   score)
        regress_ranges = []
        for num, bound in zip(num_anchors_list, self.boundary):
            regress_ranges.append(bound.tile((1, num, 1)))
        regress_ranges = paddle.concat(regress_ranges, axis=1)
        regress_mask = self.get_inside_range_mask(
            anchor_points, gt_bboxes, gt_rboxes, stride_tensor, regress_ranges)
        # [B, n, L]
        mask_positive = inside_mask * regress_mask * pad_gt_mask
        refined_score = refined_score * mask_positive - (1. - mask_positive)

        argmax_refined_score = refined_score.argmax(axis=-2)
        max_refined_score = refined_score.max(axis=-2)
        assigned_gt_index = argmax_refined_score

        # assigned target
        batch_ind = paddle.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + (batch_ind * num_max_boxes).astype(assigned_gt_index.dtype)
        assigned_labels = paddle.gather(
            gt_labels.flatten(), assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = paddle.where(
            max_refined_score > 0, assigned_labels,
            paddle.full_like(assigned_labels, bg_index))

        assigned_rboxes = paddle.gather(
            gt_rboxes.reshape([-1, 5]), assigned_gt_index.flatten(), axis=0)
        assigned_rboxes = assigned_rboxes.reshape([batch_size, num_anchors, 5])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = paddle.index_select(
            assigned_scores, paddle.to_tensor(ind), axis=-1)

        if self.score_type == 'gaussian':
            selected_scores = paddle_gather(
                score, 1, argmax_refined_score.unsqueeze(-2)).squeeze(-2)
            assigned_scores = assigned_scores * selected_scores.unsqueeze(-1)
        elif self.score_type == 'iou':
            assert pred_rboxes is not None, 'If score type is iou, pred_rboxes should not be None'
            from ext_op import matched_rbox_iou
            b, l = pred_rboxes.shape[:2]
            iou_score = matched_rbox_iou(
                pred_rboxes.reshape((-1, 5)), assigned_rboxes.reshape(
                    (-1, 5))).reshape((b, l, 1))
            assigned_scores = assigned_scores * iou_score

        return assigned_labels, assigned_rboxes, assigned_scores
