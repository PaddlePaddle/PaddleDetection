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
#
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ppdet.core.workspace import register, serializable
from ..losses.iou_loss import GIoULoss
from .utils import bbox_cxcywh_to_xyxy

__all__ = ['HungarianMatcher']


@register
@serializable
class HungarianMatcher(nn.Layer):
    __shared__ = ['use_focal_loss', 'with_mask', 'num_sample_points']

    def __init__(self,
                 matcher_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 1,
                     'dice': 1
                 },
                 use_focal_loss=False,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor|None): [b, query, h, w]
            gt_mask (List(Tensor)): list[[n, H, W]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = [len(a) for a in gt_class]
        if sum(num_gts) == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        logits = logits.detach()
        out_prob = F.sigmoid(logits.flatten(
            0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.detach().flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        out_prob = paddle.gather(out_prob, tgt_ids, axis=1)
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
                1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # Compute the L1 cost between boxes
        cost_bbox = (
            out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        giou_loss = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)
        cost_giou = giou_loss - 1

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + \
            self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        # Compute the mask cost and dice cost
        if self.with_mask:
            assert (masks is not None and gt_mask is not None,
                    'Make sure the input has `mask` and `gt_mask`')
            # all masks share the same set of points for efficient matching
            sample_points = paddle.rand([bs, 1, self.num_sample_points, 2])
            sample_points = 2.0 * sample_points - 1.0

            out_mask = F.grid_sample(
                masks.detach(), sample_points, align_corners=False).squeeze(-2)
            out_mask = out_mask.flatten(0, 1)

            tgt_mask = paddle.concat(gt_mask).unsqueeze(1)
            sample_points = paddle.concat([
                a.tile([b, 1, 1, 1]) for a, b in zip(sample_points, num_gts)
                if b > 0
            ])
            tgt_mask = F.grid_sample(
                tgt_mask, sample_points, align_corners=False).squeeze([1, 2])

            with paddle.amp.auto_cast(enable=False):
                # binary cross entropy cost
                pos_cost_mask = F.binary_cross_entropy_with_logits(
                    out_mask, paddle.ones_like(out_mask), reduction='none')
                neg_cost_mask = F.binary_cross_entropy_with_logits(
                    out_mask, paddle.zeros_like(out_mask), reduction='none')
                cost_mask = paddle.matmul(
                    pos_cost_mask, tgt_mask, transpose_y=True) + paddle.matmul(
                        neg_cost_mask, 1 - tgt_mask, transpose_y=True)
                cost_mask /= self.num_sample_points

                # dice cost
                out_mask = F.sigmoid(out_mask)
                numerator = 2 * paddle.matmul(
                    out_mask, tgt_mask, transpose_y=True)
                denominator = out_mask.sum(
                    -1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
                cost_dice = 1 - (numerator + 1) / (denominator + 1)

                C = C + self.matcher_coeff['mask'] * cost_mask + \
                    self.matcher_coeff['dice'] * cost_dice

        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]
        sizes = [a.shape[0] for a in gt_bbox]
        if hasattr(paddle.Tensor, "contiguous"):
            indices = [
                linear_sum_assignment(c.split(sizes, -1)[i].contiguous().numpy())
                for i, c in enumerate(C)
            ]
        else:
            indices = [
                linear_sum_assignment(c.split(sizes, -1)[i].numpy())
                for i, c in enumerate(C)
            ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
                j, dtype=paddle.int64)) for i, j in indices]
