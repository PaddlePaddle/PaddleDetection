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

__all__ = ['HungarianMatcher', 'OVHungarianMatcher', 'OVHungarianMatcher_ori']


@register
@serializable
class HungarianMatcher(nn.Layer):
    __shared__ = ['use_focal_loss']

    def __init__(self,
                 matcher_coeff={'class': 1,
                                'bbox': 5,
                                'giou': 2},
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self, boxes, logits, gt_bbox, gt_class):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = sum(len(a) for a in gt_class)
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.sigmoid(logits.flatten(
            0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(
                    1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                    (1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = paddle.gather(
                pos_cost_class, tgt_ids, axis=1) - paddle.gather(
                neg_cost_class, tgt_ids, axis=1)
        else:
            cost_class = -paddle.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        cost_bbox = (
                out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]

        sizes = [a.shape[0] for a in gt_bbox]
        indices = [
            linear_sum_assignment(c.split(sizes, -1)[i].numpy())
            for i, c in enumerate(C)
        ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
            j, dtype=paddle.int64)) for i, j in indices]


@register
@serializable
class OVHungarianMatcher_ori(nn.Layer):
    __shared__ = ['use_focal_loss']

    def __init__(self,
                 matcher_coeff={'class': 3,
                                'bbox': 5,
                                'giou': 2},
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        The same as matcher without cost bbox
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(OVHungarianMatcher_ori, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    @paddle.no_grad()
    def forward(self, boxes, logits, gt_bbox, gt_class):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = sum(len(a) for a in gt_class)
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.sigmoid(logits.flatten(
            0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-paddle.log(1 - out_prob + 1e-8))
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-paddle.log(out_prob + 1e-8))
            cost_class = pos_cost_class.index_select(paddle.to_tensor([0]), axis=1) \
                         - neg_cost_class.index_select(paddle.to_tensor([0]), axis=1)
        else:
            cost_class = -paddle.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        num_all_target_boxes = tgt_bbox.shape[0]
        expanded_out_bbox = paddle.expand(paddle.unsqueeze(out_bbox, [1]),
                                          [bs * num_queries, num_all_target_boxes, 4])  # [batch_size * num_queries, num_all_target_boxes, 4]
        expanded_tgt_bbox = paddle.expand(paddle.unsqueeze(tgt_bbox, [0]),
                                          [bs * num_queries, num_all_target_boxes, 4])  # [batch_size * num_queries, num_all_target_boxes, 4]
        cost_bbox = F.loss.l1_loss(expanded_out_bbox, expanded_tgt_bbox,
                                   reduction='none')  # [batch_size * num_queries, num_all_target_boxes,4]
        cost_bbox = paddle.sum(cost_bbox, -1)


        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(expanded_out_bbox),
            bbox_cxcywh_to_xyxy(expanded_tgt_bbox)).squeeze(-1)

        # Final cost matrix
        C = (self.matcher_coeff['class'] * cost_class + self.matcher_coeff['bbox'] * cost_bbox
            + self.matcher_coeff['giou'] * cost_giou)
        C = paddle.reshape(C, [bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]

        sizes = [a.shape[0] for a in gt_bbox]
        indices = [
            linear_sum_assignment(c.split(sizes, -1)[i].numpy())
            for i, c in enumerate(C)
        ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
                j, dtype=paddle.int64)) for i, j in indices]

@register
@serializable
class OVHungarianMatcher(OVHungarianMatcher_ori):
    __shared__ = ['use_focal_loss']

    def __init__(self,
                 matcher_coeff={'class': 1,
                                'bbox': 5,
                                'giou': 2},
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        self.use_focal_loss = use_focal_loss
        super(OVHungarianMatcher, self).__init__(
            matcher_coeff=matcher_coeff,
            use_focal_loss=use_focal_loss,
            alpha=alpha,
            gamma=gamma)

    @paddle.no_grad()
    def forward(self, boxes, logits, gt_bbox, gt_class, select_id):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            select_id (List(Tensor)): List[max_len]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        num_patch = len(select_id)
        bs, num_queries = logits.shape[:2]

        num_gts = sum(len(a) for a in gt_class)
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                [], dtype=paddle.int64)) for _ in range(bs)]

        num_queries = num_queries // num_patch

        out_prob_all = paddle.reshape(logits, [bs, num_patch, num_queries, -1])
        out_bbox_all = paddle.reshape(boxes, [bs, num_patch, num_queries, -1]).astype('float32')

        # Also concat the target labels and boxes
        tgt_ids_all = paddle.concat(gt_class).flatten()
        tgt_bbox_all = paddle.concat(gt_bbox)

        ans = [[[], []] for _ in range(bs)]

        for index, label in enumerate(select_id):
            out_prob = F.sigmoid(out_prob_all[:, index, :, :].flatten(0, 1))
            out_bbox = out_bbox_all[:, index, :, :].flatten(0, 1)

            mask = (tgt_ids_all == label).nonzero().squeeze(1)

            if len(mask) > 0:
                tgt_bbox = tgt_bbox_all.index_select(mask, axis=0)

                # Compute the classification cost.
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-paddle.log(1 - out_prob + 1e-8))
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-paddle.log(out_prob + 1e-8))
                cost_class = pos_cost_class.index_select(paddle.to_tensor([0]), axis=1) \
                             - neg_cost_class.index_select(paddle.to_tensor([0]), axis=1)

                # Compute the L1 cost between boxes
                num_all_target_boxes = tgt_bbox.shape[0]
                expanded_out_bbox = paddle.expand(paddle.unsqueeze(out_bbox, [1]),
                                                  [bs * num_queries, num_all_target_boxes,
                                                   4])  # [batch_size * num_queries, num_all_target_boxes, 4]
                expanded_tgt_bbox = paddle.expand(paddle.unsqueeze(tgt_bbox, [0]),
                                                  [bs * num_queries, num_all_target_boxes,
                                                   4])  # [batch_size * num_queries, num_all_target_boxes, 4]
                cost_bbox = F.loss.l1_loss(expanded_out_bbox, expanded_tgt_bbox,
                                           reduction='none')  # [batch_size * num_queries, num_all_target_boxes,4]
                cost_bbox = paddle.sum(cost_bbox, -1)

                # Compute the giou cost betwen boxes
                cost_giou = self.giou_loss(
                    bbox_cxcywh_to_xyxy(expanded_out_bbox),
                    bbox_cxcywh_to_xyxy(expanded_tgt_bbox)).squeeze(-1)

                # Final cost matrix
                C = (self.matcher_coeff['class'] * cost_class
                     + self.matcher_coeff['bbox'] * cost_bbox
                     + self.matcher_coeff['giou'] * cost_giou)

                C = paddle.reshape(C, [bs, num_queries, -1])

                sizes = []
                for a in gt_class:
                    if len(a) > 0:
                        mask = a == label
                        sizes.append(len(paddle.masked_select(a, mask)))

                indices = [
                    linear_sum_assignment(c.split(sizes, -1)[i].numpy())
                    for i, c in enumerate(C)
                ]

                for ind in range(bs):
                    x, y = indices[ind]
                    if len(x) == 0:
                        continue
                    x += index * num_queries
                    ans[ind][0] += x.tolist()
                    y_label = (gt_class[ind].squeeze(1) == label).nonzero().squeeze(1).cpu().numpy()
                    y_label = y_label[y].tolist()
                    ans[ind][1] += y_label

        return [(paddle.to_tensor(i, dtype="int64"), paddle.to_tensor(j, dtype="int64"))
                for i, j in ans]

