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
import paddle.nn.functional as F


def pad_gt(gt_labels, gt_bboxes, gt_scores=None):
    r""" Pad 0 in gt_labels and gt_bboxes.
    Args:
        gt_labels (Tensor|List[Tensor], int64): Label of gt_bboxes,
            shape is [B, n, 1] or [[n_1, 1], [n_2, 1], ...], here n = sum(n_i)
        gt_bboxes (Tensor|List[Tensor], float32): Ground truth bboxes,
            shape is [B, n, 4] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
        gt_scores (Tensor|List[Tensor]|None, float32): Score of gt_bboxes,
            shape is [B, n, 1] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
    Returns:
        pad_gt_labels (Tensor, int64): shape[B, n, 1]
        pad_gt_bboxes (Tensor, float32): shape[B, n, 4]
        pad_gt_scores (Tensor, float32): shape[B, n, 1]
        pad_gt_mask (Tensor, float32): shape[B, n, 1], 1 means bbox, 0 means no bbox
    """
    if isinstance(gt_labels, paddle.Tensor) and isinstance(gt_bboxes,
                                                           paddle.Tensor):
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3
        pad_gt_mask = (
            gt_bboxes.sum(axis=-1, keepdim=True) > 0).astype(gt_bboxes.dtype)
        if gt_scores is None:
            gt_scores = pad_gt_mask.clone()
        assert gt_labels.ndim == gt_scores.ndim

        return gt_labels, gt_bboxes, gt_scores, pad_gt_mask
    elif isinstance(gt_labels, list) and isinstance(gt_bboxes, list):
        assert len(gt_labels) == len(gt_bboxes), \
            'The number of `gt_labels` and `gt_bboxes` is not equal. '
        num_max_boxes = max([len(a) for a in gt_bboxes])
        batch_size = len(gt_bboxes)
        # pad label and bbox
        pad_gt_labels = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_labels[0].dtype)
        pad_gt_bboxes = paddle.zeros(
            [batch_size, num_max_boxes, 4], dtype=gt_bboxes[0].dtype)
        pad_gt_scores = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_bboxes[0].dtype)
        pad_gt_mask = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_bboxes[0].dtype)
        for i, (label, bbox) in enumerate(zip(gt_labels, gt_bboxes)):
            if len(label) > 0 and len(bbox) > 0:
                pad_gt_labels[i, :len(label)] = label
                pad_gt_bboxes[i, :len(bbox)] = bbox
                pad_gt_mask[i, :len(bbox)] = 1.
                if gt_scores is not None:
                    pad_gt_scores[i, :len(gt_scores[i])] = gt_scores[i]
        if gt_scores is None:
            pad_gt_scores = pad_gt_mask.clone()
        return pad_gt_labels, pad_gt_bboxes, pad_gt_scores, pad_gt_mask
    else:
        raise ValueError('The input `gt_labels` or `gt_bboxes` is invalid! ')


def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, topk], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = paddle.topk(
        metrics, topk, axis=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).tile(
            [1, 1, topk])
    topk_idxs = paddle.where(topk_mask, topk_idxs, paddle.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
    is_in_topk = paddle.where(is_in_topk > 1,
                              paddle.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.astype(metrics.dtype)


def check_points_inside_bboxes(points, bboxes, eps=1e-9):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = points.unsqueeze([0, 1])
    x, y = points.chunk(2, axis=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, axis=-1)
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    bbox_ltrb = paddle.concat([l, t, r, b], axis=-1)
    return (bbox_ltrb.min(axis=-1) > eps).astype(bboxes.dtype)


def compute_max_iou_anchor(ious):
    r"""
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(axis=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).transpose([0, 2, 1])
    return is_max_iou.astype(ious.dtype)


def compute_max_iou_gt(ious):
    r"""
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(axis=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou.astype(ious.dtype)
