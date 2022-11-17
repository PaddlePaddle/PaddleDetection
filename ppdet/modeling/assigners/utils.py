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

__all__ = [
    'pad_gt', 'gather_topk_anchors', 'check_points_inside_bboxes',
    'compute_max_iou_anchor', 'compute_max_iou_gt',
    'generate_anchors_for_grid_cell'
]


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
        topk_mask (Tensor, float32): shape[B, n, 1], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = paddle.topk(
        metrics, topk, axis=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (
            topk_metrics.max(axis=-1, keepdim=True) > eps).astype(metrics.dtype)
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(
        axis=-2).astype(metrics.dtype)
    return is_in_topk * topk_mask


def check_points_inside_bboxes(points,
                               bboxes,
                               center_radius_tensor=None,
                               eps=1e-9,
                               sm_use=False):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1]. Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = points.unsqueeze([0, 1])
    x, y = points.chunk(2, axis=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, axis=-1)
    # check whether `points` is in `bboxes`
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    delta_ltrb = paddle.concat([l, t, r, b], axis=-1)
    is_in_bboxes = (delta_ltrb.min(axis=-1) > eps)
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = center_radius_tensor.unsqueeze([0, 1])
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        l = x - (cx - center_radius_tensor)
        t = y - (cy - center_radius_tensor)
        r = (cx + center_radius_tensor) - x
        b = (cy + center_radius_tensor) - y
        delta_ltrb_c = paddle.concat([l, t, r, b], axis=-1)
        is_in_center = (delta_ltrb_c.min(axis=-1) > eps)
        if sm_use:
            return is_in_bboxes.astype(bboxes.dtype), is_in_center.astype(
                bboxes.dtype)
        else:
            return (paddle.logical_and(is_in_bboxes, is_in_center),
                    paddle.logical_or(is_in_bboxes, is_in_center))

    return is_in_bboxes.astype(bboxes.dtype)


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


def generate_anchors_for_grid_cell(feats,
                                   fpn_strides,
                                   grid_cell_size=5.0,
                                   grid_cell_offset=0.5,
                                   dtype='float32'):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (paddle.arange(end=w) + grid_cell_offset) * stride
        shift_y = (paddle.arange(end=h) + grid_cell_offset) * stride
        shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
        anchor = paddle.stack(
            [
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ],
            axis=-1).astype(dtype)
        anchor_point = paddle.stack([shift_x, shift_y], axis=-1).astype(dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            paddle.full(
                [num_anchors_list[-1], 1], stride, dtype=dtype))
    anchors = paddle.concat(anchors)
    anchors.stop_gradient = True
    anchor_points = paddle.concat(anchor_points)
    anchor_points.stop_gradient = True
    stride_tensor = paddle.concat(stride_tensor)
    stride_tensor.stop_gradient = True
    return anchors, anchor_points, num_anchors_list, stride_tensor
