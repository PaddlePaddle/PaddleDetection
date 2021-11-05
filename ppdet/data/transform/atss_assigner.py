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

# The code is based on:
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ['iou', 'iof', 'giou'], 'Unsupported mode {}'.format(mode)
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2] if bboxes1.shape[0] > 0 else 0
    cols = bboxes2.shape[-2] if bboxes2.shape[0] > 0 else 0
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return np.random.random(batch_shape + (rows, ))
        else:
            return np.random.random(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = np.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = np.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = np.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = np.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = np.maximum(bboxes1[..., :, None, :2],
                        bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = np.minimum(bboxes1[..., :, None, 2:],
                        bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = np.minimum(bboxes1[..., :, None, :2],
                                     bboxes2[..., None, :, :2])
            enclosed_rb = np.maximum(bboxes1[..., :, None, 2:],
                                     bboxes2[..., None, :, 2:])

    eps = np.array([eps])
    union = np.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = np.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def topk_(input, k, axis=1, largest=True):
    x = -input if largest else input
    if axis == 0:
        row_index = np.arange(input.shape[1 - axis])
        topk_index = np.argpartition(x, k, axis=axis)[0:k, :]
        topk_data = x[topk_index, row_index]

        topk_index_sort = np.argsort(topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:k, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(x.shape[1 - axis])[:, None]
        topk_index = np.argpartition(x, k, axis=axis)[:, 0:k]
        topk_data = x[column_index, topk_index]
        topk_data = -topk_data if largest else topk_data
        topk_index_sort = np.argsort(topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:k][column_index, topk_index_sort]

    return topk_data_sort, topk_index_sort


class ATSSAssigner(object):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, topk=9):
        self.topk = topk

    def __call__(self,
                 bboxes,
                 num_level_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 gt_labels=None):
        """Assign gt to bboxes.
        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt
        Args:
            bboxes (np.array): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (np.array): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (np.array, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (np.array, optional): Label of gt_bboxes, shape (k, ).
        """
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        # assign 0 by default
        assigned_gt_inds = np.zeros((num_bboxes, ), dtype=np.int64)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = np.zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if not np.any(gt_labels):
                assigned_labels = None
            else:
                assigned_labels = -np.ones((num_bboxes, ), dtype=np.int64)
            return assigned_gt_inds, max_overlaps

        # compute iou between all bbox and gt
        overlaps = bbox_overlaps(bboxes, gt_bboxes)
        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = np.stack((gt_cx, gt_cy), axis=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = np.stack((bboxes_cx, bboxes_cy), axis=1)

        distances = np.sqrt(
            np.power((bboxes_points[:, None, :] - gt_points[None, :, :]), 2)
            .sum(-1))

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for bboxes_per_level in num_level_bboxes:
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = topk_(
                distances_per_level, selectable_k, axis=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = np.concatenate(candidate_idxs, axis=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, np.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = np.broadcast_to(
            bboxes_cx.reshape(1, -1), [num_gt, num_bboxes]).reshape(-1)
        ep_bboxes_cy = np.broadcast_to(
            bboxes_cy.reshape(1, -1), [num_gt, num_bboxes]).reshape(-1)
        candidate_idxs = candidate_idxs.reshape(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].reshape(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].reshape(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].reshape(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].reshape(-1, num_gt)
        is_in_gts = np.stack([l_, t_, r_, b_], axis=1).min(axis=1) > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = -np.inf * np.ones_like(overlaps).T.reshape(-1)
        index = candidate_idxs.reshape(-1)[is_pos.reshape(-1)]
        overlaps_inf[index] = overlaps.T.reshape(-1)[index]
        overlaps_inf = overlaps_inf.reshape(num_gt, -1).T

        max_overlaps = overlaps_inf.max(axis=1)
        argmax_overlaps = overlaps_inf.argmax(axis=1)
        assigned_gt_inds[max_overlaps !=
                         -np.inf] = argmax_overlaps[max_overlaps != -np.inf] + 1

        return assigned_gt_inds, max_overlaps
