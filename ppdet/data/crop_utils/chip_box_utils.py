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

import numpy as np


def bbox_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection_over_box(chips, boxes):
    """
    intersection area over box area
    :param chips:  C
    :param boxes:  B
    :return: iob, CxB
    """
    M = chips.shape[0]
    N = boxes.shape[0]
    if M * N == 0:
        return np.zeros([M, N], dtype='float32')

    box_area = bbox_area(boxes)  # B

    inter_x2y2 = np.minimum(np.expand_dims(chips, 1)[:, :, 2:], boxes[:, 2:])  # CxBX2
    inter_x1y1 = np.maximum(np.expand_dims(chips, 1)[:, :, :2], boxes[:, :2])  # CxBx2
    inter_wh = inter_x2y2 - inter_x1y1
    inter_wh = np.clip(inter_wh, a_min=0, a_max=None)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # CxB

    iob = inter_area / np.expand_dims(box_area, 0)
    return iob


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4]
    :param im_shape: tuple of 2, [h, w]
    :return: [N, 4]
    """
    # x1 >= 0
    boxes[:, 0] = np.clip(boxes[:, 0], 0, im_shape[1] - 1)
    # y1 >= 0
    boxes[:, 1] = np.clip(boxes[:, 1], 0, im_shape[0] - 1)
    # x2 < im_shape[1]
    boxes[:, 2] = np.clip(boxes[:, 2], 1, im_shape[1])
    # y2 < im_shape[0]
    boxes[:, 3] = np.clip(boxes[:, 3], 1, im_shape[0])
    return boxes


def transform_chip_box(gt_bbox: 'Gx4', boxes_idx: 'B', chip: '4'):
    boxes_idx = np.array(boxes_idx)
    cur_gt_bbox = gt_bbox[boxes_idx].copy()  # Bx4
    x1, y1, x2, y2 = chip
    cur_gt_bbox[:, 0] -= x1
    cur_gt_bbox[:, 1] -= y1
    cur_gt_bbox[:, 2] -= x1
    cur_gt_bbox[:, 3] -= y1
    h = y2 - y1
    w = x2 - x1
    cur_gt_bbox = clip_boxes(cur_gt_bbox, (h, w))
    ws = (cur_gt_bbox[:, 2] - cur_gt_bbox[:, 0]).astype(np.int32)
    hs = (cur_gt_bbox[:, 3] - cur_gt_bbox[:, 1]).astype(np.int32)
    valid_idx = (ws >= 2) & (hs >= 2)
    return cur_gt_bbox[valid_idx], boxes_idx[valid_idx]


def find_chips_to_cover_overlaped_boxes(iob, overlap_threshold):
    chip_ids, box_ids = np.nonzero(iob >= overlap_threshold)
    chip_id2overlap_box_num = np.bincount(chip_ids)  # 1d array
    chip_id2overlap_box_num = np.pad(chip_id2overlap_box_num, (0, len(iob) - len(chip_id2overlap_box_num)),
                                     constant_values=0)

    chosen_chip_ids = []
    while len(box_ids) > 0:
        value_counts = np.bincount(chip_ids)  # 1d array
        max_count_chip_id = np.argmax(value_counts)
        assert max_count_chip_id not in chosen_chip_ids
        chosen_chip_ids.append(max_count_chip_id)

        box_ids_in_cur_chip = box_ids[chip_ids == max_count_chip_id]
        ids_not_in_cur_boxes_mask = np.logical_not(np.isin(box_ids, box_ids_in_cur_chip))
        chip_ids = chip_ids[ids_not_in_cur_boxes_mask]
        box_ids = box_ids[ids_not_in_cur_boxes_mask]
    return chosen_chip_ids, chip_id2overlap_box_num


def transform_chip_boxes2image_boxes(chip_boxes, chip, img_h, img_w):
    chip_boxes = np.array(sorted(chip_boxes, key=lambda item: -item[1]))
    xmin, ymin, _, _ = chip
    # Transform to origin image loc
    chip_boxes[:, 2] += xmin
    chip_boxes[:, 4] += xmin
    chip_boxes[:, 3] += ymin
    chip_boxes[:, 5] += ymin
    chip_boxes = clip_boxes(chip_boxes, (img_h, img_w))
    return chip_boxes


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 1]
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    # _i, _j
    # sorted indices
    # i, j
    # temp variables for box i's (the box currently under consideration)
    # ix1, iy1, ix2, iy2, iarea

    # variables for computing overlap with box j (lower scoring box)
    # xx1, yy1, xx2, yy2
    # w, h
    # inter, ovr

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets
