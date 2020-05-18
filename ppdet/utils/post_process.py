# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import numpy as np
import cv2
import paddle.fluid as fluid

__all__ = ['nms']

logger = logging.getLogger(__name__)


def box_flip(boxes, im_shape):
    im_width = im_shape[0][1]
    flipped_boxes = boxes.copy()

    flipped_boxes[:, 0::4] = im_width - boxes[:, 2::4] - 1
    flipped_boxes[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return flipped_boxes


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]

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


def soft_nms(dets, sigma, thres):
    dets_final = []
    while len(dets) > 0:
        maxpos = np.argmax(dets[:, 0])
        dets_final.append(dets[maxpos].copy())
        ts, tx1, ty1, tx2, ty2 = dets[maxpos]
        scores = dets[:, 0]
        # force remove bbox at maxpos
        scores[maxpos] = -1
        x1 = dets[:, 1]
        y1 = dets[:, 2]
        x2 = dets[:, 3]
        y2 = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        xx1 = np.maximum(tx1, x1)
        yy1 = np.maximum(ty1, y1)
        xx2 = np.minimum(tx2, x2)
        yy2 = np.minimum(ty2, y2)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas + areas[maxpos] - inter)
        weight = np.exp(-(ovr * ovr) / sigma)
        scores = scores * weight
        idx_keep = np.where(scores >= thres)
        dets[:, 0] = scores
        dets = dets[idx_keep]
    dets_final = np.array(dets_final).reshape(-1, 5)
    return dets_final


def bbox_area(box):
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1
    return w * h


def bbox_overlaps(x, y):
    N = x.shape[0]
    K = y.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        y_area = bbox_area(y[k])
        for n in range(N):
            iw = min(x[n, 2], y[k, 2]) - max(x[n, 0], y[k, 0]) + 1
            if iw > 0:
                ih = min(x[n, 3], y[k, 3]) - max(x[n, 1], y[k, 1]) + 1
                if ih > 0:
                    x_area = bbox_area(x[n])
                    ua = x_area + y_area - iw * ih
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def box_voting(nms_dets, dets, vote_thresh):
    top_dets = nms_dets.copy()
    top_boxes = nms_dets[:, 1:]
    all_boxes = dets[:, 1:]
    all_scores = dets[:, 0]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(nms_dets.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= vote_thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets[k, 1:] = np.average(boxes_to_vote, axis=0, weights=ws)

    return top_dets


def get_nms_result(boxes,
                   scores,
                   config,
                   num_classes,
                   background_label=0,
                   labels=None):
    has_labels = labels is not None
    cls_boxes = [[] for _ in range(num_classes)]
    start_idx = 1 if background_label == 0 else 0
    for j in range(start_idx, num_classes):
        inds = np.where(labels == j)[0] if has_labels else np.where(
            scores[:, j] > config['score_thresh'])[0]
        scores_j = scores[inds] if has_labels else scores[inds, j]
        boxes_j = boxes[inds, :] if has_labels else boxes[inds, j * 4:(j + 1) *
                                                          4]
        dets_j = np.hstack((scores_j[:, np.newaxis], boxes_j)).astype(
            np.float32, copy=False)
        if config.get('use_soft_nms', False):
            nms_dets = soft_nms(dets_j, config['sigma'], config['nms_thresh'])
        else:
            nms_dets = nms(dets_j, config['nms_thresh'])
        if config.get('enable_voting', False):
            nms_dets = box_voting(nms_dets, dets_j, config['vote_thresh'])
        #add labels
        label = np.array([j for _ in range(len(nms_dets))])
        nms_dets = np.hstack((label[:, np.newaxis], nms_dets)).astype(
            np.float32, copy=False)
        cls_boxes[j] = nms_dets
    # Limit to max_per_image detections **over all classes**
    image_scores = np.hstack(
        [cls_boxes[j][:, 1] for j in range(start_idx, num_classes)])
    if len(image_scores) > config['detections_per_im']:
        image_thresh = np.sort(image_scores)[-config['detections_per_im']]
        for j in range(start_idx, num_classes):
            keep = np.where(cls_boxes[j][:, 1] >= image_thresh)[0]
            cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack(
        [cls_boxes[j] for j in range(start_idx, num_classes)])
    return im_results


def mstest_box_post_process(result, config, num_classes):
    """
    Multi-scale Test
    Only available for batch_size=1 now.
    """
    post_bbox = {}
    use_flip = False
    ms_boxes = []
    ms_scores = []
    im_shape = result['im_shape'][0]
    for k in result.keys():
        if 'bbox' in k:
            boxes = result[k][0]
            boxes = np.reshape(boxes, (-1, 4 * num_classes))
            scores = result['score' + k[4:]][0]
            if 'flip' in k:
                boxes = box_flip(boxes, im_shape)
                use_flip = True
            ms_boxes.append(boxes)
            ms_scores.append(scores)

    ms_boxes = np.concatenate(ms_boxes)
    ms_scores = np.concatenate(ms_scores)
    bbox_pred = get_nms_result(ms_boxes, ms_scores, config, num_classes)
    post_bbox.update({'bbox': (bbox_pred, [[len(bbox_pred)]])})
    if use_flip:
        bbox = bbox_pred[:, 2:]
        bbox_flip = np.append(
            bbox_pred[:, :2], box_flip(bbox, im_shape), axis=1)
        post_bbox.update({'bbox_flip': (bbox_flip, [[len(bbox_flip)]])})
    return post_bbox


def mstest_mask_post_process(result, cfg):
    mask_list = []
    im_shape = result['im_shape'][0]
    M = cfg.FPNRoIAlign['mask_resolution']
    for k in result.keys():
        if 'mask' in k:
            masks = result[k][0]
            if len(masks.shape) != 4:
                masks = np.zeros((0, M, M))
                mask_list.append(masks)
                continue
            if 'flip' in k:
                masks = masks[:, :, :, ::-1]
            mask_list.append(masks)

    mask_pred = np.mean(mask_list, axis=0)
    return {'mask': (mask_pred, [[len(mask_pred)]])}


def mask_encode(results, resolution, thresh_binarize=0.5):
    import pycocotools.mask as mask_util
    from ppdet.utils.coco_eval import expand_boxes
    scale = (resolution + 2.0) / resolution
    bboxes = results['bbox'][0]
    masks = results['mask'][0]
    lengths = results['mask'][1][0]
    im_shapes = results['im_shape'][0]
    segms = []
    if bboxes.shape == (1, 1) or bboxes is None:
        return segms
    if len(bboxes.tolist()) == 0:
        return segms

    s = 0
    # for each sample
    for i in range(len(lengths)):
        num = lengths[i]
        im_shape = im_shapes[i]

        bbox = bboxes[s:s + num][:, 2:]
        clsid_scores = bboxes[s:s + num][:, 0:2]
        mask = masks[s:s + num]
        s += num

        im_h = int(im_shape[0])
        im_w = int(im_shape[1])
        expand_bbox = expand_boxes(bbox, scale)
        expand_bbox = expand_bbox.astype(np.int32)
        padded_mask = np.zeros(
            (resolution + 2, resolution + 2), dtype=np.float32)

        for j in range(num):
            xmin, ymin, xmax, ymax = expand_bbox[j].tolist()
            clsid, score = clsid_scores[j].tolist()
            clsid = int(clsid)
            padded_mask[1:-1, 1:-1] = mask[j, clsid, :, :]

            w = xmax - xmin + 1
            h = ymax - ymin + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)
            resized_mask = cv2.resize(padded_mask, (w, h))
            resized_mask = np.array(
                resized_mask > thresh_binarize, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x0 = min(max(xmin, 0), im_w)
            x1 = min(max(xmax + 1, 0), im_w)
            y0 = min(max(ymin, 0), im_h)
            y1 = min(max(ymax + 1, 0), im_h)

            im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
                x0 - xmin):(x1 - xmin)]
            segm = mask_util.encode(
                np.array(
                    im_mask[:, :, np.newaxis], order='F'))[0]
            segms.append(segm)
    return segms


def corner_post_process(results, config, num_classes):
    detections = results['bbox'][0]
    keep_inds = (detections[:, 1] > -1)
    detections = detections[keep_inds]
    labels = detections[:, 0]
    scores = detections[:, 1]
    boxes = detections[:, 2:6]
    cls_boxes = get_nms_result(
        boxes, scores, config, num_classes, background_label=-1, labels=labels)
    results.update({'bbox': (cls_boxes, [[len(cls_boxes)]])})
