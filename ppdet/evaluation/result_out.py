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
from __future__ import unicode_literals

import os
import sys
import json
import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'bbox2out',
    'mask2out',
    'proposal2out',
    'segm2out',
]


def clip_bbox(bbox, im_size=None):
    h = 1. if im_size is None else im_size[0]
    w = 1. if im_size is None else im_size[1]
    xmin = max(min(bbox[0], w), 0.)
    ymin = max(min(bbox[1], h), 0.)
    xmax = max(min(bbox[2], w), 0.)
    ymax = max(min(bbox[3], h), 0.)
    return xmin, ymin, xmax, ymax


def proposal2out(results, is_bbox_normalized=False):
    xywh_res = []
    for t in results:
        bboxes = t['proposal'][0]
        lengths = t['proposal'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        assert len(lengths) == im_ids.size
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i])
            for j in range(num):
                dt = bboxes[k]
                xmin, ymin, xmax, ymax = dt.tolist()

                if is_bbox_normalized:
                    xmin, ymin, xmax, ymax = \
                            clip_bbox([xmin, ymin, xmax, ymax])
                    w = xmax - xmin
                    h = ymax - ymin
                else:
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': 1,
                    'bbox': bbox,
                    'score': 1.0
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def bbox2out(results, clsid2catid, is_bbox_normalized=False):
    """
    Args:
        results: request a dict, should include: `bbox`, `im_id`,
                 if is_bbox_normalized=True, also need `im_shape`.
        clsid2catid: class id to category id map of COCO2017 dataset.
        is_bbox_normalized: whether or not bbox is normalized.
    """
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        if len(t['bbox'][1]) == 0: continue
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None or len(bboxes) == 0:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i])
            for j in range(num):
                dt = bboxes[k]
                clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                if clsid < 0: continue
                catid = (clsid2catid[int(clsid)])

                if is_bbox_normalized:
                    xmin, ymin, xmax, ymax = \
                            clip_bbox([xmin, ymin, xmax, ymax])
                    w = xmax - xmin
                    h = ymax - ymin
                    im_shape = t['im_shape'][0][i].tolist()
                    im_height, im_width = int(im_shape[0]), int(im_shape[1])
                    xmin *= im_width
                    ymin *= im_height
                    w *= im_width
                    h *= im_height
                else:
                    # for yolov4
                    # w = xmax - xmin
                    # h = ymax - ymin
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def mask2out(results, clsid2catid, resolution, thresh_binarize=0.5):
    import pycocotools.mask as mask_util
    scale = (resolution + 2.0) / resolution

    segm_res = []

    # for each batch
    for t in results:
        bboxes = t['bbox'][0]

        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0])
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        if len(bboxes.tolist()) == 0:
            continue

        masks = t['mask'][0]

        s = 0
        # for each sample
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])
            im_shape = t['im_shape'][0][i]

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

                catid = clsid2catid[clsid]

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
                catid = clsid2catid[clsid]
                segm['counts'] = segm['counts'].decode('utf8')
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'segmentation': segm,
                    'score': score
                }
                segm_res.append(coco_res)
    return segm_res


def segm2out(results, clsid2catid, thresh_binarize=0.5):
    import pycocotools.mask as mask_util
    segm_res = []

    # for each batch
    for t in results:
        segms = t['segm'][0].astype(np.uint8)
        clsid_labels = t['cate_label'][0]
        clsid_scores = t['cate_score'][0]
        lengths = segms.shape[0]
        im_id = int(t['im_id'][0][0])
        im_shape = t['im_shape'][0][0]
        if lengths == 0 or segms is None:
            continue
        # for each sample
        for i in range(lengths - 1):
            im_h = int(im_shape[0])
            im_w = int(im_shape[1])

            clsid = int(clsid_labels[i]) + 1
            catid = clsid2catid[clsid]
            score = clsid_scores[i]
            mask = segms[i]
            segm = mask_util.encode(
                np.array(
                    mask[:, :, np.newaxis], order='F'))[0]
            segm['counts'] = segm['counts'].decode('utf8')
            coco_res = {
                'image_id': im_id,
                'category_id': catid,
                'segmentation': segm,
                'score': score
            }
            segm_res.append(coco_res)
    return segm_res


def expand_boxes(boxes, scale):
    """
    Expand an array of boxes by a given scale.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def mask_encode(results, resolution, thresh_binarize=0.5):
    import pycocotools.mask as mask_util
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