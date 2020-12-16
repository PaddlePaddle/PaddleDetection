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
    'bbox_eval',
    'mask_eval',
    'bbox2out',
    'mask2out',
    'get_category_info',
    'proposal_eval',
    'cocoapi_eval',
]


def clip_bbox(bbox, im_size=None):
    h = 1. if im_size is None else im_size[0]
    w = 1. if im_size is None else im_size[1]
    xmin = max(min(bbox[0], w), 0.)
    ymin = max(min(bbox[1], h), 0.)
    xmax = max(min(bbox[2], w), 0.)
    ymax = max(min(bbox[3], h), 0.)
    return xmin, ymin, xmax, ymax


def proposal_eval(results, anno_file, outfile, max_dets=(100, 300, 1000)):
    assert 'proposal' in results[0]
    assert outfile.endswith('.json')

    xywh_results = proposal2out(results)
    assert len(
        xywh_results) > 0, "The number of valid proposal detected is zero.\n \
        Please use reasonable model and check input data."

    with open(outfile, 'w') as f:
        json.dump(xywh_results, f)

    cocoapi_eval(outfile, 'proposal', anno_file=anno_file, max_dets=max_dets)
    # flush coco evaluation result
    sys.stdout.flush()


def bbox_eval(results,
              anno_file,
              outfile,
              with_background=True,
              is_bbox_normalized=False,
              save_only=False):
    assert 'bbox' in results[0]
    assert outfile.endswith('.json')
    from pycocotools.coco import COCO

    coco_gt = COCO(anno_file)
    cat_ids = coco_gt.getCatIds()

    # when with_background = True, mapping category to classid, like:
    #   background:0, first_class:1, second_class:2, ...
    clsid2catid = dict(
        {i + int(with_background): catid
         for i, catid in enumerate(cat_ids)})

    xywh_results = bbox2out(
        results, clsid2catid, is_bbox_normalized=is_bbox_normalized)

    if len(xywh_results) == 0:
        logger.warning("The number of valid bbox detected is zero.\n \
            Please use reasonable model and check input data.\n \
            stop eval!")
        return [0.0]
    with open(outfile, 'w') as f:
        json.dump(xywh_results, f)

    if save_only:
        logger.info('The bbox result is saved to {} and do not '
                    'evaluate the mAP.'.format(outfile))
        return

    map_stats = cocoapi_eval(outfile, 'bbox', coco_gt=coco_gt)
    # flush coco evaluation result
    sys.stdout.flush()
    return map_stats


def mask_eval(results,
              anno_file,
              outfile,
              resolution,
              thresh_binarize=0.5,
              save_only=False):
    """
    Format the output of mask and get mask ap by coco api evaluation.
    It will be used in Mask-RCNN.
    """
    assert 'mask' in results[0]
    assert outfile.endswith('.json')
    from pycocotools.coco import COCO

    coco_gt = COCO(anno_file)
    clsid2catid = {i + 1: v for i, v in enumerate(coco_gt.getCatIds())}

    segm_results = []
    for t in results:
        im_ids = np.array(t['im_id'][0])
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        masks = t['mask']
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        if len(bboxes.tolist()) == 0:
            continue
        s = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])
            clsid_scores = bboxes[s:s + num][:, 0:2]
            mask = masks[s:s + num]
            s += num
            for j in range(num):
                clsid, score = clsid_scores[j].tolist()
                catid = int(clsid2catid[clsid])
                segm = mask[j]
                segm['counts'] = segm['counts'].decode('utf8')
                coco_res = {
                    'image_id': im_id,
                    'category_id': int(catid),
                    'segmentation': segm,
                    'score': score
                }
                segm_results.append(coco_res)

    if len(segm_results) == 0:
        logger.warning("The number of valid mask detected is zero.\n \
            Please use reasonable model and check input data.")
        return

    with open(outfile, 'w') as f:
        json.dump(segm_results, f)

    if save_only:
        logger.info('The mask result is saved to {} and do not '
                    'evaluate the mAP.'.format(outfile))
        return

    cocoapi_eval(outfile, 'segm', coco_gt=coco_gt)


def segm_eval(results, anno_file, outfile, save_only=False):
    """
    Format the output of segmentation, category_id and score in mask.josn, and
    get mask ap by coco api evaluation. It will be used in instance segmentation
    networks, such as: SOLOv2.
    """
    assert 'segm' in results[0]
    assert outfile.endswith('.json')
    from pycocotools.coco import COCO
    coco_gt = COCO(anno_file)
    clsid2catid = {i: v for i, v in enumerate(coco_gt.getCatIds())}
    segm_results = []
    for t in results:
        im_id = int(t['im_id'][0][0])
        segs = t['segm']
        for mask in segs:
            catid = int(clsid2catid[mask[0]])
            masks = mask[1]
            mask_score = masks[1]
            segm = masks[0]
            segm['counts'] = segm['counts'].decode('utf8')
            coco_res = {
                'image_id': im_id,
                'category_id': catid,
                'segmentation': segm,
                'score': mask_score
            }
            segm_results.append(coco_res)

    if len(segm_results) == 0:
        logger.warning("The number of valid mask detected is zero.\n \
            Please use reasonable model and check input data.")
        return

    with open(outfile, 'w') as f:
        json.dump(segm_results, f)

    if save_only:
        logger.info('The mask result is saved to {} and do not '
                    'evaluate the mAP.'.format(outfile))
        return

    map_stats = cocoapi_eval(outfile, 'segm', coco_gt=coco_gt)
    return map_stats


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        jsonfile: Evaluation json file, eg: bbox.json, mask.json.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt != None or anno_file != None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


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
                if clsid < 0:
                    k += 1  #to continue to handle the subsequent bbox of current image
                    continue
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


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "coco17 categories.".format(anno_file))
        return coco17_category_info(with_background)
    else:
        logger.info("Load categories from {}".format(anno_file))
        return get_category_info_from_anno(anno_file, with_background)


def get_category_info_from_anno(anno_file, with_background=True):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        anno_file (str): annotation file path
        with_background (bool, default True):
            whether load background as class 0.
    """
    from pycocotools.coco import COCO
    coco = COCO(anno_file)
    cats = coco.loadCats(coco.getCatIds())
    clsid2catid = {
        i + int(with_background): cat['id']
        for i, cat in enumerate(cats)
    }
    catid2name = {cat['id']: cat['name'] for cat in cats}
    if with_background:
        clsid2catid.update({0: 0})
        catid2name.update({0: 'background'})
    return clsid2catid, catid2name


def coco17_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of COCO2017 dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    catid2name = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }

    if not with_background:
        clsid2catid = {k - 1: v for k, v in clsid2catid.items()}
        catid2name.pop(0)
    else:
        clsid2catid.update({0: 0})

    return clsid2catid, catid2name
