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

from ..data.source.voc import pascalvoc_label
from .map_utils import DetectionMAP
from .coco_eval import bbox2out

import logging
logger = logging.getLogger(__name__)

__all__ = ['bbox_eval', 'bbox2out', 'get_category_info']


def bbox_eval(results,
              class_num,
              overlap_thresh=0.5,
              map_type='11point',
              is_bbox_normalized=False,
              evaluate_difficult=False):
    """
    Bounding box evaluation for VOC dataset

    Args:
        results (list): prediction bounding box results.
        class_num (int): evaluation class number.
        overlap_thresh (float): the postive threshold of
                        bbox overlap
        map_type (string): method for mAP calcualtion,
                        can only be '11point' or 'integral'
        is_bbox_normalized (bool): whether bbox is normalized
                        to range [0, 1].
        evaluate_difficult (bool): whether to evaluate
                        difficult gt bbox.
    """
    assert 'bbox' in results[0]
    logger.info("Start evaluate...")

    detection_map = DetectionMAP(
        class_num=class_num,
        overlap_thresh=overlap_thresh,
        map_type=map_type,
        is_bbox_normalized=is_bbox_normalized,
        evaluate_difficult=evaluate_difficult)

    for t in results:
        bboxes = t['bbox'][0]
        bbox_lengths = t['bbox'][1][0]

        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        gt_boxes = t['gt_bbox'][0]
        gt_labels = t['gt_class'][0]
        difficults = t['is_difficult'][0] if not evaluate_difficult \
                            else None

        if len(t['gt_bbox'][1]) == 0:
            # gt_bbox, gt_class, difficult read as zero padded Tensor
            bbox_idx = 0
            for i in range(len(gt_boxes)):
                gt_box = gt_boxes[i]
                gt_label = gt_labels[i]
                difficult = None if difficults is None \
                                else difficults[i]
                bbox_num = bbox_lengths[i]
                bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
                gt_box, gt_label, difficult = prune_zero_padding(
                    gt_box, gt_label, difficult)
                detection_map.update(bbox, gt_box, gt_label, difficult)
                bbox_idx += bbox_num
        else:
            # gt_box, gt_label, difficult read as LoDTensor
            gt_box_lengths = t['gt_bbox'][1][0]
            bbox_idx = 0
            gt_box_idx = 0
            for i in range(len(bbox_lengths)):
                bbox_num = bbox_lengths[i]
                gt_box_num = gt_box_lengths[i]
                bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
                gt_box = gt_boxes[gt_box_idx:gt_box_idx + gt_box_num]
                gt_label = gt_labels[gt_box_idx:gt_box_idx + gt_box_num]
                difficult = None if difficults is None else \
                            difficults[gt_box_idx: gt_box_idx + gt_box_num]
                detection_map.update(bbox, gt_box, gt_label, difficult)
                bbox_idx += bbox_num
                gt_box_idx += gt_box_num

    logger.info("Accumulating evaluatation results...")
    detection_map.accumulate()
    map_stat = 100. * detection_map.get_map()
    logger.info("mAP({:.2f}, {}) = {:.2f}%".format(overlap_thresh, map_type,
                                                   map_stat))
    return map_stat


def prune_zero_padding(gt_box, gt_label, difficult=None):
    valid_cnt = 0
    for i in range(len(gt_box)):
        if gt_box[i, 0] == 0 and gt_box[i, 1] == 0 and \
                gt_box[i, 2] == 0 and gt_box[i, 3] == 0:
            break
        valid_cnt += 1
    return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
            if difficult is not None else None)


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "voc2012 categories.".format(anno_file))
        return vocall_category_info(with_background)
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
    cats = []
    with open(anno_file) as f:
        for line in f.readlines():
            cats.append(line.strip())

    if cats[0] != 'background' and with_background:
        cats.insert(0, 'background')
    if cats[0] == 'background' and not with_background:
        cats = cats[1:]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def vocall_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup voc dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = pascalvoc_label(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
