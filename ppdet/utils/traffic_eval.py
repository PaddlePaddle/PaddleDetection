# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import json

from ..data.source.traffic import trans_label as label_list
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
        difficults = t['is_difficult'][0] if evaluate_difficult \
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
    logger.info("mAP({:.2f}, {}) = {:.2f}".format(overlap_thresh, map_type,
                                                  map_stat))
    return map_stat


def write_output(results, im_info_file, catid2name, gt_path, outpath='test'):
    im_info_dict = {}
    with open(im_info_file, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            im_infos = line.strip().split()
            im_info_dict[int(im_infos[
                2])] = [str(im_infos[0]), str(im_infos[1])]

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    out_results = []
    for res in results:
        assert res['image_id'] in im_info_dict.keys()
        out_dict = {
            "pic_id": im_info_dict[res['image_id']][1],
            "x": res['bbox'][0],
            "y": res['bbox'][1],
            "w": res['bbox'][2],
            "h": res['bbox'][3],
            "score": res['score'],
            "type": catid2name[res['category_id']]
        }

        group_id = im_info_dict[res['image_id']][0]
        out_res_file = os.path.join(outpath, group_id)
        if not os.path.exists(out_res_file):
            match_data = json.load(open(os.path.join(gt_path, group_id)))
            match_data['signs'] = [out_dict]
        else:
            match_data['signs'].append(out_dict)
        with open(out_res_file, 'w') as fp:
            json.dump(match_data, fp)
        fp.close()


def prune_zero_padding(gt_box, gt_label, difficult=None):
    valid_cnt = 0
    for i in range(len(gt_box)):
        if gt_box[i, 0] == 0 and gt_box[i, 1] == 0 and \
                gt_box[i, 2] == 0 and gt_box[i, 3] == 0:
            break
        valid_cnt += 1
    return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
            if difficult is not None else None)


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
        #print(t)
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


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "voc2012 categories.".format(anno_file))
        return all_category_info(with_background)
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


def all_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup voc dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = label_list(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
