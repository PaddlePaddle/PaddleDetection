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

import sys
import numpy as np

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['bbox_area', 'jaccard_overlap', 'prune_zero_padding', 'DetectionMAP']


def bbox_area(bbox, is_bbox_normalized):
    """
    Calculate area of a bounding box
    """
    norm = 1. - float(is_bbox_normalized)
    width = bbox[2] - bbox[0] + norm
    height = bbox[3] - bbox[1] + norm
    return width * height


def jaccard_overlap(pred, gt, is_bbox_normalized=False):
    """
    Calculate jaccard overlap ratio between two bounding box
    """
    if pred[0] >= gt[2] or pred[2] <= gt[0] or \
        pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.
    inter_xmin = max(pred[0], gt[0])
    inter_ymin = max(pred[1], gt[1])
    inter_xmax = min(pred[2], gt[2])
    inter_ymax = min(pred[3], gt[3])
    inter_size = bbox_area([inter_xmin, inter_ymin, inter_xmax, inter_ymax],
                           is_bbox_normalized)
    pred_size = bbox_area(pred, is_bbox_normalized)
    gt_size = bbox_area(gt, is_bbox_normalized)
    overlap = float(inter_size) / (pred_size + gt_size - inter_size)
    return overlap


def prune_zero_padding(gt_box, gt_label, difficult=None):
    valid_cnt = 0
    for i in range(len(gt_box)):
        if gt_box[i, 0] == 0 and gt_box[i, 1] == 0 and \
                gt_box[i, 2] == 0 and gt_box[i, 3] == 0:
            break
        valid_cnt += 1
    return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
            if difficult is not None else None)


class DetectionMAP(object):
    """
    Calculate detection mean average precision.
    Currently support two types: 11point and integral

    Args:
        class_num (int): the class number.
        overlap_thresh (float): The threshold of overlap
            ratio between prediction bounding box and 
            ground truth bounding box for deciding 
            true/false positive. Default 0.5.
        map_type (str): calculation method of mean average
            precision, currently support '11point' and
            'integral'. Default '11point'.
        is_bbox_normalized (bool): whther bounding boxes
            is normalized to range[0, 1]. Default False.
        evaluate_difficult (bool): whether to evaluate
            difficult bounding boxes. Default False.
    """

    def __init__(self,
                 class_num,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False):
        self.class_num = class_num
        self.overlap_thresh = overlap_thresh
        assert map_type in ['11point', 'integral'], \
                "map_type currently only support '11point' "\
                "and 'integral'"
        self.map_type = map_type
        self.is_bbox_normalized = is_bbox_normalized
        self.evaluate_difficult = evaluate_difficult
        self.reset()

    def update(self, bbox, score, label, gt_box, gt_label, difficult=None):
        """
        Update metric statics from given prediction and ground
        truth infomations.
        """
        if difficult is None:
            difficult = np.zeros_like(gt_label)

        # record class gt count
        for gtl, diff in zip(gt_label, difficult):
            if self.evaluate_difficult or int(diff) == 0:
                self.class_gt_counts[int(np.array(gtl))] += 1

        # record class score positive
        visited = [False] * len(gt_label)
        for b, s, l in zip(bbox, score, label):
            xmin, ymin, xmax, ymax = b.tolist()
            pred = [xmin, ymin, xmax, ymax]
            max_idx = -1
            max_overlap = -1.0
            for i, gl in enumerate(gt_label):
                if int(gl) == int(l):
                    overlap = jaccard_overlap(pred, gt_box[i],
                                              self.is_bbox_normalized)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = i

            if max_overlap > self.overlap_thresh:
                if self.evaluate_difficult or \
                        int(np.array(difficult[max_idx])) == 0:
                    if not visited[max_idx]:
                        self.class_score_poss[int(l)].append([s, 1.0])
                        visited[max_idx] = True
                    else:
                        self.class_score_poss[int(l)].append([s, 0.0])
            else:
                self.class_score_poss[int(l)].append([s, 0.0])

    def reset(self):
        """
        Reset metric statics
        """
        self.class_score_poss = [[] for _ in range(self.class_num)]
        self.class_gt_counts = [0] * self.class_num
        self.mAP = None

    def accumulate(self):
        """
        Accumulate metric results and calculate mAP
        """
        mAP = 0.
        valid_cnt = 0
        for score_pos, count in zip(self.class_score_poss,
                                    self.class_gt_counts):
            if count == 0: continue
            if len(score_pos) == 0:
                valid_cnt += 1
                continue

            accum_tp_list, accum_fp_list = \
                    self._get_tp_fp_accum(score_pos)
            precision = []
            recall = []
            for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
                precision.append(float(ac_tp) / (ac_tp + ac_fp))
                recall.append(float(ac_tp) / count)

            if self.map_type == '11point':
                max_precisions = [0.] * 11
                start_idx = len(precision) - 1
                for j in range(10, -1, -1):
                    for i in range(start_idx, -1, -1):
                        if recall[i] < float(j) / 10.:
                            start_idx = i
                            if j > 0:
                                max_precisions[j - 1] = max_precisions[j]
                                break
                        else:
                            if max_precisions[j] < precision[i]:
                                max_precisions[j] = precision[i]
                mAP += sum(max_precisions) / 11.
                valid_cnt += 1
            elif self.map_type == 'integral':
                import math
                ap = 0.
                prev_recall = 0.
                for i in range(len(precision)):
                    recall_gap = math.fabs(recall[i] - prev_recall)
                    if recall_gap > 1e-6:
                        ap += precision[i] * recall_gap
                        prev_recall = recall[i]
                mAP += ap
                valid_cnt += 1
            else:
                logger.error("Unspported mAP type {}".format(self.map_type))
                sys.exit(1)

        self.mAP = mAP / float(valid_cnt) if valid_cnt > 0 else mAP

    def get_map(self):
        """
        Get mAP result
        """
        if self.mAP is None:
            logger.error("mAP is not calculated.")
        return self.mAP

    def _get_tp_fp_accum(self, score_pos_list):
        """
        Calculate accumulating true/false positive results from
        [score, pos] records
        """
        sorted_list = sorted(score_pos_list, key=lambda s: s[0], reverse=True)
        accum_tp = 0
        accum_fp = 0
        accum_tp_list = []
        accum_fp_list = []
        for (score, pos) in sorted_list:
            accum_tp += int(pos)
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)
            accum_fp_list.append(accum_fp)
        return accum_tp_list, accum_fp_list
