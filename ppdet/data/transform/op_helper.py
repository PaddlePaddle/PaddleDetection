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
# this file contains helper methods for BBOX processing

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox[2] + src_bbox[0]) / 2
    center_y = (src_bbox[3] + src_bbox[1]) / 2
    if center_x >= sample_bbox[0] and \
            center_x <= sample_bbox[2] and \
            center_y >= sample_bbox[1] and \
            center_y <= sample_bbox[3]:
        return True
    return False


def clip_bbox(src_bbox):
    src_bbox[0] = max(min(src_bbox[0], 1.0), 0.0)
    src_bbox[1] = max(min(src_bbox[1], 1.0), 0.0)
    src_bbox[2] = max(min(src_bbox[2], 1.0), 0.0)
    src_bbox[3] = max(min(src_bbox[3], 1.0), 0.0)
    return src_bbox


def bbox_area(src_bbox):
    width = src_bbox[2] - src_bbox[0]
    height = src_bbox[3] - src_bbox[1]
    return width * height


def filter_and_process(sample_bbox, bboxes, labels, scores=None):
    new_bboxes = []
    new_labels = []
    new_scores = []
    for i in range(len(labels)):
        new_bbox = [0, 0, 0, 0]
        obj_bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
        if not meet_emit_constraint(obj_bbox, sample_bbox):
            continue
        sample_width = sample_bbox[2] - sample_bbox[0]
        sample_height = sample_bbox[3] - sample_bbox[1]
        new_bbox[0] = (obj_bbox[0] - sample_bbox[0]) / sample_width
        new_bbox[1] = (obj_bbox[1] - sample_bbox[1]) / sample_height
        new_bbox[2] = (obj_bbox[2] - sample_bbox[0]) / sample_width
        new_bbox[3] = (obj_bbox[3] - sample_bbox[1]) / sample_height
        new_bbox = clip_bbox(new_bbox)
        if bbox_area(new_bbox) > 0:
            new_bboxes.append(new_bbox)
            new_labels.append([labels[i][0]])
            if scores is not None:
                new_scores.append([scores[i][0]])
    bboxes = np.array(new_bboxes)
    labels = np.array(new_labels)
    scores = np.array(new_scores)
    return bboxes, labels, scores


def generate_sample_bbox(sampler):
    scale = np.random.uniform(sampler[2], sampler[3])
    aspect_ratio = np.random.uniform(sampler[4], sampler[5])
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))
    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = [xmin, ymin, xmax, ymax]
    return sampled_bbox


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
        sample_bbox[2] <= object_bbox[0] or \
        sample_bbox[1] >= object_bbox[3] or \
        sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler,
                              sample_bbox,
                              gt_bboxes,
                              satisfy_all=False):
    if sampler[6] == 0 and sampler[7] == 0:
        return True
    satisfied = []
    for i in range(len(gt_bboxes)):
        object_bbox = [
            gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]
        ]
        overlap = jaccard_overlap(sample_bbox, object_bbox)
        if sampler[6] != 0 and \
                overlap < sampler[6]:
            satisfied.append(False)
            continue
        if sampler[7] != 0 and \
                overlap > sampler[7]:
            satisfied.append(False)
            continue
        satisfied.append(True)
        if not satisfy_all:
            return True

    if satisfy_all:
        return np.all(satisfied)
    else:
        return False
