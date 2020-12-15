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

__all__ = ["bbox_overlaps", "box_to_delta"]

logger = logging.getLogger(__name__)


def bbox_overlaps(boxes_1, boxes_2):
    '''
    bbox_overlaps
        boxes_1: x1, y, x2, y2
        boxes_2: x1, y, x2, y2
    '''
    assert boxes_1.shape[1] == 4 and boxes_2.shape[1] == 4

    num_1 = boxes_1.shape[0]
    num_2 = boxes_2.shape[0]

    x1_1 = boxes_1[:, 0:1]
    y1_1 = boxes_1[:, 1:2]
    x2_1 = boxes_1[:, 2:3]
    y2_1 = boxes_1[:, 3:4]
    area_1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)

    x1_2 = boxes_2[:, 0].transpose()
    y1_2 = boxes_2[:, 1].transpose()
    x2_2 = boxes_2[:, 2].transpose()
    y2_2 = boxes_2[:, 3].transpose()
    area_2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    xx1 = np.maximum(x1_1, x1_2)
    yy1 = np.maximum(y1_1, y1_2)
    xx2 = np.minimum(x2_1, x2_2)
    yy2 = np.minimum(y2_1, y2_2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    ovr = inter / (area_1 + area_2 - inter)
    return ovr


def box_to_delta(ex_boxes, gt_boxes, weights):
    """ box_to_delta """
    ex_w = ex_boxes[:, 2] - ex_boxes[:, 0] + 1
    ex_h = ex_boxes[:, 3] - ex_boxes[:, 1] + 1
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_w
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    targets = np.vstack([dx, dy, dw, dh]).transpose()
    return targets
