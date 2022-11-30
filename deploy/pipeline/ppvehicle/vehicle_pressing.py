# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np
import math


class VehiclePressingRecognizer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def judge(self, Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):

        if (max(Ax1,Ax2)>=min(Bx1,Bx2) and min(Ax1,Ax2)<=max(Bx1,Bx2)) and \
           (max(Ay1,Ay2)>=min(By1,By2) and min(Ay1,Ay2)<=max(By1,By2)):

            if ((Bx1-Ax1)*(Ay2-Ay1)-(By1-Ay1)*(Ax2-Ax1)) * ((Bx2-Ax1)*(Ay2-Ay1)-(By2-Ay1)*(Ax2-Ax1))<=0 \
               and ((Ax1-Bx1)*(By2-By1)-(Ay1-By1)*(Bx2-Bx1)) * ((Ax2-Bx1)*(By2-By1)-(Ay2-By1)*(Bx2-Bx1)) <=0:
                return True
            else:
                return False
        else:
            return False

    def is_intersect(self, line, bbox):
        Ax1, Ay1, Ax2, Ay2 = line

        xmin, ymin, xmax, ymax = bbox

        bottom = self.judge(Ax1, Ay1, Ax2, Ay2, xmin, ymax, xmax, ymax)
        return bottom

    def run(self, lanes, det_res):
        intersect_bbox_list = []
        start_idx, boxes_num_i = 0, 0

        for i in range(len(lanes)):
            lane = lanes[i]
            if det_res is not None:
                det_res_i = {}
                boxes_num_i = det_res['boxes_num'][i]
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx +
                                                      boxes_num_i, :]
                intersect_bbox = []

                for line in lane:
                    for bbox in det_res_i['boxes']:
                        if self.is_intersect(line, bbox[2:]):
                            intersect_bbox.append(bbox)
                intersect_bbox_list.append(intersect_bbox)

                start_idx += boxes_num_i

        return intersect_bbox_list

    def mot_run(self, lanes, det_res):

        intersect_bbox_list = []
        if det_res is None:
            return intersect_bbox_list
        lanes_res = lanes['output']
        for i in range(len(lanes_res)):
            lane = lanes_res[i]
            for line in lane:
                for bbox in det_res:
                    if self.is_intersect(line, bbox[3:]):
                        intersect_bbox_list.append(bbox)
        return intersect_bbox_list