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

import numpy as np
import math


class VehicleRetrogradeRecognizer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.filter_horizontal_flag = self.cfg['filter_horizontal_flag']
        self.deviation = self.cfg['deviation']
        self.move_scale = self.cfg['move_scale']
        self.keep_right_flag = self.cfg['keep_right_flag']
        self.center_traj_retrograde = [{}]  #retrograde recognizer record use
        self.fence_line = None if len(self.cfg[
            'fence_line']) == 0 else self.cfg['fence_line']

    def update_center_traj(self, mot_res, max_len):
        from collections import deque, defaultdict
        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_ids[0] = ids

        if mot_res is not None:
            for cls_id in range(num_classes):
                tlwhs = online_tlwhs[cls_id]
                obj_ids = online_ids[cls_id]
                for i, tlwh in enumerate(tlwhs):
                    x1, y1, w, h = tlwh
                    center = tuple(map(int, (x1 + w / 2., y1 + h)))
                    obj_id = int(obj_ids[i])
                    if self.center_traj_retrograde is not None:
                        if obj_id not in self.center_traj_retrograde[cls_id]:
                            self.center_traj_retrograde[cls_id][obj_id] = deque(
                                maxlen=max_len)
                        self.center_traj_retrograde[cls_id][obj_id].append(
                            center)

    def get_angle(self, array):

        x1, y1, x2, y2 = array
        a_x = x2 - x1
        a_y = y2 - y1
        angle1 = math.atan2(a_y, a_x)
        angle1 = int(angle1 * 180 / math.pi)

        a_x = x2 - x1 if y2 >= y1 else x1 - x2
        a_y = y2 - y1 if y2 >= y1 else y1 - y2
        angle2 = math.atan2(a_y, a_x)
        angle2 = int(angle2 * 180 / math.pi)
        if angle2 > 90:
            angle2 = 180 - angle2

        return angle1, angle2

    def is_move(self, array, frame_shape):
        x1, y1, x2, y2 = array
        h, w, _ = frame_shape

        if abs(x1 - x2) > w * self.move_scale or abs(y1 -
                                                     y2) > h * self.move_scale:
            return True
        else:
            return False

    def get_distance_point2line(self, point, line):

        line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 -
                                                                 line_point2)

        return distance

    def driving_direction(self, line1, line2, is_init=False):
        x1, y1 = line1[2] - line1[0], line1[3] - line1[1]
        x2, y2 = line2[0] - line1[0], line2[1] - line1[1]
        result = x1 * y2 - x2 * y1

        distance = self.get_distance_point2line([x2, y2], line1)

        if result < 0:
            result = 1
        elif result == 0:
            if line2[3] >= line2[1]:
                return -1
            else:
                return 1
        else:
            result = -1

        return result, distance

    def get_long_fence_line(self, h, w, line):

        x1, y1, x2, y2 = line
        if x1 == x2:
            return [x1, 0, x1, h]
        if y1 == y2:
            return [0, y1, w, y1]
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1

        if k == 1 and b == 0:
            return [0, 0, w, h]
        if k == -1 and b == 0:
            return [w, 0, h, h]

        top = [-b / k, 0]
        left = [0, b]
        right = [w, k * w + b]
        bottom = [(h - b) / k, h]
        candidate = np.array([top, left, right, bottom])

        flag = np.array([0, 0, 0, 0])

        if top[0] >= 0 and top[0] <= w:
            flag[0] = 1
        if left[1] > 0 and left[1] <= h:
            flag[1] = 1
        if right[1] > 0 and right[1] <= h:
            flag[2] = 1
        if bottom[0] > 0 and bottom[0] < w:
            flag[3] = 1

        ind = np.where(flag == 1)
        candidate = candidate[ind]
        candidate_sort = candidate[candidate[:, 1].argsort()]

        return [
            int(candidate_sort[0][0]), int(candidate_sort[0][1]),
            int(candidate_sort[1][0]), int(candidate_sort[1][1])
        ]

    def init_fence_line(self, lanes, pos_dir_traj, neg_dir_traj, frame_shape):

        fence_lines_candidate = None
        h, w, _ = frame_shape
        abs_distance = h * h + w * w

        for lane in lanes[0]:
            pos_dir_distansce = h * h + w * w
            neg_dir_distansce = h * h + w * w
            pos_dir = 0
            neg_dir = 0

            for traj_line in pos_dir_traj:
                dir_result, distansce = self.driving_direction(
                    lane, traj_line['traj_line'])
                if dir_result > 0:
                    pos_dir_distansce = distansce if distansce < pos_dir_distansce else pos_dir_distansce
                    pos_dir = 1
                else:
                    neg_dir_distansce = distansce if distansce < neg_dir_distansce else neg_dir_distansce
                    neg_dir = 1

            if pos_dir > 0 and neg_dir > 0:
                continue

            for traj_line in neg_dir_traj:

                dir_result, distansce = self.driving_direction(
                    lane, traj_line['traj_line'])

                if dir_result > 0:
                    pos_dir_distansce = distansce if distansce < pos_dir_distansce else pos_dir_distansce
                    pos_dir = 1
                else:
                    neg_dir_distansce = distansce if distansce < neg_dir_distansce else neg_dir_distansce
                    neg_dir = 1

            if pos_dir > 0 and neg_dir > 0:
                diff_dir_distance = abs(pos_dir_distansce - neg_dir_distansce)
                if diff_dir_distance < abs_distance:
                    fence_lines_candidate = lane
                    abs_distance = diff_dir_distance

        if fence_lines_candidate is None:
            return None

        fence_lines_candidate = self.get_long_fence_line(h, w,
                                                         fence_lines_candidate)

        return fence_lines_candidate

    def judge_retrograde(self, traj_line):

        line1 = self.fence_line
        x1, y1 = line1[2] - line1[0], line1[3] - line1[1]

        line2 = traj_line['traj_line']
        x2_start_point, y2_start_point = line2[0] - line1[0], line2[1] - line1[
            1]
        x2_end_point, y2_end_point = line2[2] - line1[0], line2[3] - line1[1]

        start_point_dir = x1 * y2_start_point - x2_start_point * y1
        end_point_dir = x1 * y2_end_point - x2_end_point * y1

        if start_point_dir < 0:
            start_point_dir = 1

        elif start_point_dir == 0:
            if line2[3] >= line2[1]:
                start_point_dir = -1
            else:
                start_point_dir = 1
        else:
            start_point_dir = -1

        if end_point_dir < 0:
            end_point_dir = 1

        elif end_point_dir == 0:
            if line2[3] >= line2[1]:
                end_point_dir = -1
            else:
                end_point_dir = 1
        else:
            end_point_dir = -1

        if self.keep_right_flag:
            driver_dir = -1 if (line2[3] - line2[1]) >= 0 else 1
        else:
            driver_dir = -1 if (line2[3] - line2[1]) <= 0 else 1

        return start_point_dir == driver_dir and start_point_dir == end_point_dir

    def mot_run(self, lanes_res, det_res, frame_shape):

        det = det_res['boxes']
        directions = lanes_res['directions']
        lanes = lanes_res['output']
        if len(directions) > 0:
            direction = directions[0]
        else:
            return [], self.fence_line

        if len(det) == 0:
            return [], self.fence_line

        traj_lines = []
        pos_dir_traj = []
        neg_dir_traj = []
        for i in range(len(det)):
            class_id = int(det[i][1])
            mot_id = int(det[i][0])
            traj_i = self.center_traj_retrograde[class_id][mot_id]
            if len(traj_i) < 2:
                continue

            traj_line = {
                'index': i,
                'mot_id': mot_id,
                'traj_line':
                [traj_i[0][0], traj_i[0][1], traj_i[-1][0], traj_i[-1][1]]
            }

            if not self.is_move(traj_line['traj_line'], frame_shape):
                continue
            angle, angle_deviation = self.get_angle(traj_line['traj_line'])
            if direction is not None and self.filter_horizontal_flag:
                if abs(angle_deviation - direction) > self.deviation:
                    continue

            traj_line['angle'] = angle
            traj_lines.append(traj_line)

            if self.fence_line is None:
                if angle >= 0:
                    pos_dir_traj.append(traj_line)
                else:
                    neg_dir_traj.append(traj_line)

        if len(traj_lines) == 0:
            return [], self.fence_line

        if self.fence_line is None:

            if len(pos_dir_traj) < 1 or len(neg_dir_traj) < 1:
                return [], None

            self.fence_line = self.init_fence_line(lanes, pos_dir_traj,
                                                   neg_dir_traj, frame_shape)
            return [], self.fence_line

        else:
            retrograde_list = []
            for traj_line in traj_lines:
                if self.judge_retrograde(traj_line) == False:
                    retrograde_list.append(det[traj_line['index']][0])

            return retrograde_list, self.fence_line
