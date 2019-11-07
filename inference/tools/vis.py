# coding: utf-8
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

import detection_result_pb2
import cv2
import sys
import gflags
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

Flags = gflags.FLAGS
gflags.DEFINE_string('img_path', 'abc', 'image path')
gflags.DEFINE_string('img_result_path', 'def', 'image result path')
gflags.DEFINE_float('threshold', 0.0, 'threshold of score')
gflags.DEFINE_string('c2l_path', 'ghk', 'class to label path')


def colormap(rgb=False):
    """
    Get colormap
    """
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
        0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
        0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
        0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
        0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python vis.py --img_path=/path/to/image --img_result_path=/path/to/image_result.pb --threshold=0.1 --c2l_path=/path/to/class2label.json"
        )
    else:
        Flags(sys.argv)
        color_list = colormap(rgb=True)
        text_thickness = 1
        text_scale = 0.3
        with open(Flags.img_result_path, "rb") as f:
            detection_result = detection_result_pb2.DetectionResult()
            detection_result.ParseFromString(f.read())
            img = cv2.imread(Flags.img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            class2LabelMap = dict()
            with open(Flags.c2l_path, "r", encoding="utf-8") as json_f:
                class2LabelMap = json.load(json_f)
                for box in detection_result.detection_boxes:
                    if box.score >= Flags.threshold:
                        box_class = getattr(box, 'class')
                        text_class_score_str = "%s %.2f" % (class2LabelMap.get(
                            str(box_class)), box.score)
                        text_point = (int(box.left_top_x), int(box.left_top_y))

                        ptLeftTop = (int(box.left_top_x), int(box.left_top_y))
                        ptRightBottom = (int(box.right_bottom_x),
                                         int(box.right_bottom_y))
                        box_thickness = 1
                        color = tuple([int(c) for c in color_list[box_class]])
                        cv2.rectangle(img, ptLeftTop, ptRightBottom, color,
                                      box_thickness, 8)
                        if text_point[1] < 0:
                            text_point = (int(box.left_top_x),
                                          int(box.right_bottom_y))
                        WHITE = (255, 255, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_size = cv2.getTextSize(text_class_score_str, font,
                                                    text_scale, text_thickness)

                        text_box_left_top = (text_point[0],
                                             text_point[1] - text_size[0][1])
                        text_box_right_bottom = (text_point[0] +
                                                 text_size[0][0], text_point[1])

                        cv2.rectangle(img, text_box_left_top,
                                      text_box_right_bottom, color, -1, 8)
                        cv2.putText(img, text_class_score_str, text_point, font,
                                    text_scale, WHITE, text_thickness)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(Flags.img_path + ".png", img)
