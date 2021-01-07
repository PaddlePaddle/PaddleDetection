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
import os
import cv2
import json
import math
import numpy as np
import argparse

HAT_SCALES = {
    '1.png': [3.0, 0.9, .0],
    '2.png': [3.0, 1.3, .5],
    '3.png': [2.2, 1.5, .8],
    '4.png': [2.2, 1.8, .0],
    '5.png': [1.8, 1.2, .0],
}

GLASSES_SCALES = {
    '1.png': [0.65, 2.5],
    '2.png': [0.65, 2.5],
}

BEARD_SCALES = {'1.png': [700, 0.3], '2.png': [220, 0.2]}


def rotate(image, angle):
    """
    angle is degree, not radian
    """
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    return cv2.warpAffine(image, M, (nW, nH))


def n_rotate_coord(angle, x, y):
    """
    angle is radian, not degree
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return rotatex, rotatey


def r_rotate_coord(angle, x, y):
    """
    angle is radian, not degree
    """
    rotatex = math.cos(angle) * x + math.sin(angle) * y
    rotatey = math.cos(angle) * y - math.sin(angle) * x
    return rotatex, rotatey


def add_beard(person, kypoint, element_path):
    beard_file_name = os.path.split(element_path)[1]
    # element_len: top width of beard
    # loc_offset_scale: scale relative to nose 
    element_len, loc_offset_scale = BEARD_SCALES[beard_file_name][:]

    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = kypoint[:]
    mouth_len = np.sqrt(np.square(np.abs(y4 - y5)) + np.square(x4 - x5))

    element = cv2.imread(element_path)
    h, w, _ = element.shape
    resize_scale = mouth_len / float(element_len)
    h, w = round(h * resize_scale + 0.5), round(w * resize_scale + 0.5)
    resized_element = cv2.resize(element, (w, h))
    resized_ele_h, resized_ele_w, _ = resized_element.shape

    # First find the keypoint of mouth in front face
    m_center_x = (x4 + x5) / 2.
    m_center_y = (y4 + y5) / 2.
    # cal degree only according mouth coordinates
    degree = np.arccos((x4 - x5) / mouth_len)

    # coordinate of RoI in front face
    half_w = int(resized_ele_w // 2)
    scale = loc_offset_scale
    roi_top_left_y = int(y3 + (((y5 + y4) // 2) - y3) * scale)
    roi_top_left_x = int(x3 - half_w)
    roi_top_right_y = roi_top_left_y
    roi_top_right_x = int(x3 + half_w)
    roi_bottom_left_y = roi_top_left_y + resized_ele_h
    roi_bottom_left_x = roi_top_left_x
    roi_bottom_right_y = roi_bottom_left_y
    roi_bottom_right_x = roi_top_right_x

    r_x11, r_y11 = roi_top_left_x - x3, roi_top_left_y - y3
    r_x12, r_y12 = roi_top_right_x - x3, roi_top_right_y - y3
    r_x21, r_y21 = roi_bottom_left_x - x3, roi_bottom_left_y - y3
    r_x22, r_y22 = roi_bottom_right_x - x3, roi_bottom_right_y - y3

    # coordinate of RoI in raw face
    if m_center_x > x3:
        x11, y11 = r_rotate_coord(degree, r_x11, r_y11)
        x12, y12 = r_rotate_coord(degree, r_x12, r_y12)
        x21, y21 = r_rotate_coord(degree, r_x21, r_y21)
        x22, y22 = r_rotate_coord(degree, r_x22, r_y22)
    else:
        x11, y11 = n_rotate_coord(degree, r_x11, r_y11)
        x12, y12 = n_rotate_coord(degree, r_x12, r_y12)
        x21, y21 = n_rotate_coord(degree, r_x21, r_y21)
        x22, y22 = n_rotate_coord(degree, r_x22, r_y22)

    x11, y11 = x11 + x3, y11 + y3
    x12, y12 = x12 + x3, y12 + y3
    x21, y21 = x21 + x3, y21 + y3
    x22, y22 = x22 + x3, y22 + y3

    min_x = int(min(x11, x12, x21, x22))
    max_x = int(max(x11, x12, x21, x22))
    min_y = int(min(y11, y12, y21, y22))
    max_y = int(max(y11, y12, y21, y22))

    angle = np.degrees(degree)

    if y4 < y5:
        angle = -angle

    rotated_element = rotate(resized_element, angle)

    rotated_ele_h, rotated_ele_w, _ = rotated_element.shape

    max_x = min_x + int(rotated_ele_w)
    max_y = min_y + int(rotated_ele_h)

    e2gray = cv2.cvtColor(rotated_element, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(e2gray, 238, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    roi = person[min_y:max_y, min_x:max_x]
    person_bg = cv2.bitwise_and(roi, roi, mask=mask)
    element_fg = cv2.bitwise_and(
        rotated_element, rotated_element, mask=mask_inv)

    dst = cv2.add(person_bg, element_fg)
    person[min_y:max_y, min_x:max_x] = dst
    return person


def add_hat(person, kypoint, element_path):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = kypoint[:]
    eye_len = np.sqrt(np.square(np.abs(y1 - y2)) + np.square(np.abs(x1 - x2)))
    # cal degree only according eye coordinates
    degree = np.arccos((x2 - x1) / eye_len)

    angle = np.degrees(degree)
    if y2 < y1:
        angle = -angle

    element = cv2.imread(element_path)
    hat_file_name = os.path.split(element_path)[1]
    # head_scale: size scale of hat
    # high_scale: height scale above the eyes
    # offect_scale: width offect of hat in face
    head_scale, high_scale, offect_scale = HAT_SCALES[hat_file_name][:]
    h, w, _ = element.shape

    element_len = w
    resize_scale = eye_len * head_scale / float(w)
    h, w = round(h * resize_scale + 0.5), round(w * resize_scale + 0.5)
    resized_element = cv2.resize(element, (w, h))
    resized_ele_h, resized_ele_w, _ = resized_element.shape

    m_center_x = (x1 + x2) / 2.
    m_center_y = (y1 + y2) / 2.

    head_len = int(eye_len * high_scale)

    if angle > 0:
        head_center_x = int(m_center_x + head_len * math.sin(degree))
        head_center_y = int(m_center_y - head_len * math.cos(degree))
    else:
        head_center_x = int(m_center_x + head_len * math.sin(degree))
        head_center_y = int(m_center_y - head_len * math.cos(degree))

    rotated_element = rotate(resized_element, angle)

    rotated_ele_h, rotated_ele_w, _ = rotated_element.shape
    max_x = int(head_center_x + (resized_ele_w // 2) * math.cos(degree)) + int(
        angle * head_scale) + int(eye_len * offect_scale)
    min_y = int(head_center_y - (resized_ele_w // 2) * math.cos(degree))

    pad_ele_x0 = 0 if (max_x - int(rotated_ele_w)) > 0 else -(
        max_x - int(rotated_ele_w))
    pad_ele_y0 = 0 if min_y > 0 else -(min_y)

    min_x = int(max(max_x - int(rotated_ele_w), 0))
    min_y = int(max(min_y, 0))
    max_y = min_y + int(rotated_ele_h)

    pad_y1 = max(max_y - int(person.shape[0]), 0)
    pad_x1 = max(max_x - int(person.shape[1]), 0)
    pad_w = pad_ele_x0 + pad_x1
    pad_h = pad_ele_y0 + pad_y1
    max_x += pad_w

    pad_person = np.zeros(
        (person.shape[0] + pad_h, person.shape[1] + pad_w, 3)).astype(np.uint8)

    pad_person[pad_ele_y0:pad_ele_y0 + person.shape[0], pad_ele_x0:pad_ele_x0 +
               person.shape[1], :] = person

    e2gray = cv2.cvtColor(rotated_element, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(e2gray, 1, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    roi = pad_person[min_y:max_y, min_x:max_x]

    person_bg = cv2.bitwise_and(roi, roi, mask=mask)
    element_fg = cv2.bitwise_and(
        rotated_element, rotated_element, mask=mask_inv)

    dst = cv2.add(person_bg, element_fg)
    pad_person[min_y:max_y, min_x:max_x] = dst

    return pad_person, pad_ele_x0, pad_x1, pad_ele_y0, pad_y1, min_x, min_y, max_x, max_y


def add_glasses(person, kypoint, element_path):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = kypoint[:]
    eye_len = np.sqrt(np.square(np.abs(y1 - y2)) + np.square(np.abs(x1 - x2)))
    # cal degree only according eye coordinates
    degree = np.arccos((x2 - x1) / eye_len)
    angle = np.degrees(degree)
    if y2 < y1:
        angle = -angle

    element = cv2.imread(element_path)
    glasses_file_name = os.path.split(element_path)[1]
    # height_scale: height scale above the eyes
    # glasses_scale: size ratio of glasses
    height_scale, glasses_scale = GLASSES_SCALES[glasses_file_name][:]
    h, w, _ = element.shape

    element_len = w
    resize_scale = eye_len * glasses_scale / float(element_len)
    h, w = round(h * resize_scale + 0.5), round(w * resize_scale + 0.5)
    resized_element = cv2.resize(element, (w, h))
    resized_ele_h, resized_ele_w, _ = resized_element.shape

    rotated_element = rotate(resized_element, angle)

    rotated_ele_h, rotated_ele_w, _ = rotated_element.shape

    eye_center_x = (x1 + x2) / 2.
    eye_center_y = (y1 + y2) / 2.

    min_x = int(eye_center_x) - int(rotated_ele_w * 0.5) + int(
        angle * glasses_scale * person.shape[1] / 2000)
    min_y = int(eye_center_y) - int(rotated_ele_h * height_scale)
    max_x = min_x + rotated_ele_w
    max_y = min_y + rotated_ele_h

    e2gray = cv2.cvtColor(rotated_element, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(e2gray, 1, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    roi = person[min_y:max_y, min_x:max_x]

    person_bg = cv2.bitwise_and(roi, roi, mask=mask)
    element_fg = cv2.bitwise_and(
        rotated_element, rotated_element, mask=mask_inv)

    dst = cv2.add(person_bg, element_fg)
    person[min_y:max_y, min_x:max_x] = dst
    return person
