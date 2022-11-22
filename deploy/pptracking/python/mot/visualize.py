# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import deque


def visualize_box_mask(im, results, labels, threshold=0.5):
    """
    Args:
        im (str/np.ndarray): path of image/np.ndarray read by cv2
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): Threshold of score.
    Returns:
        im (PIL.Image.Image): visualized image
    """
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    else:
        im = Image.fromarray(im)
    if 'boxes' in results and len(results['boxes']) > 0:
        im = draw_box(im, results['boxes'], labels, threshold=threshold)
    return im


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_box(im, np_boxes, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of box
    Returns:
        im (PIL.Image.Image): visualized image
    """
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color[clsid])

        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                  'right_bottom:[{:.2f},{:.2f}]'.format(
                      int(clsid), score, xmin, ymin, xmax, ymax))
            # draw bbox
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=draw_thickness,
                fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)

        # draw label
        text = "{} {:.4f}".format(labels[clsid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return im


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_tracking(image,
                  tlwhs,
                  obj_ids,
                  scores=None,
                  frame_id=0,
                  fps=0.,
                  ids2names=[],
                  do_entrance_counting=False,
                  entrance=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(0.5, image.shape[1] / 3000.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    cv2.putText(
        im,
        'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        (0, int(15 * text_scale) + 5),
        cv2.FONT_ITALIC,
        text_scale, (0, 0, 255),
        thickness=text_thickness)
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = 'ID: {}'.format(int(obj_id))
        if ids2names != []:
            assert len(
                ids2names) == 1, "plot_tracking only supports single classes."
            id_text = 'ID: {}_'.format(ids2names[0]) + id_text
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(
            im,
            id_text, (intbox[0], intbox[1] - 25),
            cv2.FONT_ITALIC,
            text_scale, (0, 255, 255),
            thickness=text_thickness)

        if scores is not None:
            text = 'score: {:.2f}'.format(float(scores[i]))
            cv2.putText(
                im,
                text, (intbox[0], intbox[1] - 6),
                cv2.FONT_ITALIC,
                text_scale, (0, 255, 0),
                thickness=text_thickness)
    if do_entrance_counting:
        entrance_line = tuple(map(int, entrance))
        cv2.rectangle(
            im,
            entrance_line[0:2],
            entrance_line[2:4],
            color=(0, 255, 255),
            thickness=line_thickness)
    return im


def plot_tracking_dict(image,
                       num_classes,
                       tlwhs_dict,
                       obj_ids_dict,
                       scores_dict,
                       frame_id=0,
                       fps=0.,
                       ids2names=[],
                       do_entrance_counting=False,
                       do_break_in_counting=False,
                       do_illegal_parking_recognition=False,
                       illegal_parking_dict=None,
                       entrance=None,
                       records=None,
                       center_traj=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    if do_break_in_counting or do_illegal_parking_recognition:
        entrance = np.array(entrance[:-1])  # last pair is [im_w, im_h] 

    text_scale = max(0.5, image.shape[1] / 3000.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    if num_classes == 1:
        if records is not None:
            start = records[-1].find('Total')
            end = records[-1].find('In')
            cv2.putText(
                im,
                records[-1][start:end], (0, int(40 * text_scale) + 10),
                cv2.FONT_ITALIC,
                text_scale, (0, 0, 255),
                thickness=text_thickness)

    if num_classes == 1 and do_entrance_counting:
        entrance_line = tuple(map(int, entrance))
        cv2.rectangle(
            im,
            entrance_line[0:2],
            entrance_line[2:4],
            color=(0, 255, 255),
            thickness=line_thickness)
        # find start location for entrance counting data
        start = records[-1].find('In')
        cv2.putText(
            im,
            records[-1][start:-1], (0, int(60 * text_scale) + 10),
            cv2.FONT_ITALIC,
            text_scale, (0, 0, 255),
            thickness=text_thickness)

    if num_classes == 1 and (do_break_in_counting or
                             do_illegal_parking_recognition):
        np_masks = np.zeros((im_h, im_w, 1), np.uint8)
        cv2.fillPoly(np_masks, [entrance], 255)

        # Draw region mask
        alpha = 0.3
        im = np.array(im).astype('float32')
        mask = np_masks[:, :, 0]
        color_mask = [0, 0, 255]
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
        im = np.array(im).astype('uint8')

        if do_break_in_counting:
            # find start location for break in counting data
            start = records[-1].find('Break_in')
            cv2.putText(
                im,
                records[-1][start:-1],
                (entrance[0][0] - 10, entrance[0][1] - 10),
                cv2.FONT_ITALIC,
                text_scale, (0, 0, 255),
                thickness=text_thickness)

        if illegal_parking_dict is not None and len(illegal_parking_dict) != 0:
            for key, value in illegal_parking_dict.items():
                x1, y1, w, h = value['bbox']
                plate = value['plate']
                if plate is None:
                    plate = ""

                # red box
                cv2.rectangle(im, (int(x1), int(y1)),
                              (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)

                cv2.putText(
                    im,
                    "illegal_parking:" + plate,
                    (int(x1) + 5, int(16 * text_scale + y1 + 15)),
                    cv2.FONT_ITALIC,
                    text_scale * 1.5, (0, 0, 255),
                    thickness=text_thickness)

    for cls_id in range(num_classes):
        tlwhs = tlwhs_dict[cls_id]
        obj_ids = obj_ids_dict[cls_id]
        scores = scores_dict[cls_id]
        cv2.putText(
            im,
            'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
            (0, int(15 * text_scale) + 5),
            cv2.FONT_ITALIC,
            text_scale, (0, 0, 255),
            thickness=text_thickness)

        record_id = set()
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            center = tuple(map(int, (x1 + w / 2., y1 + h / 2.)))
            obj_id = int(obj_ids[i])
            if center_traj is not None:
                record_id.add(obj_id)
                if obj_id not in center_traj[cls_id]:
                    center_traj[cls_id][obj_id] = deque(maxlen=30)
                center_traj[cls_id][obj_id].append(center)

            id_text = '{}'.format(int(obj_id))
            if ids2names != []:
                id_text = '{}_{}'.format(ids2names[cls_id], id_text)
            else:
                id_text = 'class{}_{}'.format(cls_id, id_text)

            _line_thickness = 1 if obj_id <= 0 else line_thickness

            in_region = False
            if do_break_in_counting:
                center_x = min(x1 + w / 2., im_w - 1)
                center_down_y = min(y1 + h, im_h - 1)
                if in_quadrangle([center_x, center_down_y], entrance, im_h,
                                 im_w):
                    in_region = True

            color = get_color(abs(obj_id)) if in_region == False else (0, 0,
                                                                       255)
            cv2.rectangle(
                im,
                intbox[0:2],
                intbox[2:4],
                color=color,
                thickness=line_thickness)
            cv2.putText(
                im,
                id_text, (intbox[0], intbox[1] - 25),
                cv2.FONT_ITALIC,
                text_scale,
                color,
                thickness=text_thickness)

            if do_break_in_counting and in_region:
                cv2.putText(
                    im,
                    'Break in now.', (intbox[0], intbox[1] - 50),
                    cv2.FONT_ITALIC,
                    text_scale, (0, 0, 255),
                    thickness=text_thickness)

            if scores is not None:
                text = 'score: {:.2f}'.format(float(scores[i]))
                cv2.putText(
                    im,
                    text, (intbox[0], intbox[1] - 6),
                    cv2.FONT_ITALIC,
                    text_scale,
                    color,
                    thickness=text_thickness)
        if center_traj is not None:
            for traj in center_traj:
                for i in traj.keys():
                    if i not in record_id:
                        continue
                    for point in traj[i]:
                        cv2.circle(im, point, 3, (0, 0, 255), -1)
    return im


def in_quadrangle(point, entrance, im_h, im_w):
    mask = np.zeros((im_h, im_w, 1), np.uint8)
    cv2.fillPoly(mask, [entrance], 255)
    p = tuple(map(int, point))
    if mask[p[1], p[0], :] > 0:
        return True
    else:
        return False
