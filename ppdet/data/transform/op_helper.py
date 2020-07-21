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
import random
import math
import cv2


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
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def is_overlap(object_bbox, sample_bbox):
    if object_bbox[0] >= sample_bbox[2] or \
       object_bbox[2] <= sample_bbox[0] or \
       object_bbox[1] >= sample_bbox[3] or \
       object_bbox[3] <= sample_bbox[1]:
        return False
    else:
        return True


def filter_and_process(sample_bbox, bboxes, labels, scores=None,
                       keypoints=None):
    new_bboxes = []
    new_labels = []
    new_scores = []
    new_keypoints = []
    new_kp_ignore = []
    for i in range(len(bboxes)):
        new_bbox = [0, 0, 0, 0]
        obj_bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
        if not meet_emit_constraint(obj_bbox, sample_bbox):
            continue
        if not is_overlap(obj_bbox, sample_bbox):
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
            if keypoints is not None:
                sample_keypoint = keypoints[0][i]
                for j in range(len(sample_keypoint)):
                    kp_len = sample_height if j % 2 else sample_width
                    sample_coord = sample_bbox[1] if j % 2 else sample_bbox[0]
                    sample_keypoint[j] = (
                        sample_keypoint[j] - sample_coord) / kp_len
                    sample_keypoint[j] = max(min(sample_keypoint[j], 1.0), 0.0)
                new_keypoints.append(sample_keypoint)
                new_kp_ignore.append(keypoints[1][i])

    bboxes = np.array(new_bboxes)
    labels = np.array(new_labels)
    scores = np.array(new_scores)
    if keypoints is not None:
        keypoints = np.array(new_keypoints)
        new_kp_ignore = np.array(new_kp_ignore)
        return bboxes, labels, scores, (keypoints, new_kp_ignore)
    return bboxes, labels, scores


def bbox_area_sampling(bboxes, labels, scores, target_size, min_size):
    new_bboxes = []
    new_labels = []
    new_scores = []
    for i, bbox in enumerate(bboxes):
        w = float((bbox[2] - bbox[0]) * target_size)
        h = float((bbox[3] - bbox[1]) * target_size)
        if w * h < float(min_size * min_size):
            continue
        else:
            new_bboxes.append(bbox)
            new_labels.append(labels[i])
            if scores is not None and scores.size != 0:
                new_scores.append(scores[i])
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


def generate_sample_bbox_square(sampler, image_width, image_height):
    scale = np.random.uniform(sampler[2], sampler[3])
    aspect_ratio = np.random.uniform(sampler[4], sampler[5])
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))
    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)
    if image_height < image_width:
        bbox_width = bbox_height * image_height / image_width
    else:
        bbox_height = bbox_width * image_width / image_height
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = [xmin, ymin, xmax, ymax]
    return sampled_bbox


def data_anchor_sampling(bbox_labels, image_width, image_height, scale_array,
                         resize_width):
    num_gt = len(bbox_labels)
    # np.random.randint range: [low, high)
    rand_idx = np.random.randint(0, num_gt) if num_gt != 0 else 0

    if num_gt != 0:
        norm_xmin = bbox_labels[rand_idx][0]
        norm_ymin = bbox_labels[rand_idx][1]
        norm_xmax = bbox_labels[rand_idx][2]
        norm_ymax = bbox_labels[rand_idx][3]

        xmin = norm_xmin * image_width
        ymin = norm_ymin * image_height
        wid = image_width * (norm_xmax - norm_xmin)
        hei = image_height * (norm_ymax - norm_ymin)
        range_size = 0

        area = wid * hei
        for scale_ind in range(0, len(scale_array) - 1):
            if area > scale_array[scale_ind] ** 2 and area < \
                    scale_array[scale_ind + 1] ** 2:
                range_size = scale_ind + 1
                break

        if area > scale_array[len(scale_array) - 2]**2:
            range_size = len(scale_array) - 2

        scale_choose = 0.0
        if range_size == 0:
            rand_idx_size = 0
        else:
            # np.random.randint range: [low, high)
            rng_rand_size = np.random.randint(0, range_size + 1)
            rand_idx_size = rng_rand_size % (range_size + 1)

        if rand_idx_size == range_size:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = min(2.0 * scale_array[rand_idx_size],
                                 2 * math.sqrt(wid * hei))
            scale_choose = random.uniform(min_resize_val, max_resize_val)
        else:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = 2.0 * scale_array[rand_idx_size]
            scale_choose = random.uniform(min_resize_val, max_resize_val)

        sample_bbox_size = wid * resize_width / scale_choose

        w_off_orig = 0.0
        h_off_orig = 0.0
        if sample_bbox_size < max(image_height, image_width):
            if wid <= sample_bbox_size:
                w_off_orig = np.random.uniform(xmin + wid - sample_bbox_size,
                                               xmin)
            else:
                w_off_orig = np.random.uniform(xmin,
                                               xmin + wid - sample_bbox_size)

            if hei <= sample_bbox_size:
                h_off_orig = np.random.uniform(ymin + hei - sample_bbox_size,
                                               ymin)
            else:
                h_off_orig = np.random.uniform(ymin,
                                               ymin + hei - sample_bbox_size)

        else:
            w_off_orig = np.random.uniform(image_width - sample_bbox_size, 0.0)
            h_off_orig = np.random.uniform(image_height - sample_bbox_size, 0.0)

        w_off_orig = math.floor(w_off_orig)
        h_off_orig = math.floor(h_off_orig)

        # Figure out top left coordinates.
        w_off = float(w_off_orig / image_width)
        h_off = float(h_off_orig / image_height)

        sampled_bbox = [
            w_off, h_off, w_off + float(sample_bbox_size / image_width),
            h_off + float(sample_bbox_size / image_height)
        ]
        return sampled_bbox
    else:
        return 0


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


def intersect_bbox(bbox1, bbox2):
    if bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or \
        bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1]:
        intersection_box = [0.0, 0.0, 0.0, 0.0]
    else:
        intersection_box = [
            max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
            min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        ]
    return intersection_box


def bbox_coverage(bbox1, bbox2):
    inter_box = intersect_bbox(bbox1, bbox2)
    intersect_size = bbox_area(inter_box)

    if intersect_size > 0:
        bbox1_size = bbox_area(bbox1)
        return intersect_size / bbox1_size
    else:
        return 0.


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


def satisfy_sample_constraint_coverage(sampler, sample_bbox, gt_bboxes):
    if sampler[6] == 0 and sampler[7] == 0:
        has_jaccard_overlap = False
    else:
        has_jaccard_overlap = True
    if sampler[8] == 0 and sampler[9] == 0:
        has_object_coverage = False
    else:
        has_object_coverage = True

    if not has_jaccard_overlap and not has_object_coverage:
        return True
    found = False
    for i in range(len(gt_bboxes)):
        object_bbox = [
            gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]
        ]
        if has_jaccard_overlap:
            overlap = jaccard_overlap(sample_bbox, object_bbox)
            if sampler[6] != 0 and \
                    overlap < sampler[6]:
                continue
            if sampler[7] != 0 and \
                    overlap > sampler[7]:
                continue
            found = True
        if has_object_coverage:
            object_coverage = bbox_coverage(object_bbox, sample_bbox)
            if sampler[8] != 0 and \
                    object_coverage < sampler[8]:
                continue
            if sampler[9] != 0 and \
                    object_coverage > sampler[9]:
                continue
            found = True
        if found:
            return True
    return found


def crop_image_sampling(img, sample_bbox, image_width, image_height,
                        target_size):
    # no clipping here
    xmin = int(sample_bbox[0] * image_width)
    xmax = int(sample_bbox[2] * image_width)
    ymin = int(sample_bbox[1] * image_height)
    ymax = int(sample_bbox[3] * image_height)

    w_off = xmin
    h_off = ymin
    width = xmax - xmin
    height = ymax - ymin
    cross_xmin = max(0.0, float(w_off))
    cross_ymin = max(0.0, float(h_off))
    cross_xmax = min(float(w_off + width - 1.0), float(image_width))
    cross_ymax = min(float(h_off + height - 1.0), float(image_height))
    cross_width = cross_xmax - cross_xmin
    cross_height = cross_ymax - cross_ymin

    roi_xmin = 0 if w_off >= 0 else abs(w_off)
    roi_ymin = 0 if h_off >= 0 else abs(h_off)
    roi_width = cross_width
    roi_height = cross_height

    roi_y1 = int(roi_ymin)
    roi_y2 = int(roi_ymin + roi_height)
    roi_x1 = int(roi_xmin)
    roi_x2 = int(roi_xmin + roi_width)

    cross_y1 = int(cross_ymin)
    cross_y2 = int(cross_ymin + cross_height)
    cross_x1 = int(cross_xmin)
    cross_x2 = int(cross_xmin + cross_width)

    sample_img = np.zeros((height, width, 3))
    sample_img[roi_y1: roi_y2, roi_x1: roi_x2] = \
        img[cross_y1: cross_y2, cross_x1: cross_x2]

    sample_img = cv2.resize(
        sample_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return sample_img


def is_poly(segm):
    assert isinstance(segm, (list, dict)), \
        "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)


def gaussian_radius(bbox_size, min_overlap):
    height, width = bbox_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    radius1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    radius2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    radius3 = (b3 + sq3) / (2 * a3)
    return min(radius1, radius2, radius3)


def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    diameter = 2 * radius + 1
    sigma = diameter / delte
    gaussian = gaussian2D((diameter, diameter), sigma_x=sigma, sigma_y=sigma)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:
                               radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y *
                                                            sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
