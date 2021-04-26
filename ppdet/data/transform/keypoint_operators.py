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

# function:
#    operators to process sample,
#    eg: decode/resize/crop image

from __future__ import absolute_import

import cv2
import numpy as np
import math
import copy
import os

from ppdet.core.workspace import serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

registered_ops = []

__all__ = ['RandomAffineHrnet', 'InputTransform', 'ToHeatmapsHrnet']

def register_keypointop(cls):
    return serializable(cls)


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def get_affine_transform(center,
                         input_size,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(input_size) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale*200, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


@register_keypointop
class RandomAffineHrnet(object):
    """apply affine transform to image and coords
    to achieve the rotate, scale and flip effect for training image

    Args:
        trainsize (list):[w, h], Image target size
        upper_body_ids (list): The upper body joint ids
        flip_pairs (list): The left-right joints exchange order list
        pixel_std (int): The pixel std of the scale
        scale (float): The scale factor to transform the image
        rot (int): The rotate factor to transform the image
        num_joints_half_body (int): The joints threshold of the half body transform
        prob_half_body (float): The threshold of the half body transform
        flip (bool): Whether to flip the image
        records(dict): the dict contained the image and coords

    Returns:
        records(dict): contain the image and coords after tranformed

    """

    def __init__(self, trainsize, upper_body_ids, flip_pairs, pixel_std,
            scale=0.35, rot=45, num_joints_half_body=8, prob_half_body=0.3,
            flip=True):
        super(RandomAffineHrnet, self).__init__()
        self.trainsize = trainsize
        self.upper_body_ids = upper_body_ids
        self.flip_pairs = flip_pairs
        self.pixel_std = pixel_std
        self.scale = scale
        self.rot = rot
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body
        self.flip = flip
        self.aspect_ratio = trainsize[0] * 1.0 / trainsize[1]

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(joints.shape[0]):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])
        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints if len(lower_joints) > 2 else upper_joints
        if len(selected_joints) < 2:
            return None, None
        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]
        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)
        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )
        scale = scale * 1.5

        return center, scale

    def fliplr_joints(self, joints, joints_vis, width, matched_parts):
        joints[:, 0] = width - joints[:, 0] - 1
        for pair in matched_parts:
            joints[pair[0], :], joints[pair[1], :] = \
                joints[pair[1], :], joints[pair[0], :].copy()
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

        return joints*joints_vis, joints_vis

    def __call__(self, records):
        image = records['image']
        joints = records['joints']
        joints_vis = records['joints_vis']
        c = records['center']
        s = records['scale']
        r = 0
        if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
            c_half_body, s_half_body = self.half_body_transform(joints, joints_vis)
            if c_half_body is not None and s_half_body is not None:
                c, s = c_half_body, s_half_body
        sf = self.scale
        rf = self.rot
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn()*rf, -rf*2, rf*2) if np.random.random() <= 0.6 else 0
        if self.flip and np.random.random() <= 0.5:
            image = image[:, ::-1, :]
            joints, joints_vis = self.fliplr_joints(joints, joints_vis, image.shape[1], self.flip_pairs)
            c[0] = image.shape[1] - c[0] - 1
        records['image'] = image
        records['joints'] = joints
        records['joints_vis'] = joints_vis
        records['center'] = c
        records['scale'] = s

        return records


@register_keypointop
class InputTransform(object):
    """apply affine transform to image and coords

    Args:
        trainsize (list): [w, h], the standard size used to train
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after tranformed

    """
    def __init__(self, trainsize):
        self.trainsize = trainsize

    def __call__(self, records):
        image = records['image']
        joints = records['joints']
        joints_vis = records['joints_vis']
        trans = get_affine_transform(records['center'], records['scale']*200, 0, self.trainsize)
        image = cv2.warpAffine(image, trans,
            (int(self.trainsize[0]), int(self.trainsize[1])),
            flags=cv2.INTER_LINEAR)
        for i in range(joints.shape[0]):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        records['image'] = image
        records['joints'] = joints

        return records


@register_keypointop
class ToHeatmapsHrnet(object):
    """to generate the gaussin heatmaps of keypoint for heatmap loss

    Args:
        hmsize (list): [w, h] output heatmap's size
        sigma (float): the std of gaussin kernel genereted
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the heatmaps used to heatmaploss

    """

    def __init__(self, hmsize, sigma):
        super(ToHeatmapsHrnet, self).__init__()
        self.hmsize = np.array(hmsize)
        self.sigma = sigma

    def __call__(self, records):
        joints = records['joints']
        joints_vis = records['joints_vis']
        num_joints = joints.shape[0]
        image_size = np.array([records['image'].shape[1],
            records['image'].shape[0]])
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        target = np.zeros((num_joints,
                           self.hmsize[1],
                           self.hmsize[0]),
                           dtype=np.float32)
        tmp_size = self.sigma * 3
        for joint_id in range(num_joints):
            feat_stride = image_size / self.hmsize
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.hmsize[0] or ul[1] >= self.hmsize[1] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue
            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.hmsize[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.hmsize[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.hmsize[0])
            img_y = max(0, ul[1]), min(br[1], self.hmsize[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        records['target'] = target
        records['target_weight'] = target_weight
        del records['joints'], records['joints_vis']

        return records


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh

    Args:
        kpts_db (list): The predicted keypoints within the image
        thresh (float): The threshold to select the boxes
        sigmas (np.array): The variance to calculate the oks iou
            Default: None
        in_vis_thre (float): The threshold to select the high confidence boxes
            Default: None

    Return:
        keep (list): indexes to keep
    """

    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores


def soft_oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh

    Args:
        kpts_db (list): The predicted keypoints within the image
        thresh (float): The threshold to select the boxes
        sigmas (np.array): The variance to calculate the oks iou
            Default: None
        in_vis_thre (float): The threshold to select the high confidence boxes
            Default: None

    Return:
        keep (list): indexes to keep
    """

    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]
    scores = scores[order]

    # max_dets = order.size
    max_dets = 20
    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        order = order[1:]
        scores = rescore(oks_ovr, scores[1:], thresh)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep
