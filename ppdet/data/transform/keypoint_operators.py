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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import cv2
import numpy as np
import math
import copy
import os

from ...modeling.keypoint_utils import get_affine_mat_kernel, warp_affine_joints
from ppdet.core.workspace import serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

registered_ops = []

__all__ = [
    'RandomAffine', 'KeyPointFlip', 'TagGenerate', 'ToHeatmaps',
    'NormalizePermute', 'EvalAffine'
]


def register_keypointop(cls):
    return serializable(cls)


@register_keypointop
class KeyPointFlip(object):
    """Get the fliped image by flip_prob. flip the coords also
    the left coords and right coords should exchange while flip, for the right keypoint will be left keypoint after image fliped

    Args:
        flip_permutation (list[17]): the left-right exchange order list corresponding to [0,1,2,...,16]
        hmsize (list[2]): output heatmap's shape list of different scale outputs of higherhrnet
        flip_prob (float): the ratio whether to flip the image
        records(dict): the dict contained the image, mask and coords

    Returns:
        records(dict): contain the image, mask and coords after tranformed

    """

    def __init__(self, flip_permutation, hmsize, flip_prob=0.5):
        super(KeyPointFlip, self).__init__()
        assert isinstance(flip_permutation, Sequence)
        self.flip_permutation = flip_permutation
        self.flip_prob = flip_prob
        self.hmsize = hmsize

    def __call__(self, records):
        image = records['image']
        kpts_lst = records['joints']
        mask_lst = records['mask']
        flip = np.random.random() < self.flip_prob
        if flip:
            image = image[:, ::-1]
            for idx, hmsize in enumerate(self.hmsize):
                if len(mask_lst) > idx:
                    mask_lst[idx] = mask_lst[idx][:, ::-1]
                if kpts_lst[idx].ndim == 3:
                    kpts_lst[idx] = kpts_lst[idx][:, self.flip_permutation]
                else:
                    kpts_lst[idx] = kpts_lst[idx][self.flip_permutation]
                kpts_lst[idx][..., 0] = hmsize - kpts_lst[idx][..., 0]
                kpts_lst[idx] = kpts_lst[idx].astype(np.int64)
                kpts_lst[idx][kpts_lst[idx][..., 0] >= hmsize, 2] = 0
                kpts_lst[idx][kpts_lst[idx][..., 1] >= hmsize, 2] = 0
                kpts_lst[idx][kpts_lst[idx][..., 0] < 0, 2] = 0
                kpts_lst[idx][kpts_lst[idx][..., 1] < 0, 2] = 0
        records['image'] = image
        records['joints'] = kpts_lst
        records['mask'] = mask_lst
        return records


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * math.cos(theta) + 0.5 * size_input[1] *
        math.sin(theta) + 0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * math.sin(theta) - 0.5 * size_input[1] *
        math.cos(theta) + 0.5 * size_target[1])
    return matrix


@register_keypointop
class RandomAffine(object):
    """apply affine transform to image, mask and coords
    to achieve the rotate, scale and shift effect for training image

    Args:
        max_degree (float): the max abslute rotate degree to apply, transform range is [-max_degree, max_degree]
        max_scale (list[2]): the scale range to apply, transform range is [min, max]
        max_shift (float): the max abslute shift ratio to apply, transform range is [-max_shift*imagesize, max_shift*imagesize]
        hmsize (list[2]): output heatmap's shape list of different scale outputs of higherhrnet
        trainsize (int): the standard length used to train, the 'scale_type' of [h,w] will be resize to trainsize for standard
        scale_type (str): the length of [h,w] to used for trainsize, chosed between 'short' and 'long'
        records(dict): the dict contained the image, mask and coords

    Returns:
        records(dict): contain the image, mask and coords after tranformed

    """

    def __init__(self,
                 max_degree=30,
                 scale=[0.75, 1.5],
                 max_shift=0.2,
                 hmsize=[128, 256],
                 trainsize=512,
                 scale_type='short'):
        super(RandomAffine, self).__init__()
        self.max_degree = max_degree
        self.min_scale = scale[0]
        self.max_scale = scale[1]
        self.max_shift = max_shift
        self.hmsize = hmsize
        self.trainsize = trainsize
        self.scale_type = scale_type

    def _get_affine_matrix(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        h = scale
        t = np.zeros((3, 3), dtype=np.float32)
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if rot != 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3), dtype=np.float32)
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def __call__(self, records):
        image = records['image']
        keypoints = records['joints']
        heatmap_mask = records['mask']

        degree = (np.random.random() * 2 - 1) * self.max_degree
        shape = np.array(image.shape[:2][::-1])
        center = center = np.array((np.array(shape) / 2))

        aug_scale = np.random.random() * (self.max_scale - self.min_scale
                                          ) + self.min_scale
        if self.scale_type == 'long':
            scale = max(shape[0], shape[1]) / 1.0
        elif self.scale_type == 'short':
            scale = min(shape[0], shape[1]) / 1.0
        else:
            raise ValueError('Unknown scale type: {}'.format(self.scale_type))
        roi_size = aug_scale * scale
        dx = int(0)
        dy = int(0)
        if self.max_shift > 0:

            dx = np.random.randint(-self.max_shift * roi_size,
                                   self.max_shift * roi_size)
            dy = np.random.randint(-self.max_shift * roi_size,
                                   self.max_shift * roi_size)

        center += np.array([dx, dy])
        input_size = 2 * center

        keypoints[..., :2] *= shape
        heatmap_mask *= 255
        kpts_lst = []
        mask_lst = []

        image_affine_mat = self._get_affine_matrix(
            center, roi_size, (self.trainsize, self.trainsize), degree)[:2]
        image = cv2.warpAffine(
            image,
            image_affine_mat, (self.trainsize, self.trainsize),
            flags=cv2.INTER_LINEAR)
        for hmsize in self.hmsize:
            kpts = copy.deepcopy(keypoints)
            mask_affine_mat = self._get_affine_matrix(
                center, roi_size, (hmsize, hmsize), degree)[:2]
            if heatmap_mask is not None:
                mask = cv2.warpAffine(heatmap_mask, mask_affine_mat,
                                      (hmsize, hmsize))
                mask = ((mask / 255) > 0.5).astype(np.float32)
            kpts[..., 0:2] = warp_affine_joints(kpts[..., 0:2].copy(),
                                                mask_affine_mat)
            kpts[np.trunc(kpts[..., 0]) >= hmsize, 2] = 0
            kpts[np.trunc(kpts[..., 1]) >= hmsize, 2] = 0
            kpts[np.trunc(kpts[..., 0]) < 0, 2] = 0
            kpts[np.trunc(kpts[..., 1]) < 0, 2] = 0
            kpts_lst.append(kpts)
            mask_lst.append(mask)
        records['image'] = image
        records['joints'] = kpts_lst
        records['mask'] = mask_lst
        return records


@register_keypointop
class EvalAffine(object):
    """apply affine transform to image
    resize the short of [h,w] to standard size for eval

    Args:
        size (int): the standard length used to train, the 'short' of [h,w] will be resize to trainsize for standard
        records(dict): the dict contained the image, mask and coords

    Returns:
        records(dict): contain the image, mask and coords after tranformed

    """

    def __init__(self, size, stride=64):
        super(EvalAffine, self).__init__()
        self.size = size
        self.stride = stride

    def __call__(self, records):
        image = records['image']
        mask = records['mask'] if 'mask' in records else None
        s = self.size
        h, w, _ = image.shape
        trans, size_resized = get_affine_mat_kernel(h, w, s, inv=False)
        image_resized = cv2.warpAffine(image, trans, size_resized)
        if mask is not None:
            mask = cv2.warpAffine(mask, trans, size_resized)
            records['mask'] = mask
        if 'joints' in records:
            del records['joints']
        records['image'] = image_resized
        return records


@register_keypointop
class NormalizePermute(object):
    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.120, 57.375],
                 is_scale=True):
        super(NormalizePermute, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale

    def __call__(self, records):
        image = records['image']
        image = image.astype(np.float32)
        if self.is_scale:
            image /= 255.
        image = image.transpose((2, 0, 1))
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        invstd = 1. / std
        for v, m, s in zip(image, mean, invstd):
            v.__isub__(m).__imul__(s)
        records['image'] = image
        return records


@register_keypointop
class TagGenerate(object):
    """record gt coords for aeloss to sample coords value in tagmaps

    Args:
        num_joints (int): the keypoint numbers of dataset to train
        num_people (int): maxmum people to support for sample aeloss
        records(dict): the dict contained the image, mask and coords

    Returns:
        records(dict): contain the gt coords used in tagmap

    """

    def __init__(self, num_joints, max_people=30):
        super(TagGenerate, self).__init__()
        self.max_people = max_people
        self.num_joints = num_joints

    def __call__(self, records):
        kpts_lst = records['joints']
        kpts = kpts_lst[0]
        tagmap = np.zeros((self.max_people, self.num_joints, 4), dtype=np.int64)
        inds = np.where(kpts[..., 2] > 0)
        p, j = inds[0], inds[1]
        visible = kpts[inds]
        # tagmap is [p, j, 3], where last dim is j, y, x
        tagmap[p, j, 0] = j
        tagmap[p, j, 1] = visible[..., 1]  # y
        tagmap[p, j, 2] = visible[..., 0]  # x
        tagmap[p, j, 3] = 1
        records['tagmap'] = tagmap
        del records['joints']
        return records


@register_keypointop
class ToHeatmaps(object):
    """to generate the gaussin heatmaps of keypoint for heatmap loss

    Args:
        num_joints (int): the keypoint numbers of dataset to train
        hmsize (list[2]): output heatmap's shape list of different scale outputs of higherhrnet
        sigma (float): the std of gaussin kernel genereted
        records(dict): the dict contained the image, mask and coords

    Returns:
        records(dict): contain the heatmaps used to heatmaploss

    """

    def __init__(self, num_joints, hmsize, sigma=None):
        super(ToHeatmaps, self).__init__()
        self.num_joints = num_joints
        self.hmsize = np.array(hmsize)
        if sigma is None:
            sigma = hmsize[0] // 64
        self.sigma = sigma

        r = 6 * sigma + 3
        x = np.arange(0, r, 1, np.float32)
        y = x[:, None]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, records):
        kpts_lst = records['joints']
        mask_lst = records['mask']
        for idx, hmsize in enumerate(self.hmsize):
            mask = mask_lst[idx]
            kpts = kpts_lst[idx]
            heatmaps = np.zeros((self.num_joints, hmsize, hmsize))
            inds = np.where(kpts[..., 2] > 0)
            visible = kpts[inds].astype(np.int64)[..., :2]
            ul = np.round(visible - 3 * self.sigma - 1)
            br = np.round(visible + 3 * self.sigma + 2)
            sul = np.maximum(0, -ul)
            sbr = np.minimum(hmsize, br) - ul
            dul = np.clip(ul, 0, hmsize - 1)
            dbr = np.clip(br, 0, hmsize)
            for i in range(len(visible)):
                dx1, dy1 = dul[i]
                dx2, dy2 = dbr[i]
                sx1, sy1 = sul[i]
                sx2, sy2 = sbr[i]
                heatmaps[inds[1][i], dy1:dy2, dx1:dx2] = np.maximum(
                    self.gaussian[sy1:sy2, sx1:sx2],
                    heatmaps[inds[1][i], dy1:dy2, dx1:dx2])
            records['heatmap_gt{}x'.format(idx + 1)] = heatmaps
            records['mask_{}x'.format(idx + 1)] = mask
        del records['mask']
        return records
