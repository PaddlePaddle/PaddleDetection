# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
import cv2
import numpy as np
import math
import copy
import random
import uuid
from numbers import Number, Integral

from ...modeling.keypoint_utils import get_affine_mat_kernel, warp_affine_joints, get_affine_transform, affine_transform, get_warp_matrix
from ppdet.core.workspace import serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

registered_ops = []

__all__ = [
    'CropAndFlipImages', 'PermuteImages', 'RandomFlipHalfBody3DTransformImages'
]

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D


def register_keypointop(cls):
    return serializable(cls)


def register_op(cls):
    registered_ops.append(cls.__name__)
    if not hasattr(BaseOperator, cls.__name__):
        setattr(BaseOperator, cls.__name__, cls)
    else:
        raise KeyError("The {} class has been registered.".format(cls.__name__))
    return serializable(cls)


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        if isinstance(sample, Sequence):  # for batch_size
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            # image.shape changed
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


@register_keypointop
class CropAndFlipImages(object):
    """Crop all images"""

    def __init__(self, crop_range, flip_pairs=None):
        super(CropAndFlipImages, self).__init__()
        self.crop_range = crop_range
        self.flip_pairs = flip_pairs

    def __call__(self, records):  # tuple
        images = records["image"]
        images = images[:, :, ::-1, :]
        images = images[:, :, self.crop_range[0]:self.crop_range[1]]
        records["image"] = images

        if "kps2d" in records.keys():
            kps2d = records["kps2d"]

            width, height = images.shape[2], images.shape[1]
            kps2d = np.array(kps2d)
            kps2d[:, :, 0] = kps2d[:, :, 0] - self.crop_range[0]

            for pair in self.flip_pairs:
                kps2d[:, pair[0], :], kps2d[:,pair[1], :] = \
                    kps2d[:,pair[1], :], kps2d[:,pair[0], :].copy()

            records["kps2d"] = kps2d

        return records


@register_op
class PermuteImages(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (batch_size, C, H, W) #(6, 3, 1080, 1920)
        """
        super(PermuteImages, self).__init__()

    def apply(self, sample, context=None):
        images = sample["image"]
        images = images.transpose((0, 3, 1, 2))

        sample["image"] = images

        return sample


@register_keypointop
class RandomFlipHalfBody3DTransformImages(object):
    """apply data augment to images and coords
    to achieve the flip, scale, rotate and half body transform effect for training image
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
    Returns:
        records(dict): contain the image and coords after tranformed
    """

    def __init__(self,
                 trainsize,
                 upper_body_ids,
                 flip_pairs,
                 pixel_std,
                 scale=0.35,
                 rot=40,
                 num_joints_half_body=8,
                 prob_half_body=0.3,
                 flip=True,
                 rot_prob=0.6,
                 do_occlusion=False):
        super(RandomFlipHalfBody3DTransformImages, self).__init__()
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
        self.rot_prob = rot_prob
        self.do_occlusion = do_occlusion

    def halfbody_transform(self, joints, joints_vis):
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
            selected_joints = lower_joints if len(
                lower_joints) > 2 else upper_joints
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
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        scale = scale * 1.5

        return center, scale

    def flip_joints(self, joints, joints_vis, width, matched_parts, kps2d=None):
        # joints: (6, 24, 3),(num_frames, num_joints, 3)

        joints[:, :, 0] = width - joints[:, :, 0] - 1  # x
        if kps2d is not None:
            kps2d[:, :, 0] = width - kps2d[:, :, 0] - 1

        for pair in matched_parts:
            joints[:, pair[0], :], joints[:,pair[1], :] = \
                joints[:,pair[1], :], joints[:,pair[0], :].copy()

            joints_vis[:,pair[0], :], joints_vis[:,pair[1], :] = \
                joints_vis[:,pair[1], :], joints_vis[:,pair[0], :].copy()

            if kps2d is not None:
                kps2d[:, pair[0], :], kps2d[:,pair[1], :] = \
                    kps2d[:,pair[1], :], kps2d[:,pair[0], :].copy()

        # move to zero
        joints -= joints[:, [0], :]  # (batch_size, 24, 3),numpy.ndarray

        return joints, joints_vis, kps2d

    def __call__(self, records):
        images = records[
            'image']  #kps3d, kps3d_vis, images. images.shape(num_frames, width, height, 3)

        joints = records['kps3d']
        joints_vis = records['kps3d_vis']

        kps2d = None
        if 'kps2d' in records.keys():
            kps2d = records['kps2d']

        if self.flip and np.random.random() <= 0.5:
            images = images[:, :, ::-1, :]  # 图像水平翻转 (6, 1080, 810, 3)
            joints, joints_vis, kps2d = self.flip_joints(
                joints, joints_vis, images.shape[2], self.flip_pairs,
                kps2d)  # 关键点左右对称翻转
        occlusion = False
        if self.do_occlusion and random.random() <= 0.5:  # 随机遮挡
            height = images[0].shape[0]
            width = images[0].shape[1]
            occlusion = True
            while True:
                area_min = 0.0
                area_max = 0.2
                synth_area = (random.random() *
                              (area_max - area_min) + area_min) * width * height

                ratio_min = 0.3
                ratio_max = 1 / 0.3
                synth_ratio = (random.random() *
                               (ratio_max - ratio_min) + ratio_min)

                synth_h = math.sqrt(synth_area * synth_ratio)
                synth_w = math.sqrt(synth_area / synth_ratio)
                synth_xmin = random.random() * (width - synth_w - 1)
                synth_ymin = random.random() * (height - synth_h - 1)

                if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < width and synth_ymin + synth_h < height:
                    xmin = int(synth_xmin)
                    ymin = int(synth_ymin)
                    w = int(synth_w)
                    h = int(synth_h)

                    mask = np.random.rand(h, w, 3) * 255
                    images[:, ymin:ymin + h, xmin:xmin + w, :] = mask[
                        None, :, :, :]
                    break

        records['image'] = images
        records['kps3d'] = joints
        records['kps3d_vis'] = joints_vis
        if kps2d is not None:
            records['kps2d'] = kps2d

        return records
