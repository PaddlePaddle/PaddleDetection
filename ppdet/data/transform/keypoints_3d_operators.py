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
    'CropAndFlipImages', 'ResizeImages', 'NormalizeImages', 'PermuteImages',
    'RandomFlipHalfBody3DTransformImages'
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
        images = records[
            "images"]  # RGB， Image读取的,(6, 1080, 1920, 3),(num_frames,h,w,c)
        images = images[:, :, ::-1, :]  # 图像左右翻转
        images = images[:, :, self.crop_range[0]:self.crop_range[
            1]]  #(6, 1080, 810, 3)，裁剪
        records["images"] = images

        # 2D kps处理
        # 1. 裁剪
        # 2. 点对应翻转
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
class ResizeImages(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True, 
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size, h,w
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(ResizeImages, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale
        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        images = sample['images']  # (1080, 607, 3)，裁剪过的图像

        # apply image
        im_shape = images[0].shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        resized_images = []
        for im in images:
            im = self.apply_image(im, [im_scale_x, im_scale_y])
            resized_images.append(im)

        sample['images'] = np.array(resized_images)

        # 2d keypoints resize
        if 'kps2d' in sample.keys():
            kps2d = sample['kps2d']
            kps2d[:, :, 0] = kps2d[:, :, 0] * im_scale_x
            kps2d[:, :, 1] = kps2d[:, :, 1] * im_scale_y

            sample['kps2d'] = kps2d

        return sample


@register_op
class NormalizeImages(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1],
                 is_scale=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImages, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def apply(self, sample, context=None):
        """Normalize the images.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        images = sample["images"]
        images = images.astype(np.float32, copy=False)

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            images = images / 255.0

        images -= mean
        images /= std

        sample["images"] = images

        return sample


@register_op
class PermuteImages(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (batch_size, C, H, W) #(6, 3, 1080, 1920)
        """
        super(PermuteImages, self).__init__()

    def apply(self, sample, context=None):
        images = sample["images"]
        images = images.transpose((0, 3, 1, 2))

        sample["images"] = images

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
            'images']  #kps3d, kps3d_vis, images. images.shape(num_frames, width, height, 3)

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

        records['images'] = images
        records['kps3d'] = joints
        records['kps3d_vis'] = joints_vis
        if kps2d is not None:
            records['kps2d'] = kps2d

        return records
