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

from ...modeling.keypoint_utils import get_affine_mat_kernel, warp_affine_joints, get_affine_transform, affine_transform, get_warp_matrix
from ppdet.core.workspace import serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

registered_ops = []

__all__ = [
    'RandomAffine',
    'KeyPointFlip',
    'TagGenerate',
    'ToHeatmaps',
    'NormalizePermute',
    'EvalAffine',
    'RandomFlipHalfBodyTransform',
    'TopDownAffine',
    'ToHeatmapsTopDown',
    'ToHeatmapsTopDown_DARK',
    'ToHeatmapsTopDown_UDP',
    'TopDownEvalAffine',
    'AugmentationbyInformantionDropping',
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
                if visible[i][0] < 0 or visible[i][1] < 0 or visible[i][
                        0] >= hmsize or visible[i][1] >= hmsize:
                    continue
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


@register_keypointop
class RandomFlipHalfBodyTransform(object):
    """apply data augment to image and coords
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
                 rot_prob=0.6):
        super(RandomFlipHalfBodyTransform, self).__init__()
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

    def flip_joints(self, joints, joints_vis, width, matched_parts):
        joints[:, 0] = width - joints[:, 0] - 1
        for pair in matched_parts:
            joints[pair[0], :], joints[pair[1], :] = \
                joints[pair[1], :], joints[pair[0], :].copy()
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

        return joints * joints_vis, joints_vis

    def __call__(self, records):
        image = records['image']
        joints = records['joints']
        joints_vis = records['joints_vis']
        c = records['center']
        s = records['scale']
        r = 0
        if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and
                np.random.rand() < self.prob_half_body):
            c_half_body, s_half_body = self.halfbody_transform(joints,
                                                               joints_vis)
            if c_half_body is not None and s_half_body is not None:
                c, s = c_half_body, s_half_body
        sf = self.scale
        rf = self.rot
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn() * rf, -rf * 2,
                    rf * 2) if np.random.random() <= self.rot_prob else 0

        if self.flip and np.random.random() <= 0.5:
            image = image[:, ::-1, :]
            joints, joints_vis = self.flip_joints(
                joints, joints_vis, image.shape[1], self.flip_pairs)
            c[0] = image.shape[1] - c[0] - 1
        records['image'] = image
        records['joints'] = joints
        records['joints_vis'] = joints_vis
        records['center'] = c
        records['scale'] = s
        records['rotate'] = r

        return records


@register_keypointop
class AugmentationbyInformantionDropping(object):
    """AID: Augmentation by Informantion Dropping. Please refer 
        to https://arxiv.org/abs/2008.07139 
    
    Args:
        prob_cutout (float): The probability of the Cutout augmentation.
        offset_factor (float): Offset factor of cutout center.
        num_patch (int): Number of patches to be cutout.                       
        records(dict): the dict contained the image and coords
        
    Returns:
        records (dict): contain the image and coords after tranformed
    
    """

    def __init__(self,
                 trainsize,
                 prob_cutout=0.0,
                 offset_factor=0.2,
                 num_patch=1):
        self.prob_cutout = prob_cutout
        self.offset_factor = offset_factor
        self.num_patch = num_patch
        self.trainsize = trainsize

    def _cutout(self, img, joints, joints_vis):
        height, width, _ = img.shape
        img = img.reshape((height * width, -1))
        feat_x_int = np.arange(0, width)
        feat_y_int = np.arange(0, height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.reshape((-1, ))
        feat_y_int = feat_y_int.reshape((-1, ))
        for _ in range(self.num_patch):
            vis_idx, _ = np.where(joints_vis > 0)
            occlusion_joint_id = np.random.choice(vis_idx)
            center = joints[occlusion_joint_id, 0:2]
            offset = np.random.randn(2) * self.trainsize[0] * self.offset_factor
            center = center + offset
            radius = np.random.uniform(0.1, 0.2) * self.trainsize[0]
            x_offset = (center[0] - feat_x_int) / radius
            y_offset = (center[1] - feat_y_int) / radius
            dis = x_offset**2 + y_offset**2
            keep_pos = np.where((dis <= 1) & (dis >= 0))[0]
            img[keep_pos, :] = 0
        img = img.reshape((height, width, -1))
        return img

    def __call__(self, records):
        img = records['image']
        joints = records['joints']
        joints_vis = records['joints_vis']
        if np.random.rand() < self.prob_cutout:
            img = self._cutout(img, joints, joints_vis)
        records['image'] = img
        return records


@register_keypointop
class TopDownAffine(object):
    """apply affine transform to image and coords

    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self, trainsize, use_udp=False):
        self.trainsize = trainsize
        self.use_udp = use_udp

    def __call__(self, records):
        image = records['image']
        joints = records['joints']
        joints_vis = records['joints_vis']
        rot = records['rotate'] if "rotate" in records else 0
        if self.use_udp:
            trans = get_warp_matrix(
                rot, records['center'] * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0],
                records['scale'] * 200.0)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
            joints[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)
        else:
            trans = get_affine_transform(records['center'], records['scale'] *
                                         200, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
            for i in range(joints.shape[0]):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        records['image'] = image
        records['joints'] = joints

        return records


@register_keypointop
class TopDownEvalAffine(object):
    """apply affine transform to image and coords

    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self, trainsize, use_udp=False):
        self.trainsize = trainsize
        self.use_udp = use_udp

    def __call__(self, records):
        image = records['image']
        rot = 0
        imshape = records['im_shape'][::-1]
        center = imshape / 2.
        scale = imshape

        if self.use_udp:
            trans = get_warp_matrix(
                rot, center * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0], scale)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        else:
            trans = get_affine_transform(center, scale, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        records['image'] = image

        return records


@register_keypointop
class ToHeatmapsTopDown(object):
    """to generate the gaussin heatmaps of keypoint for heatmap loss

    Args:
        hmsize (list): [w, h] output heatmap's size
        sigma (float): the std of gaussin kernel genereted
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the heatmaps used to heatmaploss

    """

    def __init__(self, hmsize, sigma):
        super(ToHeatmapsTopDown, self).__init__()
        self.hmsize = np.array(hmsize)
        self.sigma = sigma

    def __call__(self, records):
        """refer to
            https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
            Copyright (c) Microsoft, under the MIT License.
        """
        joints = records['joints']
        joints_vis = records['joints_vis']
        num_joints = joints.shape[0]
        image_size = np.array(
            [records['image'].shape[1], records['image'].shape[0]])
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        target = np.zeros(
            (num_joints, self.hmsize[1], self.hmsize[0]), dtype=np.float32)
        tmp_size = self.sigma * 3
        feat_stride = image_size / self.hmsize
        for joint_id in range(num_joints):
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.hmsize[0] or ul[1] >= self.hmsize[1] or br[
                    0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue
            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.hmsize[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.hmsize[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.hmsize[0])
            img_y = max(0, ul[1]), min(br[1], self.hmsize[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[
                    0]:g_y[1], g_x[0]:g_x[1]]
        records['target'] = target
        records['target_weight'] = target_weight
        del records['joints'], records['joints_vis']

        return records


@register_keypointop
class ToHeatmapsTopDown_DARK(object):
    """to generate the gaussin heatmaps of keypoint for heatmap loss

    Args:
        hmsize (list): [w, h] output heatmap's size
        sigma (float): the std of gaussin kernel genereted
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the heatmaps used to heatmaploss

    """

    def __init__(self, hmsize, sigma):
        super(ToHeatmapsTopDown_DARK, self).__init__()
        self.hmsize = np.array(hmsize)
        self.sigma = sigma

    def __call__(self, records):
        joints = records['joints']
        joints_vis = records['joints_vis']
        num_joints = joints.shape[0]
        image_size = np.array(
            [records['image'].shape[1], records['image'].shape[0]])
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        target = np.zeros(
            (num_joints, self.hmsize[1], self.hmsize[0]), dtype=np.float32)
        tmp_size = self.sigma * 3
        feat_stride = image_size / self.hmsize
        for joint_id in range(num_joints):
            mu_x = joints[joint_id][0] / feat_stride[0]
            mu_y = joints[joint_id][1] / feat_stride[1]
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.hmsize[0] or ul[1] >= self.hmsize[1] or br[
                    0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            x = np.arange(0, self.hmsize[0], 1, np.float32)
            y = np.arange(0, self.hmsize[1], 1, np.float32)
            y = y[:, np.newaxis]

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id] = np.exp(-(
                    (x - mu_x)**2 + (y - mu_y)**2) / (2 * self.sigma**2))
        records['target'] = target
        records['target_weight'] = target_weight
        del records['joints'], records['joints_vis']

        return records


@register_keypointop
class ToHeatmapsTopDown_UDP(object):
    """This code is based on:
        https://github.com/HuangJunJie2017/UDP-Pose/blob/master/deep-high-resolution-net.pytorch/lib/dataset/JointsDataset.py
       
        to generate the gaussian heatmaps of keypoint for heatmap loss.
        ref: Huang et al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

    Args:
        hmsize (list): [w, h] output heatmap's size
        sigma (float): the std of gaussin kernel genereted
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the heatmaps used to heatmaploss
    """

    def __init__(self, hmsize, sigma):
        super(ToHeatmapsTopDown_UDP, self).__init__()
        self.hmsize = np.array(hmsize)
        self.sigma = sigma

    def __call__(self, records):
        joints = records['joints']
        joints_vis = records['joints_vis']
        num_joints = joints.shape[0]
        image_size = np.array(
            [records['image'].shape[1], records['image'].shape[0]])
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        target = np.zeros(
            (num_joints, self.hmsize[1], self.hmsize[0]), dtype=np.float32)
        tmp_size = self.sigma * 3
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
        feat_stride = (image_size - 1.0) / (self.hmsize - 1.0)
        for joint_id in range(num_joints):
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.hmsize[0] or ul[1] >= self.hmsize[1] or br[
                    0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            mu_x_ac = joints[joint_id][0] / feat_stride[0]
            mu_y_ac = joints[joint_id][1] / feat_stride[1]
            x0 = y0 = size // 2
            x0 += mu_x_ac - mu_x
            y0 += mu_y_ac - mu_y
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.hmsize[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.hmsize[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.hmsize[0])
            img_y = max(0, ul[1]), min(br[1], self.hmsize[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[
                    0]:g_y[1], g_x[0]:g_x[1]]
        records['target'] = target
        records['target_weight'] = target_weight
        del records['joints'], records['joints_vis']

        return records
