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
    'RandomAffine', 'KeyPointFlip', 'TagGenerate', 'ToHeatmaps',
    'NormalizePermute', 'EvalAffine', 'RandomFlipHalfBodyTransform',
    'TopDownRandomFlip', 'TopDownRandomShiftBboxCenter', 'TopDownGetRandomScaleRotation',
    'TopDownAffine', 'ToHeatmapsTopDown', 'ToHeatmapsTopDown_DARK',
    'ToHeatmapsTopDown_UDP', 'TopDownEvalAffine',
    'AugmentationbyInformantionDropping', 'SinglePoseAffine', 'NoiseJitter',
    'FlipPose', 'PETR_Resize'
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

    def __init__(self, flip_permutation, hmsize=None, flip_prob=0.5):
        super(KeyPointFlip, self).__init__()
        assert isinstance(flip_permutation, Sequence)
        self.flip_permutation = flip_permutation
        self.flip_prob = flip_prob
        self.hmsize = hmsize

    def _flipjoints(self, records, sizelst):
        '''
        records['gt_joints'] is Sequence in higherhrnet
        '''
        if not ('gt_joints' in records and len(records['gt_joints']) > 0):
            return records

        kpts_lst = records['gt_joints']
        if isinstance(kpts_lst, Sequence):
            for idx, hmsize in enumerate(sizelst):
                if kpts_lst[idx].ndim == 3:
                    kpts_lst[idx] = kpts_lst[idx][:, self.flip_permutation]
                else:
                    kpts_lst[idx] = kpts_lst[idx][self.flip_permutation]
                kpts_lst[idx][..., 0] = hmsize - kpts_lst[idx][..., 0]
        else:
            hmsize = sizelst[0]
            if kpts_lst.ndim == 3:
                kpts_lst = kpts_lst[:, self.flip_permutation]
            else:
                kpts_lst = kpts_lst[self.flip_permutation]
            kpts_lst[..., 0] = hmsize - kpts_lst[..., 0]

        records['gt_joints'] = kpts_lst
        return records

    def _flipmask(self, records, sizelst):
        if not 'mask' in records:
            return records

        mask_lst = records['mask']
        for idx, hmsize in enumerate(sizelst):
            if len(mask_lst) > idx:
                mask_lst[idx] = mask_lst[idx][:, ::-1]
        records['mask'] = mask_lst
        return records

    def _flipbbox(self, records, sizelst):
        if not 'gt_bbox' in records:
            return records

        bboxes = records['gt_bbox']
        hmsize = sizelst[0]
        bboxes[:, 0::2] = hmsize - bboxes[:, 0::2][:, ::-1]
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, hmsize)
        records['gt_bbox'] = bboxes
        return records

    def __call__(self, records):
        flip = np.random.random() < self.flip_prob
        if flip:
            image = records['image']
            image = image[:, ::-1]
            records['image'] = image
            if self.hmsize is None:
                sizelst = [image.shape[1]]
            else:
                sizelst = self.hmsize
            self._flipjoints(records, sizelst)
            self._flipmask(records, sizelst)
            self._flipbbox(records, sizelst)

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
        trainsize (list[2]): the standard length used to train, the 'scale_type' of [h,w] will be resize to trainsize for standard
        scale_type (str): the length of [h,w] to used for trainsize, chosed between 'short' and 'long'
        records(dict): the dict contained the image, mask and coords

    Returns:
        records(dict): contain the image, mask and coords after tranformed

    """

    def __init__(self,
                 max_degree=30,
                 scale=[0.75, 1.5],
                 max_shift=0.2,
                 hmsize=None,
                 trainsize=[512, 512],
                 scale_type='short',
                 boldervalue=[114, 114, 114]):
        super(RandomAffine, self).__init__()
        self.max_degree = max_degree
        self.min_scale = scale[0]
        self.max_scale = scale[1]
        self.max_shift = max_shift
        self.hmsize = hmsize
        self.trainsize = trainsize
        self.scale_type = scale_type
        self.boldervalue = boldervalue

    def _get_affine_matrix_old(self, center, scale, res, rot=0):
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

    def _get_affine_matrix(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        w, h = scale
        t = np.zeros((3, 3), dtype=np.float32)
        t[0, 0] = float(res[0]) / w
        t[1, 1] = float(res[1]) / h
        t[0, 2] = res[0] * (-float(center[0]) / w + .5)
        t[1, 2] = res[1] * (-float(center[1]) / h + .5)
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
            t_mat[0, 2] = -res[0] / 2
            t_mat[1, 2] = -res[1] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def _affine_joints_mask(self,
                            degree,
                            center,
                            roi_size,
                            dsize,
                            keypoints=None,
                            heatmap_mask=None,
                            gt_bbox=None):
        kpts = None
        mask = None
        bbox = None
        mask_affine_mat = self._get_affine_matrix(center, roi_size, dsize,
                                                  degree)[:2]
        if heatmap_mask is not None:
            mask = cv2.warpAffine(heatmap_mask, mask_affine_mat, dsize)
            mask = ((mask / 255) > 0.5).astype(np.float32)
        if keypoints is not None:
            kpts = copy.deepcopy(keypoints)
            kpts[..., 0:2] = warp_affine_joints(kpts[..., 0:2].copy(),
                                                mask_affine_mat)
            kpts[(kpts[..., 0]) > dsize[0], :] = 0
            kpts[(kpts[..., 1]) > dsize[1], :] = 0
            kpts[(kpts[..., 0]) < 0, :] = 0
            kpts[(kpts[..., 1]) < 0, :] = 0
        if gt_bbox is not None:
            temp_bbox = gt_bbox[:, [0, 3, 2, 1]]
            cat_bbox = np.concatenate((gt_bbox, temp_bbox), axis=-1)
            gt_bbox_warped = warp_affine_joints(cat_bbox, mask_affine_mat)
            bbox = np.zeros_like(gt_bbox)
            bbox[:, 0] = gt_bbox_warped[:, 0::2].min(1).clip(0, dsize[0])
            bbox[:, 2] = gt_bbox_warped[:, 0::2].max(1).clip(0, dsize[0])
            bbox[:, 1] = gt_bbox_warped[:, 1::2].min(1).clip(0, dsize[1])
            bbox[:, 3] = gt_bbox_warped[:, 1::2].max(1).clip(0, dsize[1])
        return kpts, mask, bbox

    def __call__(self, records):
        image = records['image']
        shape = np.array(image.shape[:2][::-1])
        keypoints = None
        heatmap_mask = None
        gt_bbox = None
        if 'gt_joints' in records:
            keypoints = records['gt_joints']

        if 'mask' in records:
            heatmap_mask = records['mask']
            heatmap_mask *= 255

        if 'gt_bbox' in records:
            gt_bbox = records['gt_bbox']

        degree = (np.random.random() * 2 - 1) * self.max_degree
        center = center = np.array((np.array(shape) / 2))

        aug_scale = np.random.random() * (self.max_scale - self.min_scale
                                          ) + self.min_scale
        if self.scale_type == 'long':
            scale = np.array([max(shape[0], shape[1]) / 1.0] * 2)
        elif self.scale_type == 'short':
            scale = np.array([min(shape[0], shape[1]) / 1.0] * 2)
        elif self.scale_type == 'wh':
            scale = shape
        else:
            raise ValueError('Unknown scale type: {}'.format(self.scale_type))
        roi_size = aug_scale * scale
        dx = int(0)
        dy = int(0)
        if self.max_shift > 0:

            dx = np.random.randint(-self.max_shift * roi_size[0],
                                   self.max_shift * roi_size[0])
            dy = np.random.randint(-self.max_shift * roi_size[0],
                                   self.max_shift * roi_size[1])

        center += np.array([dx, dy])
        input_size = 2 * center
        if self.trainsize != -1:
            dsize = self.trainsize
            imgshape = (dsize)
        else:
            dsize = scale
            imgshape = (shape.tolist())

        image_affine_mat = self._get_affine_matrix(center, roi_size, dsize,
                                                   degree)[:2]
        image = cv2.warpAffine(
            image,
            image_affine_mat,
            imgshape,
            flags=cv2.INTER_LINEAR,
            borderValue=self.boldervalue)

        if self.hmsize is None:
            kpts, mask, gt_bbox = self._affine_joints_mask(
                degree, center, roi_size, dsize, keypoints, heatmap_mask,
                gt_bbox)
            records['image'] = image
            if kpts is not None: records['gt_joints'] = kpts
            if mask is not None: records['mask'] = mask
            if gt_bbox is not None: records['gt_bbox'] = gt_bbox
            return records

        kpts_lst = []
        mask_lst = []
        for hmsize in self.hmsize:
            kpts, mask, gt_bbox = self._affine_joints_mask(
                degree, center, roi_size, [hmsize, hmsize], keypoints,
                heatmap_mask, gt_bbox)
            kpts_lst.append(kpts)
            mask_lst.append(mask)
        records['image'] = image

        if 'gt_joints' in records:
            records['gt_joints'] = kpts_lst
        if 'mask' in records:
            records['mask'] = mask_lst
        if 'gt_bbox' in records:
            records['gt_bbox'] = gt_bbox
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
        if 'gt_joints' in records:
            del records['gt_joints']
        records['image'] = image_resized
        records['scale_factor'] = self.size / min(h, w)
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
        kpts_lst = records['gt_joints']
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
        del records['gt_joints']
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
        kpts_lst = records['gt_joints']
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
        joints = records['gt_joints']
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
        records['gt_joints'] = joints
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
        joints = records['gt_joints']
        joints_vis = records['joints_vis']
        if np.random.rand() < self.prob_cutout:
            img = self._cutout(img, joints, joints_vis)
        records['image'] = img
        return records


@register_keypointop
class TopDownRandomFlip(object):
    """Data augmentation with random image flip.

    Args:
        flip_perm: (list[tuple]): Pairs of keypoints which are mirrored
                (for example, left ear and right ear).
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_perm=[], flip_prob=0.5):
        self.flip_perm = flip_perm
        self.flip_prob = flip_prob

    def flip_joints(self, joints_3d, joints_3d_visible, img_width, flip_pairs):
        assert len(joints_3d) == len(joints_3d_visible)
        assert img_width > 0

        joints_3d_flipped = joints_3d.copy()
        joints_3d_visible_flipped = joints_3d_visible.copy()

        # Swap left-right parts
        for left, right in flip_pairs:
            joints_3d_flipped[left, :] = joints_3d[right, :]
            joints_3d_flipped[right, :] = joints_3d[left, :]

            joints_3d_visible_flipped[left, :] = joints_3d_visible[right, :]
            joints_3d_visible_flipped[right, :] = joints_3d_visible[left, :]

        # Flip horizontally
        joints_3d_flipped[:, 0] = img_width - 1 - joints_3d_flipped[:, 0]
        joints_3d_flipped = joints_3d_flipped * (joints_3d_visible_flipped > 0)

        return joints_3d_flipped, joints_3d_visible_flipped

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        if np.random.rand() <= self.flip_prob:
            return results

        img = results['image']
        joints_3d = results['gt_joints']
        joints_3d_visible = results['joints_vis']
        center = results['center']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        if not isinstance(img, list):
            img = img[:, ::-1, :]
        else:
            img = [i[:, ::-1, :] for i in img]
        if not isinstance(img, list):
            joints_3d, joints_3d_visible = self.flip_joints(
                joints_3d, joints_3d_visible, img.shape[1],
                self.flip_perm)
            center[0] = img.shape[1] - center[0] - 1
        else:
            joints_3d, joints_3d_visible = self.flip_joints(
                joints_3d, joints_3d_visible, img[0].shape[1],
                self.flip_perm)
            center[0] = img[0].shape[1] - center[0] - 1

        results['image'] = img
        results['gt_joints'] = joints_3d
        results['joints_vis'] = joints_3d_visible
        results['center'] = center

        return results


@register_keypointop
class TopDownRandomShiftBboxCenter(object):
    """Random shift the bbox center.

    Args:
        shift_factor (float): The factor to control the shift range, which is
            scale*pixel_std*scale_factor. Default: 0.16
        shift_prob (float): Probability of applying random shift. Default: 0.3
    """

    def __init__(self, shift_factor=0.16, shift_prob=0.3):
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob

    def __call__(self, results):
        center = results['center']
        scale = results['scale']
        if np.random.rand() < self.shift_prob:
            center += np.random.uniform(
                -1, 1, 2) * self.shift_factor * scale * 200.0

        results['center'] = center
        return results

@register_keypointop
class TopDownGetRandomScaleRotation(object):
    """Data augmentation with random scaling & rotating.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['scale'] = s
        results['rotate'] = r

        return results


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
        joints = records['gt_joints']
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
        records['gt_joints'] = joints

        return records


@register_keypointop
class SinglePoseAffine(object):
    """apply affine transform to image and coords

    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self,
                 trainsize,
                 rotate=[1.0, 30],
                 scale=[1.0, 0.25],
                 use_udp=False):
        self.trainsize = trainsize
        self.use_udp = use_udp
        self.rot_prob = rotate[0]
        self.rot_range = rotate[1]
        self.scale_prob = scale[0]
        self.scale_ratio = scale[1]

    def __call__(self, records):
        image = records['image']
        if 'joints_2d' in records:
            joints = records['joints_2d'] if 'joints_2d' in records else None
            joints_vis = records[
                'joints_vis'] if 'joints_vis' in records else np.ones(
                    (len(joints), 1))
        rot = 0
        s = 1.
        if np.random.random() < self.rot_prob:
            rot = np.clip(np.random.randn() * self.rot_range,
                          -self.rot_range * 2, self.rot_range * 2)
        if np.random.random() < self.scale_prob:
            s = np.clip(np.random.randn() * self.scale_ratio + 1,
                        1 - self.scale_ratio, 1 + self.scale_ratio)

        if self.use_udp:
            trans = get_warp_matrix(
                rot,
                np.array(records['bbox_center']) * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0],
                records['bbox_scale'] * 200.0 * s)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
            if 'joints_2d' in records:
                joints[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(),
                                                    trans)
        else:
            trans = get_affine_transform(
                np.array(records['bbox_center']),
                records['bbox_scale'] * s * 200, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
            if 'joints_2d' in records:
                for i in range(len(joints)):
                    if joints_vis[i, 0] > 0.0:
                        joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        if 'joints_3d' in records:
            pose3d = records['joints_3d']
            if not rot == 0:
                trans_3djoints = np.eye(3)
                rot_rad = -rot * np.pi / 180
                sn, cs = np.sin(rot_rad), np.cos(rot_rad)
                trans_3djoints[0, :2] = [cs, -sn]
                trans_3djoints[1, :2] = [sn, cs]
                pose3d[:, :3] = np.einsum('ij,kj->ki', trans_3djoints,
                                          pose3d[:, :3])
                records['joints_3d'] = pose3d

        records['image'] = image
        if 'joints_2d' in records:
            records['joints_2d'] = joints

        return records


@register_keypointop
class NoiseJitter(object):
    """apply NoiseJitter to image

    Args:
        noise_factor (float): the noise factor ratio used to generate the jitter

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self, noise_factor=0.4):
        self.noise_factor = noise_factor

    def __call__(self, records):
        self.pn = np.random.uniform(1 - self.noise_factor,
                                    1 + self.noise_factor, 3)
        rgb_img = records['image']
        rgb_img[:, :, 0] = np.minimum(
            255.0, np.maximum(0.0, rgb_img[:, :, 0] * self.pn[0]))
        rgb_img[:, :, 1] = np.minimum(
            255.0, np.maximum(0.0, rgb_img[:, :, 1] * self.pn[1]))
        rgb_img[:, :, 2] = np.minimum(
            255.0, np.maximum(0.0, rgb_img[:, :, 2] * self.pn[2]))
        records['image'] = rgb_img
        return records


@register_keypointop
class FlipPose(object):
    """random apply flip to image

    Args:
        noise_factor (float): the noise factor ratio used to generate the jitter

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self, flip_prob=0.5, img_res=224, num_joints=14):
        self.flip_pob = flip_prob
        self.img_res = img_res
        if num_joints == 24:
            self.perm = [
                5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17,
                18, 19, 21, 20, 23, 22
            ]
        elif num_joints == 14:
            self.perm = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
        else:
            print("error num_joints in flip :{}".format(num_joints))

    def __call__(self, records):

        if np.random.random() < self.flip_pob:
            img = records['image']
            img = np.fliplr(img)

            if 'joints_2d' in records:
                joints_2d = records['joints_2d']
                joints_2d = joints_2d[self.perm]
                joints_2d[:, 0] = self.img_res - joints_2d[:, 0]
                records['joints_2d'] = joints_2d

            if 'joints_3d' in records:
                joints_3d = records['joints_3d']
                joints_3d = joints_3d[self.perm]
                joints_3d[:, 0] = -joints_3d[:, 0]
                records['joints_3d'] = joints_3d

            records['image'] = img
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
        joints = records['gt_joints']
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
        del records['gt_joints'], records['joints_vis']

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
        joints = records['gt_joints']
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
        del records['gt_joints'], records['joints_vis']

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
        joints = records['gt_joints']
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
        del records['gt_joints'], records['joints_vis']

        return records


from typing import Optional, Tuple, Union, List
import numbers


def _scale_size(
        size: Tuple[int, int],
        scale: Union[float, int, tuple], ) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool=False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, list):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(img: np.ndarray,
              scale: Union[float, Tuple[int, int]],
              return_scale: bool=False,
              interpolation: str='bilinear',
              backend: Optional[str]=None) -> Union[np.ndarray, Tuple[
                  np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imresize(
        img: np.ndarray,
        size: Tuple[int, int],
        return_scale: bool=False,
        interpolation: str='bilinear',
        out: Optional[np.ndarray]=None,
        backend: Optional[str]=None,
        interp=cv2.INTER_LINEAR, ) -> Union[Tuple[np.ndarray, float, float],
                                            np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = imread_backend
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(img, size, dst=out, interpolation=interp)
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


class PETR_Resize:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',
                 override=False,
                 keypoint_clip_border=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert isinstance(self.img_scale, list)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border
        self.keypoint_clip_border = keypoint_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert isinstance(img_scales, list)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert isinstance(img_scales, list) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (list): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, list) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0],
                                                        self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in ['image'] if 'image' in results else []:
            if self.keep_ratio:
                img, scale_factor = imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)

            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            results['im_shape'] = np.array(img.shape)
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio
            # img_pad = self.impad(img, shape=results['scale'])
            results[key] = img

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in ['gt_bbox'] if 'gt_bbox' in results else []:
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['im_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in ['mask'] if 'mask' in results else []:
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['im_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in ['seg'] if 'seg' in results else []:
            if self.keep_ratio:
                gt_seg = imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results[key] = gt_seg

    def _resize_keypoints(self, results):
        """Resize keypoints with ``results['scale_factor']``."""
        for key in ['gt_joints'] if 'gt_joints' in results else []:
            keypoints = results[key].copy()
            keypoints[..., 0] = keypoints[..., 0] * results['scale_factor'][0]
            keypoints[..., 1] = keypoints[..., 1] * results['scale_factor'][1]
            if self.keypoint_clip_border:
                img_shape = results['im_shape']
                keypoints[..., 0] = np.clip(keypoints[..., 0], 0, img_shape[1])
                keypoints[..., 1] = np.clip(keypoints[..., 1], 0, img_shape[0])
            results[key] = keypoints

    def _resize_areas(self, results):
        """Resize mask areas with ``results['scale_factor']``."""
        for key in ['gt_areas'] if 'gt_areas' in results else []:
            areas = results[key].copy()
            areas = areas * results['scale_factor'][0] * results[
                'scale_factor'][1]
            results[key] = areas

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'im_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['image'].shape[:2]
                scale_factor = results['scale_factor'][0]
                # assert isinstance(scale_factor, float)
                results['scale'] = [int(x * scale_factor)
                                    for x in img_shape][::-1]
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        self._resize_areas(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        repr_str += f'keypoint_clip_border={self.keypoint_clip_border})'
        return repr_str
