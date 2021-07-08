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

# function:
#    operators to process sample,
#    eg: decode/resize/crop image

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number

import uuid
import logging
import random
import math
import numpy as np
import os
import six

import cv2
from PIL import Image, ImageEnhance, ImageDraw, ImageOps

from ppdet.core.workspace import serializable
from ppdet.modeling.ops import AnchorGrid

from .op_helper import (satisfy_sample_constraint, filter_and_process,
                        generate_sample_bbox, clip_bbox, data_anchor_sampling,
                        satisfy_sample_constraint_coverage, crop_image_sampling,
                        generate_sample_bbox_square, bbox_area_sampling,
                        is_poly, gaussian_radius, draw_gaussian)

logger = logging.getLogger(__name__)

registered_ops = []


def register_op(cls):
    registered_ops.append(cls.__name__)
    if not hasattr(BaseOperator, cls.__name__):
        setattr(BaseOperator, cls.__name__, cls)
    else:
        raise KeyError("The {} class has been registered.".format(cls.__name__))
    return serializable(cls)


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


@register_op
class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, with_mixup=False, with_cutmix=False):
        """ Transform the image data to numpy format.
        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
            with_cutmix (bool): whether or not to cutmix image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        self.with_cutmix = with_cutmix
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode

        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warning(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warning(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)

        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context)

        # decode cutmix image
        if self.with_cutmix and 'cutmix' in sample:
            self.__call__(sample['cutmix'], context)

        # decode semantic label 
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            sem_file = sample['semantic']
            sem = cv2.imread(sem_file, cv2.IMREAD_GRAYSCALE)
            sample['semantic'] = sem.astype('int32')

        return sample


@register_op
class MultiscaleTestResize(BaseOperator):
    def __init__(self,
                 origin_target_size=800,
                 origin_max_size=1333,
                 target_size=[],
                 max_size=2000,
                 interp=cv2.INTER_LINEAR,
                 use_flip=True):
        """
        Rescale image to the each size in target size, and capped at max_size.
        Args:
            origin_target_size(int): original target size of image's short side.
            origin_max_size(int): original max size of image.
            target_size (list): A list of target sizes of image's short side.
            max_size (int): the max size of image.
            interp (int): the interpolation method.
            use_flip (bool): whether use flip augmentation.
        """
        super(MultiscaleTestResize, self).__init__()
        self.origin_target_size = int(origin_target_size)
        self.origin_max_size = int(origin_max_size)
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_flip = use_flip

        if not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid. Must be List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.origin_target_size, int) and isinstance(
                self.origin_max_size, int) and isinstance(self.max_size, int)
                and isinstance(self.interp, int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ Resize the image numpy for multi-scale test.
        """
        origin_ims = {}
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        base_name_list = ['image']
        origin_ims['image'] = im
        if self.use_flip:
            sample['image_flip'] = im[:, ::-1, :]
            base_name_list.append('image_flip')
            origin_ims['image_flip'] = sample['image_flip']

        for base_name in base_name_list:
            im_scale = float(self.origin_target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.origin_max_size:
                im_scale = float(self.origin_max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = np.round(im_scale_x * float(im_shape[1]))
            resize_h = np.round(im_scale_y * float(im_shape[0]))
            im_resize = cv2.resize(
                origin_ims[base_name],
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)

            sample[base_name] = im_resize
            info_name = 'im_info' if base_name == 'image' else 'im_info_image_flip'
            sample[base_name] = im_resize
            sample[info_name] = np.array(
                [resize_h, resize_w, im_scale], dtype=np.float32)
            for i, size in enumerate(self.target_size):
                im_scale = float(size) / float(im_size_min)
                if np.round(im_scale * im_size_max) > self.max_size:
                    im_scale = float(self.max_size) / float(im_size_max)
                im_scale_x = im_scale
                im_scale_y = im_scale
                resize_w = np.round(im_scale_x * float(im_shape[1]))
                resize_h = np.round(im_scale_y * float(im_shape[0]))
                im_resize = cv2.resize(
                    origin_ims[base_name],
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=self.interp)

                im_info = [resize_h, resize_w, im_scale]
                # hard-code here, must be consistent with
                # ppdet/modeling/architectures/input_helper.py
                name = base_name + '_scale_' + str(i)
                info_name = 'im_info_' + name
                sample[name] = im_resize
                sample[info_name] = np.array(
                    [resize_h, resize_w, im_scale], dtype=np.float32)
        return sample


@register_op
class ResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True,
                 resize_box=False):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
            resize_box (bool): whether resize ground truth bbox annotations.
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.resize_box = resize_box
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int) and isinstance(self.interp,
                                                              int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
            if 'im_info' in sample and sample['im_info'][2] != 1.:
                sample['im_info'] = np.append(
                    list(sample['im_info']), im_info).astype(np.float32)
            else:
                sample['im_info'] = np.array(im_info).astype(np.float32)
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        sample['image'] = im
        sample['scale_factor'] = [im_scale_x, im_scale_y] * 2
        if 'gt_bbox' in sample and self.resize_box and len(sample[
                'gt_bbox']) > 0:
            bboxes = sample['gt_bbox'] * sample['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, resize_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, resize_h - 1)
            sample['gt_bbox'] = bboxes
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample


@register_op
class RandomFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False, is_mask_flip=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool) and
                isinstance(self.is_mask_flip, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def flip_segms(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def flip_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                if self.is_normalized:
                    gt_keypoint[:, i] = 1 - old_x
                else:
                    gt_keypoint[:, i] = width - old_x - 1
        return gt_keypoint

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                if gt_bbox.shape[0] == 0:
                    return sample
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (
                        gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(
                        self)
                    raise BboxError(m)
                sample['gt_bbox'] = gt_bbox
                if self.is_mask_flip and len(sample['gt_poly']) != 0:
                    sample['gt_poly'] = self.flip_segms(sample['gt_poly'],
                                                        height, width)
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = self.flip_keypoint(
                        sample['gt_keypoint'], width)

                if 'semantic' in sample.keys() and sample[
                        'semantic'] is not None:
                    sample['semantic'] = sample['semantic'][:, ::-1]

                if 'gt_segm' in sample.keys() and sample['gt_segm'] is not None:
                    sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

                sample['flipped'] = True
                sample['image'] = im
        sample = samples if batch_input else samples[0]
        return sample


@register_op
class RandomErasingImage(BaseOperator):
    def __init__(self, prob=0.5, sl=0.02, sh=0.4, r1=0.3):
        """
        Random Erasing Data Augmentation, see https://arxiv.org/abs/1708.04896
        Args:
            prob (float): probability to carry out random erasing
            sl (float): lower limit of the erasing area ratio
            sh (float): upper limit of the erasing area ratio
            r1 (float): aspect ratio of the erasing region
        """
        super(RandomErasingImage, self).__init__()
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))

            for idx in range(gt_bbox.shape[0]):
                if self.prob <= np.random.rand():
                    continue

                x1, y1, x2, y2 = gt_bbox[idx, :]
                w_bbox = x2 - x1 + 1
                h_bbox = y2 - y1 + 1
                area = w_bbox * h_bbox

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < w_bbox and h < h_bbox:
                    off_y1 = random.randint(0, int(h_bbox - h))
                    off_x1 = random.randint(0, int(w_bbox - w))
                    im[int(y1 + off_y1):int(y1 + off_y1 + h), int(x1 + off_x1):
                       int(x1 + off_x1 + w), :] = 0
            sample['image'] = im

        sample = samples if batch_input else samples[0]
        return sample


@register_op
class GridMaskOp(BaseOperator):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 prob=0.7,
                 upper_iter=360000):
        """
        GridMask Data Augmentation, see https://arxiv.org/abs/2001.04086
        Args:
            use_h (bool): whether to mask vertically
            use_w (boo;): whether to mask horizontally
            rotate (float): angle for the mask to rotate
            offset (float): mask offset
            ratio (float): mask ratio
            mode (int): gridmask mode
            prob (float): max probability to carry out gridmask
            upper_iter (int): suggested to be equal to global max_iter
        """
        super(GridMaskOp, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.upper_iter = upper_iter

        from .gridmask_utils import GridMask
        self.gridmask_op = GridMask(
            use_h,
            use_w,
            rotate=rotate,
            offset=offset,
            ratio=ratio,
            mode=mode,
            prob=prob,
            upper_iter=upper_iter)

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            sample['image'] = self.gridmask_op(sample['image'],
                                               sample['curr_iter'])
        if not batch_input:
            samples = samples[0]
        return samples


@register_op
class AutoAugmentImage(BaseOperator):
    def __init__(self, is_normalized=False, autoaug_type="v1"):
        """
        Args:
            is_normalized (bool): whether the bbox scale to [0,1]
            autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
        """
        super(AutoAugmentImage, self).__init__()
        self.is_normalized = is_normalized
        self.autoaug_type = autoaug_type
        if not isinstance(self.is_normalized, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """
        Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            if len(gt_bbox) == 0:
                continue

            # gt_boxes : [x1, y1, x2, y2]
            # norm_gt_boxes: [y1, x1, y2, x2]
            height, width, _ = im.shape
            norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
            if not self.is_normalized:
                norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
                norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
                norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
                norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)
            else:
                norm_gt_bbox[:, 0] = gt_bbox[:, 1]
                norm_gt_bbox[:, 1] = gt_bbox[:, 0]
                norm_gt_bbox[:, 2] = gt_bbox[:, 3]
                norm_gt_bbox[:, 3] = gt_bbox[:, 2]

            from .autoaugment_utils import distort_image_with_autoaugment
            im, norm_gt_bbox = distort_image_with_autoaugment(im, norm_gt_bbox,
                                                              self.autoaug_type)
            if not self.is_normalized:
                gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
                gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
                gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
                gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)
            else:
                gt_bbox[:, 0] = norm_gt_bbox[:, 1]
                gt_bbox[:, 1] = norm_gt_bbox[:, 0]
                gt_bbox[:, 2] = norm_gt_bbox[:, 3]
                gt_bbox[:, 3] = norm_gt_bbox[:, 2]

            sample['gt_bbox'] = gt_bbox
            sample['image'] = im

        sample = samples if batch_input else samples[0]
        return sample


@register_op
class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


@register_op
class RandomDistort(BaseOperator):
    def __init__(self,
                 brightness_lower=0.5,
                 brightness_upper=1.5,
                 contrast_lower=0.5,
                 contrast_upper=1.5,
                 saturation_lower=0.5,
                 saturation_upper=1.5,
                 hue_lower=-18,
                 hue_upper=18,
                 brightness_prob=0.5,
                 contrast_prob=0.5,
                 saturation_prob=0.5,
                 hue_prob=0.5,
                 count=4,
                 is_order=False):
        """
        Args:
            brightness_lower/ brightness_upper (float): the brightness
                between brightness_lower and brightness_upper
            contrast_lower/ contrast_upper (float): the contrast between
                contrast_lower and contrast_lower
            saturation_lower/ saturation_upper (float): the saturation
                between saturation_lower and saturation_upper
            hue_lower/ hue_upper (float): the hue between
                hue_lower and hue_upper
            brightness_prob (float): the probability of changing brightness
            contrast_prob (float): the probability of changing contrast
            saturation_prob (float): the probability of changing saturation
            hue_prob (float): the probability of changing hue
            count (int): the kinds of doing distrot
            is_order (bool): whether determine the order of distortion
        """
        super(RandomDistort, self).__init__()
        self.brightness_lower = brightness_lower
        self.brightness_upper = brightness_upper
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.saturation_lower = saturation_lower
        self.saturation_upper = saturation_upper
        self.hue_lower = hue_lower
        self.hue_upper = hue_upper
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.saturation_prob = saturation_prob
        self.hue_prob = hue_prob
        self.count = count
        self.is_order = is_order

    def random_brightness(self, img):
        brightness_delta = np.random.uniform(self.brightness_lower,
                                             self.brightness_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.brightness_prob:
            img = ImageEnhance.Brightness(img).enhance(brightness_delta)
        return img

    def random_contrast(self, img):
        contrast_delta = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.contrast_prob:
            img = ImageEnhance.Contrast(img).enhance(contrast_delta)
        return img

    def random_saturation(self, img):
        saturation_delta = np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.saturation_prob:
            img = ImageEnhance.Color(img).enhance(saturation_delta)
        return img

    def random_hue(self, img):
        hue_delta = np.random.uniform(self.hue_lower, self.hue_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.hue_prob:
            img = np.array(img.convert('HSV'))
            img[:, :, 0] = img[:, :, 0] + hue_delta
            img = Image.fromarray(img, mode='HSV').convert('RGB')
        return img

    def __call__(self, sample, context):
        """random distort the image"""
        ops = [
            self.random_brightness, self.random_contrast,
            self.random_saturation, self.random_hue
        ]
        if self.is_order:
            prob = np.random.uniform(0, 1)
            if prob < 0.5:
                ops = [
                    self.random_brightness,
                    self.random_saturation,
                    self.random_hue,
                    self.random_contrast,
                ]
        else:
            ops = random.sample(ops, self.count)
        assert 'image' in sample, "image data not found"
        im = sample['image']
        im = Image.fromarray(im)
        for id in range(self.count):
            im = ops[id](im)
        im = np.asarray(im)
        sample['image'] = im
        return sample


@register_op
class ExpandImage(BaseOperator):
    def __init__(self, max_ratio, prob, mean=[127.5, 127.5, 127.5]):
        """
        Args:
            max_ratio (float): the ratio of expanding
            prob (float): the probability of expanding image
            mean (list): the pixel mean
        """
        super(ExpandImage, self).__init__()
        self.max_ratio = max_ratio
        self.mean = mean
        self.prob = prob

    def __call__(self, sample, context):
        """
        Expand the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Construct new images with new height and width.
            3. Fill the new image with the mean.
            4. Put original imge into new image.
            5. Rescale the bounding box.
            6. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """

        prob = np.random.uniform(0, 1)
        assert 'image' in sample, 'not found image data'
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_width = sample['w']
        im_height = sample['h']
        if prob < self.prob:
            if self.max_ratio - 1 >= 0.01:
                expand_ratio = np.random.uniform(1, self.max_ratio)
                height = int(im_height * expand_ratio)
                width = int(im_width * expand_ratio)
                h_off = math.floor(np.random.uniform(0, height - im_height))
                w_off = math.floor(np.random.uniform(0, width - im_width))
                expand_bbox = [
                    -w_off / im_width, -h_off / im_height,
                    (width - w_off) / im_width, (height - h_off) / im_height
                ]
                expand_im = np.ones((height, width, 3))
                expand_im = np.uint8(expand_im * np.squeeze(self.mean))
                expand_im = Image.fromarray(expand_im)
                im = Image.fromarray(im)
                expand_im.paste(im, (int(w_off), int(h_off)))
                expand_im = np.asarray(expand_im)
                if 'gt_keypoint' in sample.keys(
                ) and 'keypoint_ignore' in sample.keys():
                    keypoints = (sample['gt_keypoint'],
                                 sample['keypoint_ignore'])
                    gt_bbox, gt_class, _, gt_keypoints = filter_and_process(
                        expand_bbox, gt_bbox, gt_class, keypoints=keypoints)
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                else:
                    gt_bbox, gt_class, _ = filter_and_process(expand_bbox,
                                                              gt_bbox, gt_class)
                sample['image'] = expand_im
                sample['gt_bbox'] = gt_bbox
                sample['gt_class'] = gt_class
                sample['w'] = width
                sample['h'] = height

        return sample


@register_op
class CropImage(BaseOperator):
    def __init__(self, batch_sampler, satisfy_all=False, avoid_no_bbox=True):
        """
        Args:
            batch_sampler (list): Multiple sets of different
                                  parameters for cropping.
            satisfy_all (bool): whether all boxes must satisfy.
            e.g.[[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]
           [max sample, max trial, min scale, max scale,
            min aspect ratio, max aspect ratio,
            min overlap, max overlap]
            avoid_no_bbox (bool): whether to to avoid the
                                  situation where the box does not appear.
        """
        super(CropImage, self).__init__()
        self.batch_sampler = batch_sampler
        self.satisfy_all = satisfy_all
        self.avoid_no_bbox = avoid_no_bbox

    def __call__(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Crop the image according to a radom sample.
            3. Rescale the bounding box.
            4. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """
        assert 'image' in sample, "image data not found"
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_width = sample['w']
        im_height = sample['h']
        gt_score = None
        if 'gt_score' in sample:
            gt_score = sample['gt_score']
        sampled_bbox = []
        gt_bbox = gt_bbox.tolist()
        for sampler in self.batch_sampler:
            found = 0
            for i in range(sampler[1]):
                if found >= sampler[0]:
                    break
                sample_bbox = generate_sample_bbox(sampler)
                if satisfy_sample_constraint(sampler, sample_bbox, gt_bbox,
                                             self.satisfy_all):
                    sampled_bbox.append(sample_bbox)
                    found = found + 1
        im = np.array(im)
        while sampled_bbox:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            sample_bbox = sampled_bbox.pop(idx)
            sample_bbox = clip_bbox(sample_bbox)
            crop_bbox, crop_class, crop_score = \
                filter_and_process(sample_bbox, gt_bbox, gt_class, scores=gt_score)
            if self.avoid_no_bbox:
                if len(crop_bbox) < 1:
                    continue
            xmin = int(sample_bbox[0] * im_width)
            xmax = int(sample_bbox[2] * im_width)
            ymin = int(sample_bbox[1] * im_height)
            ymax = int(sample_bbox[3] * im_height)
            im = im[ymin:ymax, xmin:xmax]
            sample['image'] = im
            sample['gt_bbox'] = crop_bbox
            sample['gt_class'] = crop_class
            sample['gt_score'] = crop_score
            return sample
        return sample


@register_op
class CropImageWithDataAchorSampling(BaseOperator):
    def __init__(self,
                 batch_sampler,
                 anchor_sampler=None,
                 target_size=None,
                 das_anchor_scales=[16, 32, 64, 128],
                 sampling_prob=0.5,
                 min_size=8.,
                 avoid_no_bbox=True):
        """
        Args:
            anchor_sampler (list): anchor_sampling sets of different
                                  parameters for cropping.
            batch_sampler (list): Multiple sets of different
                                  parameters for cropping.
              e.g.[[1, 10, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.0]]
                  [[1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]
              [max sample, max trial, min scale, max scale,
               min aspect ratio, max aspect ratio,
               min overlap, max overlap, min coverage, max coverage]
            target_size (bool): target image size.
            das_anchor_scales (list[float]): a list of anchor scales in data
                anchor smapling.
            min_size (float): minimum size of sampled bbox.
            avoid_no_bbox (bool): whether to to avoid the
                                  situation where the box does not appear.
        """
        super(CropImageWithDataAchorSampling, self).__init__()
        self.anchor_sampler = anchor_sampler
        self.batch_sampler = batch_sampler
        self.target_size = target_size
        self.sampling_prob = sampling_prob
        self.min_size = min_size
        self.avoid_no_bbox = avoid_no_bbox
        self.das_anchor_scales = np.array(das_anchor_scales)

    def __call__(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Crop the image according to a radom sample.
            3. Rescale the bounding box.
            4. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """
        assert 'image' in sample, "image data not found"
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        image_width = sample['w']
        image_height = sample['h']
        gt_score = None
        if 'gt_score' in sample:
            gt_score = sample['gt_score']
        sampled_bbox = []
        gt_bbox = gt_bbox.tolist()

        prob = np.random.uniform(0., 1.)
        if prob > self.sampling_prob:  # anchor sampling
            assert self.anchor_sampler
            for sampler in self.anchor_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = data_anchor_sampling(
                        gt_bbox, image_width, image_height,
                        self.das_anchor_scales, self.target_size)
                    if sample_bbox == 0:
                        break
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,
                                                          gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)

                if 'gt_keypoint' in sample.keys():
                    keypoints = (sample['gt_keypoint'],
                                 sample['keypoint_ignore'])
                    crop_bbox, crop_class, crop_score, gt_keypoints = \
                        filter_and_process(sample_bbox, gt_bbox, gt_class,
                                scores=gt_score,
                                keypoints=keypoints)
                else:
                    crop_bbox, crop_class, crop_score = filter_and_process(
                        sample_bbox, gt_bbox, gt_class, scores=gt_score)
                crop_bbox, crop_class, crop_score = bbox_area_sampling(
                    crop_bbox, crop_class, crop_score, self.target_size,
                    self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                im = crop_image_sampling(im, sample_bbox, image_width,
                                         image_height, self.target_size)
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                sample['gt_score'] = crop_score
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                return sample
            return sample

        else:
            for sampler in self.batch_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = generate_sample_bbox_square(
                        sampler, image_width, image_height)
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,
                                                          gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)
                sample_bbox = clip_bbox(sample_bbox)

                if 'gt_keypoint' in sample.keys():
                    keypoints = (sample['gt_keypoint'],
                                 sample['keypoint_ignore'])
                    crop_bbox, crop_class, crop_score, gt_keypoints = \
                        filter_and_process(sample_bbox, gt_bbox, gt_class,
                                scores=gt_score,
                                keypoints=keypoints)
                else:
                    crop_bbox, crop_class, crop_score = filter_and_process(
                        sample_bbox, gt_bbox, gt_class, scores=gt_score)
                # sampling bbox according the bbox area
                crop_bbox, crop_class, crop_score = bbox_area_sampling(
                    crop_bbox, crop_class, crop_score, self.target_size,
                    self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                xmin = int(sample_bbox[0] * image_width)
                xmax = int(sample_bbox[2] * image_width)
                ymin = int(sample_bbox[1] * image_height)
                ymax = int(sample_bbox[3] * image_height)
                im = im[ymin:ymax, xmin:xmax]
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                sample['gt_score'] = crop_score
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                return sample
            return sample


@register_op
class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample, context):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']

            for i in range(gt_keypoint.shape[1]):
                if i % 2:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / height
                else:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / width
            sample['gt_keypoint'] = gt_keypoint

        return sample


@register_op
class Permute(BaseOperator):
    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool) and
                isinstance(self.channel_first, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert 'image' in sample, "image data not found"
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    if self.channel_first:
                        im = np.swapaxes(im, 1, 2)
                        im = np.swapaxes(im, 1, 0)
                    if self.to_bgr:
                        im = im[[2, 1, 0], :, :]
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


@register_op
class MixupImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            return sample['mixup']
        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        gt_bbox1 = sample['gt_bbox'].reshape((-1, 4))
        gt_bbox2 = sample['mixup']['gt_bbox'].reshape((-1, 4))
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)

        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)

        is_crowd1 = sample['is_crowd']
        is_crowd2 = sample['mixup']['is_crowd']
        is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)

        sample['image'] = im
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['is_crowd'] = is_crowd
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample.pop('mixup')
        return sample


@register_op
class CutmixImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ 
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(CutmixImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

        def _rand_bbox(self, img1, img2, factor):
            """ _rand_bbox """
            h = max(img1.shape[0], img2.shape[0])
            w = max(img1.shape[1], img2.shape[1])
            cut_rat = np.sqrt(1. - factor)

            cut_w = np.int32(w * cut_rat)
            cut_h = np.int32(h * cut_rat)

            # uniform
            cx = np.random.randint(w)
            cy = np.random.randint(h)

            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)

            img_1 = np.zeros((h, w, img1.shape[2]), 'float32')
            img_1[:img1.shape[0], :img1.shape[1], :] = \
                img1.astype('float32')
            img_2 = np.zeros((h, w, img2.shape[2]), 'float32')
            img_2[:img2.shape[0], :img2.shape[1], :] = \
                img2.astype('float32')
            img_1[bby1:bby2, bbx1:bbx2, :] = img2[bby1:bby2, bbx1:bbx2, :]
            return img_1

        def __call__(self, sample, context=None):
            if 'cutmix' not in sample:
                return sample
            factor = np.random.beta(self.alpha, self.beta)
            factor = max(0.0, min(1.0, factor))
            if factor >= 1.0:
                sample.pop('cutmix')
                return sample
            if factor <= 0.0:
                return sample['cutmix']
            img1 = sample['image']
            img2 = sample['cutmix']['image']
            img = self._rand_bbox(img1, img2, factor)
            gt_bbox1 = sample['gt_bbox']
            gt_bbox2 = sample['cutmix']['gt_bbox']
            gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
            gt_class1 = sample['gt_class']
            gt_class2 = sample['cutmix']['gt_class']
            gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
            gt_score1 = sample['gt_score']
            gt_score2 = sample['cutmix']['gt_score']
            gt_score = np.concatenate(
                (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
            sample['image'] = img
            sample['gt_bbox'] = gt_bbox
            sample['gt_score'] = gt_score
            sample['gt_class'] = gt_class
            sample['h'] = img.shape[0]
            sample['w'] = img.shape[1]
            sample.pop('cutmix')
            return sample


@register_op
class RandomInterpImage(BaseOperator):
    def __init__(self, target_size=0, max_size=0):
        """
        Random reisze image by multiply interpolate method.
        Args:
            target_size (int): the taregt size of image's short side
            max_size (int): the max size of image
        """
        super(RandomInterpImage, self).__init__()
        self.target_size = target_size
        self.max_size = max_size
        if not (isinstance(self.target_size, int) and
                isinstance(self.max_size, int)):
            raise TypeError('{}: input type is invalid.'.format(self))
        interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.resizers = []
        for interp in interps:
            self.resizers.append(ResizeImage(target_size, max_size, interp))

    def __call__(self, sample, context=None):
        """Resise the image numpy by random resizer."""
        resizer = random.choice(self.resizers)
        return resizer(sample, context)


@register_op
class Resize(BaseOperator):
    """Resize image and bbox.
    Args:
        target_dim (int or list): target size, can be a single number or a list
            (for random shape).
        interp (int or str): interpolation method, can be an integer or
            'random' (for randomized interpolation).
            default to `cv2.INTER_LINEAR`.
    """

    def __init__(self, target_dim=[], interp=cv2.INTER_LINEAR):
        super(Resize, self).__init__()
        self.target_dim = target_dim
        self.interp = interp  # 'random' for yolov3

    def __call__(self, sample, context=None):
        w = sample['w']
        h = sample['h']

        interp = self.interp
        if interp == 'random':
            interp = np.random.choice(range(5))

        if isinstance(self.target_dim, Sequence):
            dim = np.random.choice(self.target_dim)
        else:
            dim = self.target_dim
        resize_w = resize_h = dim
        scale_x = dim / w
        scale_y = dim / h
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
            sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0,
                                        dim - 1)
        sample['scale_factor'] = [scale_x, scale_y] * 2
        sample['h'] = resize_h
        sample['w'] = resize_w

        sample['image'] = cv2.resize(
            sample['image'], (resize_w, resize_h), interpolation=interp)
        return sample


@register_op
class ColorDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings.
            in [lower, upper, probability] format.
        saturation (list): saturation settings.
            in [lower, upper, probability] format.
        contrast (list): contrast settings.
            in [lower, upper, probability] format.
        brightness (list): brightness settings.
            in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        hsv_format (bool): whether to convert color from BGR to HSV
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 hsv_format=False,
                 random_channel=False):
        super(ColorDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.hsv_format = hsv_format
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 0] += random.uniform(low, high)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
            return img

        # XXX works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 1] *= delta
            return img
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness,
                self.apply_contrast,
                self.apply_saturation,
                self.apply_hue,
            ]
            distortions = np.random.permutation(functions)
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)

        if np.random.randint(0, 2):
            img = self.apply_contrast(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


@register_op
class CornerRandColor(ColorDistort):
    """Random color for CornerNet series models.
    Args:
        saturation (float): saturation settings.
        contrast (float): contrast settings.
        brightness (float): brightness settings.
        is_scale (bool): whether to scale the input image.
    """

    def __init__(self,
                 saturation=0.4,
                 contrast=0.4,
                 brightness=0.4,
                 is_scale=True):
        super(CornerRandColor, self).__init__(
            saturation=saturation, contrast=contrast, brightness=brightness)
        self.is_scale = is_scale

    def apply_saturation(self, img, img_gray):
        alpha = 1. + np.random.uniform(
            low=-self.saturation, high=self.saturation)
        self._blend(alpha, img, img_gray[:, :, None])
        return img

    def apply_contrast(self, img, img_gray):
        alpha = 1. + np.random.uniform(low=-self.contrast, high=self.contrast)
        img_mean = img_gray.mean()
        self._blend(alpha, img, img_mean)
        return img

    def apply_brightness(self, img, img_gray):
        alpha = 1 + np.random.uniform(
            low=-self.brightness, high=self.brightness)
        img *= alpha
        return img

    def _blend(self, alpha, img, img_mean):
        img *= alpha
        img_mean *= (1 - alpha)
        img += img_mean

    def __call__(self, sample, context=None):
        img = sample['image']
        if self.is_scale:
            img = img.astype(np.float32, copy=False)
            img /= 255.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        functions = [
            self.apply_brightness,
            self.apply_contrast,
            self.apply_saturation,
        ]
        distortions = np.random.permutation(functions)
        for func in distortions:
            img = func(img, img_gray)
        sample['image'] = img
        return sample


@register_op
class NormalizePermute(BaseOperator):
    """Normalize and permute channel order.
    Args:
        mean (list): mean values in RGB order.
        std (list): std values in RGB order.
    """

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.120, 57.375]):
        super(NormalizePermute, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample, context=None):
        img = sample['image']
        img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        invstd = 1. / std
        for v, m, s in zip(img, mean, invstd):
            v.__isub__(m).__imul__(s)
        sample['image'] = img
        return sample


@register_op
class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
        is_mask_expand(bool): whether expand the segmentation.
    """

    def __init__(self,
                 ratio=4.,
                 prob=0.5,
                 fill_value=(127.5, ) * 3,
                 is_mask_expand=False):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value, ) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value
        self.is_mask_expand = is_mask_expand

    def expand_segms(self, segms, x, y, height, width, ratio):
        def _expand_poly(poly, x, y):
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, ratio):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((int(height * ratio), int(width * ratio)),
                                    0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [_expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                expanded_segms.append(
                    _expand_rle(segm, x, y, height, width, ratio))
        return expanded_segms

    def __call__(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        img = sample['image']
        height = int(sample['h'])
        width = int(sample['w'])

        expand_ratio = np.random.uniform(1., self.ratio)
        h = int(height * expand_ratio)
        w = int(width * expand_ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas *= np.array(self.fill_value, dtype=np.uint8)
        canvas[y:y + height, x:x + width, :] = img.astype(np.uint8)

        sample['h'] = h
        sample['w'] = w
        sample['image'] = canvas
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] += np.array([x, y] * 2, dtype=np.float32)
        if self.is_mask_expand and 'gt_poly' in sample and len(sample[
                'gt_poly']) > 0:
            sample['gt_poly'] = self.expand_segms(sample['gt_poly'], x, y,
                                                  height, width, expand_ratio)
        return sample


@register_op
class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def crop_segms(self, segms, valid_ids, crop, height, width):
        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)

            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly) // 2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(
                                np.array(part.exterior.coords[:-1]).reshape(1,
                                                                            -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(
                            np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if is_poly(segm):
                import copy
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                logging.getLogger("shapely").setLevel(logging.WARNING)
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def __call__(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h = sample['h']
        w = sample['w']
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(
                        max(min_ar, scale**2), min(max_ar, scale**-2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                if self.is_mask_crop and 'gt_poly' in sample and len(sample[
                        'gt_poly']) > 0:
                    crop_polys = self.crop_segms(
                        sample['gt_poly'],
                        valid_ids,
                        np.array(
                            crop_box, dtype=np.int64),
                        h,
                        w)
                    if [] in crop_polys:
                        delete_id = list()
                        valid_polys = list()
                        for id, crop_poly in enumerate(crop_polys):
                            if crop_poly == []:
                                delete_id.append(id)
                            else:
                                valid_polys.append(crop_poly)
                        valid_ids = np.delete(valid_ids, delete_id)
                        if len(valid_polys) == 0:
                            return sample
                        sample['gt_poly'] = valid_polys
                    else:
                        sample['gt_poly'] = crop_polys

                if 'gt_segm' in sample:
                    sample['gt_segm'] = self._crop_segm(sample['gt_segm'],
                                                        crop_box)
                    sample['gt_segm'] = np.take(
                        sample['gt_segm'], valid_ids, axis=0)
                sample['image'] = self._crop_image(sample['image'], crop_box)
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                sample['w'] = crop_box[2] - crop_box[0]
                sample['h'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

    def _crop_segm(self, segm, crop):
        x1, y1, x2, y2 = crop
        return segm[:, y1:y2, x1:x2]


@register_op
class PadBox(BaseOperator):
    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        self.num_max_boxes = num_max_boxes
        super(PadBox, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes
        fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in fields:
            pad_class = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in fields:
            pad_score = np.zeros((num_max), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'is_difficult' in fields:
            pad_diff = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        return sample


@register_op
class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


@register_op
class Lighting(BaseOperator):
    """
    Lighting the image by eigenvalues and eigenvectors
    Args:
        eigval (list): eigenvalues
        eigvec (list): eigenvectors
        alphastd (float): random weight of lighting, 0.1 by default
    """

    def __init__(self, eigval, eigvec, alphastd=0.1):
        super(Lighting, self).__init__()
        self.alphastd = alphastd
        self.eigval = np.array(eigval).astype('float32')
        self.eigvec = np.array(eigvec).astype('float32')

    def __call__(self, sample, context=None):
        alpha = np.random.normal(scale=self.alphastd, size=(3, ))
        sample['image'] += np.dot(self.eigvec, self.eigval * alpha)
        return sample


@register_op
class CornerTarget(BaseOperator):
    """
    Generate targets for CornerNet by ground truth data. 
    Args:
        output_size (int): the size of output heatmaps.
        num_classes (int): num of classes.
        gaussian_bump (bool): whether to apply gaussian bump on gt targets.
            True by default.
        gaussian_rad (int): radius of gaussian bump. If it is set to -1, the 
            radius will be calculated by iou. -1 by default.
        gaussian_iou (float): the threshold iou of predicted bbox to gt bbox. 
            If the iou is larger than threshold, the predicted bboox seems as
            positive sample. 0.3 by default
        max_tag_len (int): max num of gt box per image.
    """

    def __init__(self,
                 output_size,
                 num_classes,
                 gaussian_bump=True,
                 gaussian_rad=-1,
                 gaussian_iou=0.3,
                 max_tag_len=128):
        super(CornerTarget, self).__init__()
        self.num_classes = num_classes
        self.output_size = output_size
        self.gaussian_bump = gaussian_bump
        self.gaussian_rad = gaussian_rad
        self.gaussian_iou = gaussian_iou
        self.max_tag_len = max_tag_len

    def __call__(self, sample, context=None):
        tl_heatmaps = np.zeros(
            (self.num_classes, self.output_size[0], self.output_size[1]),
            dtype=np.float32)
        br_heatmaps = np.zeros(
            (self.num_classes, self.output_size[0], self.output_size[1]),
            dtype=np.float32)

        tl_regrs = np.zeros((self.max_tag_len, 2), dtype=np.float32)
        br_regrs = np.zeros((self.max_tag_len, 2), dtype=np.float32)
        tl_tags = np.zeros((self.max_tag_len), dtype=np.int64)
        br_tags = np.zeros((self.max_tag_len), dtype=np.int64)
        tag_masks = np.zeros((self.max_tag_len), dtype=np.uint8)
        tag_lens = np.zeros((), dtype=np.int32)
        tag_nums = np.zeros((1), dtype=np.int32)

        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        keep_inds  = ((gt_bbox[:, 2] - gt_bbox[:, 0]) > 0) & \
                ((gt_bbox[:, 3] - gt_bbox[:, 1]) > 0)
        gt_bbox = gt_bbox[keep_inds]
        gt_class = gt_class[keep_inds]
        sample['gt_bbox'] = gt_bbox
        sample['gt_class'] = gt_class
        width_ratio = self.output_size[1] / sample['w']
        height_ratio = self.output_size[0] / sample['h']
        for i in range(gt_bbox.shape[0]):
            width = gt_bbox[i][2] - gt_bbox[i][0]
            height = gt_bbox[i][3] - gt_bbox[i][1]

            xtl, ytl = gt_bbox[i][0], gt_bbox[i][1]
            xbr, ybr = gt_bbox[i][2], gt_bbox[i][3]

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)
            if self.gaussian_bump:
                width = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)
                if self.gaussian_rad == -1:
                    radius = gaussian_radius((height, width), self.gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = self.gaussian_rad
                draw_gaussian(tl_heatmaps[gt_class[i][0]], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[gt_class[i][0]], [xbr, ybr], radius)
            else:
                tl_heatmaps[gt_class[i][0], ytl, xtl] = 1
                br_heatmaps[gt_class[i][0], ybr, xbr] = 1

            tl_regrs[i, :] = [fxtl - xtl, fytl - ytl]
            br_regrs[i, :] = [fxbr - xbr, fybr - ybr]
            tl_tags[tag_lens] = ytl * self.output_size[1] + xtl
            br_tags[tag_lens] = ybr * self.output_size[1] + xbr
            tag_lens += 1

        tag_masks[:tag_lens] = 1

        sample['tl_heatmaps'] = tl_heatmaps
        sample['br_heatmaps'] = br_heatmaps
        sample['tl_regrs'] = tl_regrs
        sample['br_regrs'] = br_regrs
        sample['tl_tags'] = tl_tags
        sample['br_tags'] = br_tags
        sample['tag_masks'] = tag_masks

        return sample


@register_op
class CornerCrop(BaseOperator):
    """
    Random crop for CornerNet
    Args:
        random_scales (list): scales of output_size to input_size.
        border (int): border of corp center
        is_train (bool): train or test
        input_size (int): size of input image
    """

    def __init__(self,
                 random_scales=[0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3],
                 border=128,
                 is_train=True,
                 input_size=511):
        super(CornerCrop, self).__init__()
        self.random_scales = random_scales
        self.border = border
        self.is_train = is_train
        self.input_size = input_size

    def __call__(self, sample, context=None):
        im_h, im_w = int(sample['h']), int(sample['w'])
        if self.is_train:
            scale = np.random.choice(self.random_scales)
            height = int(self.input_size * scale)
            width = int(self.input_size * scale)

            w_border = self._get_border(self.border, im_w)
            h_border = self._get_border(self.border, im_h)

            ctx = np.random.randint(low=w_border, high=im_w - w_border)
            cty = np.random.randint(low=h_border, high=im_h - h_border)

        else:
            cty, ctx = im_h // 2, im_w // 2
            height = im_h | 127
            width = im_w | 127

        cropped_image = np.zeros(
            (height, width, 3), dtype=sample['image'].dtype)

        x0, x1 = max(ctx - width // 2, 0), min(ctx + width // 2, im_w)
        y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, im_h)

        left_w, right_w = ctx - x0, x1 - ctx
        top_h, bottom_h = cty - y0, y1 - cty

        # crop image
        cropped_ctx, cropped_cty = width // 2, height // 2
        x_slice = slice(int(cropped_ctx - left_w), int(cropped_ctx + right_w))
        y_slice = slice(int(cropped_cty - top_h), int(cropped_cty + bottom_h))
        cropped_image[y_slice, x_slice, :] = sample['image'][y0:y1, x0:x1, :]

        sample['image'] = cropped_image
        sample['h'], sample['w'] = height, width

        if self.is_train:
            # crop detections
            gt_bbox = sample['gt_bbox']
            gt_bbox[:, 0:4:2] -= x0
            gt_bbox[:, 1:4:2] -= y0
            gt_bbox[:, 0:4:2] += cropped_ctx - left_w
            gt_bbox[:, 1:4:2] += cropped_cty - top_h
        else:
            sample['borders'] = np.array(
                [
                    cropped_cty - top_h, cropped_cty + bottom_h,
                    cropped_ctx - left_w, cropped_ctx + right_w
                ],
                dtype=np.float32)

        return sample

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i


@register_op
class CornerRatio(BaseOperator):
    """
    Ratio of output size to image size
    Args:
        input_size (int): the size of input size
        output_size (int): the size of heatmap
    """

    def __init__(self, input_size=511, output_size=64):
        super(CornerRatio, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, sample, context=None):
        scale = (self.input_size + 1) // self.output_size
        out_height, out_width = (sample['h'] + 1) // scale, (
            sample['w'] + 1) // scale
        height_ratio = out_height / float(sample['h'])
        width_ratio = out_width / float(sample['w'])
        sample['ratios'] = np.array([height_ratio, width_ratio])

        return sample


@register_op
class RandomScaledCrop(BaseOperator):
    """Resize image and bbox based on long side (with optional random scaling),
       then crop or pad image to target size.
    Args:
        target_dim (int): target size.
        scale_range (list): random scale range.
        interp (int): interpolation method, default to `cv2.INTER_LINEAR`.
    """

    def __init__(self,
                 target_dim=512,
                 scale_range=[.1, 2.],
                 interp=cv2.INTER_LINEAR):
        super(RandomScaledCrop, self).__init__()
        self.target_dim = target_dim
        self.scale_range = scale_range
        self.interp = interp

    def __call__(self, sample, context=None):
        w = sample['w']
        h = sample['h']
        random_scale = np.random.uniform(*self.scale_range)
        dim = self.target_dim
        random_dim = int(dim * random_scale)
        dim_max = max(h, w)
        scale = random_dim / dim_max
        resize_w = int(round(w * scale))
        resize_h = int(round(h * scale))
        offset_x = int(max(0, np.random.uniform(0., resize_w - dim)))
        offset_y = int(max(0, np.random.uniform(0., resize_h - dim)))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale, scale] * 2, dtype=np.float32)
            shift_array = np.array([offset_x, offset_y] * 2, dtype=np.float32)
            boxes = sample['gt_bbox'] * scale_array - shift_array
            boxes = np.clip(boxes, 0, dim - 1)
            # filter boxes with no area
            area = np.prod(boxes[..., 2:] - boxes[..., :2], axis=1)
            valid = (area > 1.).nonzero()[0]
            sample['gt_bbox'] = boxes[valid]
            sample['gt_class'] = sample['gt_class'][valid]

        img = sample['image']
        img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interp)
        img = np.array(img)
        canvas = np.zeros((dim, dim, 3), dtype=img.dtype)
        canvas[:min(dim, resize_h), :min(dim, resize_w), :] = img[
            offset_y:offset_y + dim, offset_x:offset_x + dim, :]
        sample['h'] = dim
        sample['w'] = dim
        sample['image'] = canvas
        sample['im_info'] = [resize_h, resize_w, scale]
        return sample


@register_op
class ResizeAndPad(BaseOperator):
    """Resize image and bbox, then pad image to target size.
    Args:
        target_dim (int): target size
        interp (int): interpolation method, default to `cv2.INTER_LINEAR`.
    """

    def __init__(self, target_dim=512, interp=cv2.INTER_LINEAR):
        super(ResizeAndPad, self).__init__()
        self.target_dim = target_dim
        self.interp = interp

    def __call__(self, sample, context=None):
        w = sample['w']
        h = sample['h']
        interp = self.interp
        dim = self.target_dim
        dim_max = max(h, w)
        scale = self.target_dim / dim_max
        resize_w = int(round(w * scale))
        resize_h = int(round(h * scale))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale, scale] * 2, dtype=np.float32)
            sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0,
                                        dim - 1)
        img = sample['image']
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
        img = np.array(img)
        canvas = np.zeros((dim, dim, 3), dtype=img.dtype)
        canvas[:resize_h, :resize_w, :] = img
        sample['h'] = dim
        sample['w'] = dim
        sample['image'] = canvas
        sample['im_info'] = [resize_h, resize_w, scale]
        return sample


@register_op
class TargetAssign(BaseOperator):
    """Assign regression target and labels.
    Args:
        image_size (int or list): input image size, a single integer or list of
            [h, w]. Default: 512
        min_level (int): min level of the feature pyramid. Default: 3
        max_level (int): max level of the feature pyramid. Default: 7
        anchor_base_scale (int): base anchor scale. Default: 4
        num_scales (int): number of anchor scales. Default: 3
        aspect_ratios (list): aspect ratios.
            Default: [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        match_threshold (float): threshold for foreground IoU. Default: 0.5
    """

    def __init__(self,
                 image_size=512,
                 min_level=3,
                 max_level=7,
                 anchor_base_scale=4,
                 num_scales=3,
                 aspect_ratios=[(1, 1), (1.4, 0.7), (0.7, 1.4)],
                 match_threshold=0.5):
        super(TargetAssign, self).__init__()
        assert image_size % 2 ** max_level == 0, \
            "image size should be multiple of the max level stride"
        self.image_size = image_size
        self.min_level = min_level
        self.max_level = max_level
        self.anchor_base_scale = anchor_base_scale
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.match_threshold = match_threshold

    @property
    def anchors(self):
        if not hasattr(self, '_anchors'):
            anchor_grid = AnchorGrid(self.image_size, self.min_level,
                                     self.max_level, self.anchor_base_scale,
                                     self.num_scales, self.aspect_ratios)
            self._anchors = np.concatenate(anchor_grid.generate())
        return self._anchors

    def iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        # return area_i / (area_o + 1e-10)
        return np.where(area_i == 0., np.zeros_like(area_i), area_i / area_o)

    def match(self, anchors, gt_boxes):
        # XXX put smaller matrix first would be a little bit faster
        mat = self.iou_matrix(gt_boxes, anchors)
        max_anchor_for_each_gt = mat.argmax(axis=1)
        max_for_each_anchor = mat.max(axis=0)
        anchor_to_gt = mat.argmax(axis=0)
        anchor_to_gt[max_for_each_anchor < self.match_threshold] = -1
        # XXX ensure each gt has at least one anchor assigned,
        # see `force_match_for_each_row` in TF implementation
        one_hot = np.zeros_like(mat)
        one_hot[np.arange(mat.shape[0]), max_anchor_for_each_gt] = 1.
        max_anchor_indices = one_hot.sum(axis=0).nonzero()[0]
        max_gt_indices = one_hot.argmax(axis=0)[max_anchor_indices]
        anchor_to_gt[max_anchor_indices] = max_gt_indices
        return anchor_to_gt

    def encode(self, anchors, boxes):
        wha = anchors[..., 2:] - anchors[..., :2] + 1
        ca = anchors[..., :2] + wha * .5
        whb = boxes[..., 2:] - boxes[..., :2] + 1
        cb = boxes[..., :2] + whb * .5
        offsets = np.empty_like(anchors)
        offsets[..., :2] = (cb - ca) / wha
        offsets[..., 2:] = np.log(whb / wha)
        return offsets

    def __call__(self, sample, context=None):
        gt_boxes = sample['gt_bbox']
        gt_labels = sample['gt_class']
        labels = np.full((self.anchors.shape[0], 1), 0, dtype=np.int32)
        targets = np.full((self.anchors.shape[0], 4), 0., dtype=np.float32)
        sample['gt_label'] = labels
        sample['gt_target'] = targets

        if len(gt_boxes) < 1:
            sample['fg_num'] = np.array(0, dtype=np.int32)
            return sample

        anchor_to_gt = self.match(self.anchors, gt_boxes)
        matched_indices = (anchor_to_gt >= 0).nonzero()[0]
        labels[matched_indices] = gt_labels[anchor_to_gt[matched_indices]]

        matched_boxes = gt_boxes[anchor_to_gt[matched_indices]]
        matched_anchors = self.anchors[matched_indices]
        matched_targets = self.encode(matched_anchors, matched_boxes)
        targets[matched_indices] = matched_targets
        sample['fg_num'] = np.array(len(matched_targets), dtype=np.int32)
        return sample


@register_op
class DebugVisibleImage(BaseOperator):
    """
    In debug mode, visualize images according to `gt_box`.
    (Currently only supported when not cropping and flipping image.)
    """

    def __init__(self,
                 output_dir='output/debug',
                 use_vdl=False,
                 is_normalized=False):
        super(DebugVisibleImage, self).__init__()
        self.is_normalized = is_normalized
        self.output_dir = output_dir
        self.use_vdl = use_vdl
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if not isinstance(self.is_normalized, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if self.use_vdl:
            assert six.PY3, "VisualDL requires Python >= 3.5"
            from visualdl import LogWriter
            self.vdl_writer = LogWriter(self.output_dir)

    def __call__(self, sample, context=None):
        out_file_name = sample['im_file'].split('/')[-1]
        if self.use_vdl:
            origin_image = Image.open(sample['im_file']).convert('RGB')
            origin_image = ImageOps.exif_transpose(origin_image)
            image_np = np.array(origin_image)
            self.vdl_writer.add_image("original/{}".format(out_file_name),
                                      image_np, 0)

        if not isinstance(sample['image'], np.ndarray):
            raise TypeError("{}: sample[image] type is not numpy.".format(self))
        image = Image.fromarray(np.uint8(sample['image']))

        width = sample['w']
        height = sample['h']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']

        if 'gt_poly' in sample.keys():
            poly_to_mask = Poly2Mask()
            sample = poly_to_mask(sample)

        if 'gt_segm' in sample.keys():
            import pycocotools.mask as mask_util
            from ppdet.utils.colormap import colormap
            image_np = np.array(image).astype('float32')
            mask_color_id = 0
            w_ratio = .4
            alpha = 0.7
            color_list = colormap(rgb=True)
            gt_segm = sample['gt_segm']
            for mask in gt_segm:
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                mask_color_id += 1
                for c in range(3):
                    color_mask[c] = color_mask[c] * (1 - w_ratio
                                                     ) + w_ratio * 255
                idx = np.nonzero(mask)
                image_np[idx[0], idx[1], :] *= 1.0 - alpha
                image_np[idx[0], idx[1], :] += alpha * color_mask
            image = Image.fromarray(np.uint8(image_np))

        draw = ImageDraw.Draw(image)
        for i in range(gt_bbox.shape[0]):
            if self.is_normalized:
                gt_bbox[i][0] = gt_bbox[i][0] * width
                gt_bbox[i][1] = gt_bbox[i][1] * height
                gt_bbox[i][2] = gt_bbox[i][2] * width
                gt_bbox[i][3] = gt_bbox[i][3] * height

            xmin, ymin, xmax, ymax = gt_bbox[i]
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill='green')
            # draw label
            text = 'id' + str(gt_class[i][0])
            tw, th = draw.textsize(text)
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill='green')
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']
            if self.is_normalized:
                for i in range(gt_keypoint.shape[1]):
                    if i % 2:
                        gt_keypoint[:, i] = gt_keypoint[:, i] * height
                    else:
                        gt_keypoint[:, i] = gt_keypoint[:, i] * width
            for i in range(gt_keypoint.shape[0]):
                keypoint = gt_keypoint[i]
                for j in range(int(keypoint.shape[0] / 2)):
                    x1 = round(keypoint[2 * j])
                    y1 = round(keypoint[2 * j + 1])
                    draw.ellipse(
                        (x1, y1, x1 + 5, y1 + 5), fill='green', outline='green')
        save_path = os.path.join(self.output_dir, out_file_name)
        if self.use_vdl:
            preprocess_image_np = np.array(image)
            self.vdl_writer.add_image("preprocess/{}".format(out_file_name),
                                      preprocess_image_np, 0)
        else:
            image.save(save_path, quality=95)
        return sample


@register_op
class Poly2Mask(BaseOperator):
    """
    gt poly to mask annotations
    """

    def __init__(self):
        super(Poly2Mask, self).__init__()
        import pycocotools.mask as maskUtils
        self.maskutils = maskUtils

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = self.maskutils.frPyObjects(mask_ann, img_h, img_w)
            rle = self.maskutils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = self.maskutils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = self.maskutils.decode(rle)
        return mask

    def __call__(self, sample, context=None):
        assert 'gt_poly' in sample
        im_h = sample['h']
        im_w = sample['w']
        masks = [
            self._poly2mask(gt_poly, im_h, im_w)
            for gt_poly in sample['gt_poly']
        ]
        sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample
