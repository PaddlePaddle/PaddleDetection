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

from numbers import Number, Integral

import uuid
import random
import math
import numpy as np
import os
import copy
import logging
import cv2
from PIL import Image, ImageDraw
import pickle
import threading
MUTEX = threading.Lock()

from ppdet.core.workspace import serializable
from ppdet.modeling import bbox_utils
from ..reader import Compose

from .op_helper import (satisfy_sample_constraint, filter_and_process,
                        generate_sample_bbox, clip_bbox, data_anchor_sampling,
                        satisfy_sample_constraint_coverage, crop_image_sampling,
                        generate_sample_bbox_square, bbox_area_sampling,
                        is_poly, get_border)

from ppdet.utils.logger import setup_logger
from ppdet.modeling.keypoint_utils import get_affine_transform, affine_transform
logger = setup_logger(__name__)

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
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


@register_op
class Decode(BaseOperator):
    def __init__(self):
        """ Transform the image data to numpy format following the rgb format
        """
        super(Decode, self).__init__()

    def apply(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()
            sample.pop('im_file')

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if 'keep_ori_im' in sample and sample['keep_ori_im']:
            sample['ori_image'] = im
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

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample


def _make_dirs(dirname):
    try:
        from pathlib import Path
    except ImportError:
        from pathlib2 import Path
    Path(dirname).mkdir(exist_ok=True)


@register_op
class DecodeCache(BaseOperator):
    def __init__(self, cache_root=None):
        '''decode image and caching
        '''
        super(DecodeCache, self).__init__()

        self.use_cache = False if cache_root is None else True
        self.cache_root = cache_root

        if cache_root is not None:
            _make_dirs(cache_root)

    def apply(self, sample, context=None):

        if self.use_cache and os.path.exists(
                self.cache_path(self.cache_root, sample['im_file'])):
            path = self.cache_path(self.cache_root, sample['im_file'])
            im = self.load(path)

        else:
            if 'image' not in sample:
                with open(sample['im_file'], 'rb') as f:
                    sample['image'] = f.read()

            im = sample['image']
            data = np.frombuffer(im, dtype='uint8')
            im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
            if 'keep_ori_im' in sample and sample['keep_ori_im']:
                sample['ori_image'] = im
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if self.use_cache and not os.path.exists(
                    self.cache_path(self.cache_root, sample['im_file'])):
                path = self.cache_path(self.cache_root, sample['im_file'])
                self.dump(im, path)

        sample['image'] = im
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)

        sample.pop('im_file')

        return sample

    @staticmethod
    def cache_path(dir_oot, im_file):
        return os.path.join(dir_oot, os.path.basename(im_file) + '.pkl')

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            im = pickle.load(f)
        return im

    @staticmethod
    def dump(obj, path):
        MUTEX.acquire()
        try:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)

        except Exception as e:
            logger.warning('dump {} occurs exception {}'.format(path, str(e)))

        finally:
            MUTEX.release()


@register_op
class SniperDecodeCrop(BaseOperator):
    def __init__(self):
        super(SniperDecodeCrop, self).__init__()

    def __call__(self, sample, context=None):
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()
            sample.pop('im_file')

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR mode, but need RGB mode
        if 'keep_ori_im' in sample and sample['keep_ori_im']:
            sample['ori_image'] = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        chip = sample['chip']
        x1, y1, x2, y2 = [int(xi) for xi in chip]
        im = im[max(y1, 0):min(y2, im.shape[0]), max(x1, 0):min(x2, im.shape[
            1]), :]

        sample['image'] = im
        h = im.shape[0]
        w = im.shape[1]
        # sample['im_info'] = [h, w, 1.0]
        sample['h'] = h
        sample['w'] = w

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample


@register_op
class Permute(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (C, H, W)
        """
        super(Permute, self).__init__()

    def apply(self, sample, context=None):
        im = sample['image']
        im = im.transpose((2, 0, 1))
        sample['image'] = im
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

    def apply(self, sample, context=None):
        alpha = np.random.normal(scale=self.alphastd, size=(3, ))
        sample['image'] += np.dot(self.eigvec, self.eigval * alpha)
        return sample


@register_op
class RandomErasingImage(BaseOperator):
    def __init__(self, prob=0.5, lower=0.02, higher=0.4, aspect_ratio=0.3):
        """
        Random Erasing Data Augmentation, see https://arxiv.org/abs/1708.04896
        Args:
            prob (float): probability to carry out random erasing
            lower (float): lower limit of the erasing area ratio
            higher (float): upper limit of the erasing area ratio
            aspect_ratio (float): aspect ratio of the erasing region
        """
        super(RandomErasingImage, self).__init__()
        self.prob = prob
        self.lower = lower
        self.higher = higher
        self.aspect_ratio = aspect_ratio

    def apply(self, sample, context=None):
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
            w_bbox = x2 - x1
            h_bbox = y2 - y1
            area = w_bbox * h_bbox

            target_area = random.uniform(self.lower, self.higher) * area
            aspect_ratio = random.uniform(self.aspect_ratio,
                                          1 / self.aspect_ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < w_bbox and h < h_bbox:
                off_y1 = random.randint(0, int(h_bbox - h))
                off_x1 = random.randint(0, int(w_bbox - w))
                im[int(y1 + off_y1):int(y1 + off_y1 + h), int(x1 + off_x1):int(
                    x1 + off_x1 + w), :] = 0
        sample['image'] = im
        return sample


@register_op
class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1],
                 is_scale=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
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
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['image']
        im = im.astype(np.float32, copy=False)
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            im = im / 255.0

        im -= mean
        im /= std

        sample['image'] = im
        return sample


@register_op
class GridMask(BaseOperator):
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
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.upper_iter = upper_iter

        from .gridmask_utils import Gridmask
        self.gridmask_op = Gridmask(
            use_h,
            use_w,
            rotate=rotate,
            offset=offset,
            ratio=ratio,
            mode=mode,
            prob=prob,
            upper_iter=upper_iter)

    def apply(self, sample, context=None):
        sample['image'] = self.gridmask_op(sample['image'], sample['curr_iter'])
        return sample


@register_op
class RandomDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        # it works, but result differ from HSV version
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
        # it works, but result differ from HSV version
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

    def apply(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        img = self.apply_saturation(img)
        img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


@register_op
class AutoAugment(BaseOperator):
    def __init__(self, autoaug_type="v1"):
        """
        Args:
            autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
        """
        super(AutoAugment, self).__init__()
        self.autoaug_type = autoaug_type

    def apply(self, sample, context=None):
        """
        Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
        """
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image is not a numpy array.".format(self))
        if len(im.shape) != 3:
            raise ImageError("{}: image is not 3-dimensional.".format(self))
        if len(gt_bbox) == 0:
            return sample

        height, width, _ = im.shape
        norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
        norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
        norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
        norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
        norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)

        from .autoaugment_utils import distort_image_with_autoaugment
        im, norm_gt_bbox = distort_image_with_autoaugment(im, norm_gt_bbox,
                                                          self.autoaug_type)

        gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
        gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
        gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
        gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)

        sample['image'] = im
        sample['gt_bbox'] = gt_bbox
        return sample


@register_op
class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
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

    def apply_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, i] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        bbox = [bbox_utils.get_best_begin_point_single(e) for e in bbox]
        return bbox

    def apply(self, sample, context=None):
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
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], height,
                                                    width)
            if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
                sample['gt_keypoint'] = self.apply_keypoint(
                    sample['gt_keypoint'], width)

            if 'semantic' in sample and sample['semantic']:
                sample['semantic'] = sample['semantic'][:, ::-1]

            if 'gt_segm' in sample and sample['gt_segm'].any():
                sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

            if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
                sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'],
                                                         width)

            sample['flipped'] = True
            sample['image'] = im
        return sample


@register_op
class Resize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True, 
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
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

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
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

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warning(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                    format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
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

        # apply gt_segm
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
class MultiscaleTestResize(BaseOperator):
    def __init__(self,
                 origin_target_size=[800, 1333],
                 target_size=[],
                 interp=cv2.INTER_LINEAR,
                 use_flip=True):
        """
        Rescale image to the each size in target size, and capped at max_size.
        Args:
            origin_target_size (list): origin target size of image
            target_size (list): A list of target sizes of image.
            interp (int): the interpolation method.
            use_flip (bool): whether use flip augmentation.
        """
        super(MultiscaleTestResize, self).__init__()
        self.interp = interp
        self.use_flip = use_flip

        if not isinstance(target_size, Sequence):
            raise TypeError(
                "Type of target_size is invalid. Must be List or Tuple, now is {}".
                format(type(target_size)))
        self.target_size = target_size

        if not isinstance(origin_target_size, Sequence):
            raise TypeError(
                "Type of origin_target_size is invalid. Must be List or Tuple, now is {}".
                format(type(origin_target_size)))

        self.origin_target_size = origin_target_size

    def apply(self, sample, context=None):
        """ Resize the image numpy for multi-scale test.
        """
        samples = []
        resizer = Resize(
            self.origin_target_size, keep_ratio=True, interp=self.interp)
        samples.append(resizer(sample.copy(), context))
        if self.use_flip:
            flipper = RandomFlip(1.1)
            samples.append(flipper(sample.copy(), context=context))

        for size in self.target_size:
            resizer = Resize(size, keep_ratio=True, interp=self.interp)
            samples.append(resizer(sample.copy(), context))

        return samples


@register_op
class RandomResize(BaseOperator):
    def __init__(self,
                 target_size,
                 keep_ratio=True,
                 interp=cv2.INTER_LINEAR,
                 random_size=True,
                 random_interp=False):
        """
        Resize image to target size randomly. random target_size and interpolation method
        Args:
            target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
            keep_ratio (bool): whether keep_raio or not, default true
            interp (int): the interpolation method
            random_size (bool): whether random select target size of image
            random_interp (bool): whether random select interpolation method
        """
        super(RandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        assert isinstance(target_size, (
            Integral, Sequence)), "target_size must be Integer, List or Tuple"
        if random_size and not isinstance(target_size, Sequence):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List or Tuple, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        if self.random_size:
            target_size = random.choice(self.target_size)
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, self.keep_ratio, interp)
        return resizer(sample, context=context)


@register_op
class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
    """

    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5, 127.5, 127.5)):
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

    def apply(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        im = sample['image']
        height, width = im.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        pad = Pad(size,
                  pad_mode=-1,
                  offsets=offsets,
                  fill_value=self.fill_value)

        return pad(sample, context=context)


@register_op
class CropWithSampling(BaseOperator):
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
            avoid_no_bbox (bool): whether to avoid the
                                  situation where the box does not appear.
        """
        super(CropWithSampling, self).__init__()
        self.batch_sampler = batch_sampler
        self.satisfy_all = satisfy_all
        self.avoid_no_bbox = avoid_no_bbox

    def apply(self, sample, context):
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
        im_height, im_width = im.shape[:2]
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
class CropWithDataAchorSampling(BaseOperator):
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
            target_size (int): target image size.
            das_anchor_scales (list[float]): a list of anchor scales in data
                anchor smapling.
            min_size (float): minimum size of sampled bbox.
            avoid_no_bbox (bool): whether to avoid the
                                  situation where the box does not appear.
        """
        super(CropWithDataAchorSampling, self).__init__()
        self.anchor_sampler = anchor_sampler
        self.batch_sampler = batch_sampler
        self.target_size = target_size
        self.sampling_prob = sampling_prob
        self.min_size = min_size
        self.avoid_no_bbox = avoid_no_bbox
        self.das_anchor_scales = np.array(das_anchor_scales)

    def apply(self, sample, context):
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
        image_height, image_width = im.shape[:2]
        gt_bbox[:, 0] /= image_width
        gt_bbox[:, 1] /= image_height
        gt_bbox[:, 2] /= image_width
        gt_bbox[:, 3] /= image_height
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
                height, width = im.shape[:2]
                crop_bbox[:, 0] *= width
                crop_bbox[:, 1] *= height
                crop_bbox[:, 2] *= width
                crop_bbox[:, 3] *= height
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                if 'gt_score' in sample:
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
                height, width = im.shape[:2]
                crop_bbox[:, 0] *= width
                crop_bbox[:, 1] *= height
                crop_bbox[:, 2] *= width
                crop_bbox[:, 3] *= height
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                if 'gt_score' in sample:
                    sample['gt_score'] = crop_score
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                return sample
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

    def apply(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h, w = sample['image'].shape[:2]
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
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)

                if 'difficult' in sample:
                    sample['difficult'] = np.take(
                        sample['difficult'], valid_ids, axis=0)

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

    def apply(self, sample, context=None):
        img = sample['image']
        h, w = img.shape[:2]
        random_scale = np.random.uniform(*self.scale_range)
        dim = self.target_dim
        random_dim = int(dim * random_scale)
        dim_max = max(h, w)
        scale = random_dim / dim_max
        resize_w = w * scale
        resize_h = h * scale
        offset_x = int(max(0, np.random.uniform(0., resize_w - dim)))
        offset_y = int(max(0, np.random.uniform(0., resize_h - dim)))

        img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interp)
        img = np.array(img)
        canvas = np.zeros((dim, dim, 3), dtype=img.dtype)
        canvas[:min(dim, resize_h), :min(dim, resize_w), :] = img[
            offset_y:offset_y + dim, offset_x:offset_x + dim, :]
        sample['image'] = canvas
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        scale_factor = sample['sacle_factor']
        sample['scale_factor'] = np.asarray(
            [scale_factor[0] * scale, scale_factor[1] * scale],
            dtype=np.float32)

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

        return sample


@register_op
class Cutmix(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ 
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(Cutmix, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def apply_image(self, img1, img2, factor):
        """ _rand_bbox """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        cut_rat = np.sqrt(1. - factor)

        cut_w = np.int32(w * cut_rat)
        cut_h = np.int32(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w - 1)
        bby1 = np.clip(cy - cut_h // 2, 0, h - 1)
        bbx2 = np.clip(cx + cut_w // 2, 0, w - 1)
        bby2 = np.clip(cy + cut_h // 2, 0, h - 1)

        img_1_pad = np.zeros((h, w, img1.shape[2]), 'float32')
        img_1_pad[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32')
        img_2_pad = np.zeros((h, w, img2.shape[2]), 'float32')
        img_2_pad[:img2.shape[0], :img2.shape[1], :] = \
            img2.astype('float32')
        img_1_pad[bby1:bby2, bbx1:bbx2, :] = img_2_pad[bby1:bby2, bbx1:bbx2, :]
        return img_1_pad

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'cutmix need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        img1 = sample[0]['image']
        img2 = sample[1]['image']
        img = self.apply_image(img1, img2, factor)
        gt_bbox1 = sample[0]['gt_bbox']
        gt_bbox2 = sample[1]['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample[0]['gt_class']
        gt_class2 = sample[1]['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = np.ones_like(sample[0]['gt_class'])
        gt_score2 = np.ones_like(sample[1]['gt_class'])
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        result = copy.deepcopy(sample[0])
        result['image'] = img
        result['gt_bbox'] = gt_bbox
        result['gt_score'] = gt_score
        result['gt_class'] = gt_class
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult
        return result


@register_op
class Mixup(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def apply_image(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'mixup need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        im = self.apply_image(sample[0]['image'], sample[1]['image'], factor)
        result = copy.deepcopy(sample[0])
        result['image'] = im
        # apply bbox and score
        if 'gt_bbox' in sample[0]:
            gt_bbox1 = sample[0]['gt_bbox']
            gt_bbox2 = sample[1]['gt_bbox']
            gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
            result['gt_bbox'] = gt_bbox
        if 'gt_class' in sample[0]:
            gt_class1 = sample[0]['gt_class']
            gt_class2 = sample[1]['gt_class']
            gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
            result['gt_class'] = gt_class

            gt_score1 = np.ones_like(sample[0]['gt_class'])
            gt_score2 = np.ones_like(sample[1]['gt_class'])
            gt_score = np.concatenate(
                (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
            result['gt_score'] = gt_score.astype('float32')
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult

        if 'gt_ide' in sample[0]:
            gt_ide1 = sample[0]['gt_ide']
            gt_ide2 = sample[1]['gt_ide']
            gt_ide = np.concatenate((gt_ide1, gt_ide2), axis=0)
            result['gt_ide'] = gt_ide
        return result


@register_op
class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def apply(self, sample, context):
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        height, width, _ = im.shape
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
class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


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

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes
        # fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in sample:
            pad_class = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in sample:
            pad_score = np.zeros((num_max, ), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'difficult' in sample:
            pad_diff = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        if 'is_crowd' in sample:
            pad_crowd = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_crowd[:gt_num] = sample['is_crowd'][:gt_num, 0]
            sample['is_crowd'] = pad_crowd
        if 'gt_ide' in sample:
            pad_ide = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_ide[:gt_num] = sample['gt_ide'][:gt_num, 0]
            sample['gt_ide'] = pad_ide
        return sample


@register_op
class DebugVisibleImage(BaseOperator):
    """
    In debug mode, visualize images according to `gt_box`.
    (Currently only supported when not cropping and flipping image.)
    """

    def __init__(self, output_dir='output/debug', is_normalized=False):
        super(DebugVisibleImage, self).__init__()
        self.is_normalized = is_normalized
        self.output_dir = output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if not isinstance(self.is_normalized, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply(self, sample, context=None):
        image = Image.fromarray(sample['image'].astype(np.uint8))
        out_file_name = '{:012d}.jpg'.format(sample['im_id'][0])
        width = sample['w']
        height = sample['h']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
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
            text = str(gt_class[i][0])
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
                    x1 = round(keypoint[2 * j]).astype(np.int32)
                    y1 = round(keypoint[2 * j + 1]).astype(np.int32)
                    draw.ellipse(
                        (x1, y1, x1 + 5, y1 + 5), fill='green', outline='green')
        save_path = os.path.join(self.output_dir, out_file_name)
        image.save(save_path, quality=95)
        return sample


@register_op
class Pad(BaseOperator):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, Sequence): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """
        super(Pad, self).__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_segm(self, segms, offsets, im_size, size):
        def _expand_poly(poly, x, y):
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, h, w):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((h, w), 0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        x, y = offsets
        height, width = im_size
        h, w = size
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
                    _expand_rle(segm, x, y, height, width, h, w))
        return expanded_segms

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_keypoint(self, keypoints, offsets):
        n = len(keypoints[0]) // 2
        return keypoints + np.array(offsets * n, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def apply(self, sample, context=None):
        im = sample['image']
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert (
                im_h <= h and im_w <= w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            sample['image'] = im.astype(np.float32)
            return sample

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        sample['image'] = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            return sample
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], offsets,
                                                im_size, size)

        if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
            sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'],
                                                        offsets)

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

    def apply(self, sample, context=None):
        assert 'gt_poly' in sample
        im_h = sample['h']
        im_w = sample['w']
        masks = [
            self._poly2mask(gt_poly, im_h, im_w)
            for gt_poly in sample['gt_poly']
        ]
        sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample


@register_op
class Rbox2Poly(BaseOperator):
    """
    Convert rbbox format to poly format.
    """

    def __init__(self):
        super(Rbox2Poly, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_rbox' in sample
        assert sample['gt_rbox'].shape[1] == 5
        rrects = sample['gt_rbox']
        x_ctr = rrects[:, 0]
        y_ctr = rrects[:, 1]
        width = rrects[:, 2]
        height = rrects[:, 3]
        x1 = x_ctr - width / 2.0
        y1 = y_ctr - height / 2.0
        x2 = x_ctr + width / 2.0
        y2 = y_ctr + height / 2.0
        sample['gt_bbox'] = np.stack([x1, y1, x2, y2], axis=1)
        polys = bbox_utils.rbox2poly_np(rrects)
        sample['gt_rbox2poly'] = polys
        return sample


@register_op
class AugmentHSV(BaseOperator):
    """ 
    Augment the SV channel of image data.
    Args:
        fraction (float): the fraction for augment. Default: 0.5.
        is_bgr (bool): whether the image is BGR mode. Default: True.
        hgain (float): H channel gains
        sgain (float): S channel gains
        vgain (float): V channel gains
    """

    def __init__(self,
                 fraction=0.50,
                 is_bgr=True,
                 hgain=None,
                 sgain=None,
                 vgain=None):
        super(AugmentHSV, self).__init__()
        self.fraction = fraction
        self.is_bgr = is_bgr
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.use_hsvgain = False if hgain is None else True

    def apply(self, sample, context=None):
        img = sample['image']
        if self.is_bgr:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        if self.use_hsvgain:
            hsv_augs = np.random.uniform(
                -1, 1, 3) * [self.hgain, self.sgain, self.vgain]
            # random selection of h, s, v
            hsv_augs *= np.random.randint(0, 2, 3)
            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

        else:
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * self.fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * self.fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)

        if self.is_bgr:
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        else:
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)

        sample['image'] = img
        return sample


@register_op
class Norm2PixelBbox(BaseOperator):
    """
    Transform the bounding box's coornidates which is in [0,1] to pixels.
    """

    def __init__(self):
        super(Norm2PixelBbox, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        height, width = sample['image'].shape[:2]
        bbox[:, 0::2] = bbox[:, 0::2] * width
        bbox[:, 1::2] = bbox[:, 1::2] * height
        sample['gt_bbox'] = bbox
        return sample


@register_op
class BboxCXCYWH2XYXY(BaseOperator):
    """
    Convert bbox CXCYWH format to XYXY format.
    [center_x, center_y, width, height] -> [x0, y0, x1, y1]
    """

    def __init__(self):
        super(BboxCXCYWH2XYXY, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox0 = sample['gt_bbox']
        bbox = bbox0.copy()

        bbox[:, :2] = bbox0[:, :2] - bbox0[:, 2:4] / 2.
        bbox[:, 2:4] = bbox0[:, :2] + bbox0[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


@register_op
class RandomResizeCrop(BaseOperator):
    """Random resize and crop image and bboxes.
    Args:
        resizes (list): resize image to one of resizes. if keep_ratio is True and mode is
        'long', resize the image's long side to the maximum of target_size, if keep_ratio is
        True and mode is 'short', resize the image's short side to the minimum of target_size.
        cropsizes (list): crop sizes after resize, [(min_crop_1, max_crop_1), ...]
        mode (str): resize mode, `long` or `short`. Details see resizes. 
        prob (float): probability of this op.
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): the interpolation method
        thresholds (list): iou thresholds for decide a valid bbox crop.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(
            self,
            resizes,
            cropsizes,
            prob=0.5,
            mode='short',
            keep_ratio=True,
            interp=cv2.INTER_LINEAR,
            num_attempts=3,
            cover_all_box=False,
            allow_no_crop=False,
            thresholds=[0.3, 0.5, 0.7],
            is_mask_crop=False, ):
        super(RandomResizeCrop, self).__init__()

        self.resizes = resizes
        self.cropsizes = cropsizes
        self.prob = prob
        self.mode = mode

        self.resizer = Resize(0, keep_ratio=keep_ratio, interp=interp)
        self.croper = RandomCrop(
            num_attempts=num_attempts,
            cover_all_box=cover_all_box,
            thresholds=thresholds,
            allow_no_crop=allow_no_crop,
            is_mask_crop=is_mask_crop)

    def _format_size(self, size):
        if isinstance(size, Integral):
            size = (size, size)
        return size

    def apply(self, sample, context=None):
        if random.random() < self.prob:
            _resize = self._format_size(random.choice(self.resizes))
            _cropsize = self._format_size(random.choice(self.cropsizes))
            sample = self._resize(
                self.resizer,
                sample,
                size=_resize,
                mode=self.mode,
                context=context)
            sample = self._random_crop(
                self.croper, sample, size=_cropsize, context=context)
        return sample

    @staticmethod
    def _random_crop(croper, sample, size, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        self = croper
        h, w = sample['image'].shape[:2]
        gt_bbox = sample['gt_bbox']
        cropsize = size
        min_crop = min(cropsize)
        max_crop = max(cropsize)

        thresholds = list(self.thresholds)
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            found = False
            for _ in range(self.num_attempts):

                crop_h = random.randint(min_crop, min(h, max_crop))
                crop_w = random.randint(min_crop, min(w, max_crop))

                crop_y = random.randint(0, h - crop_h)
                crop_x = random.randint(0, w - crop_w)

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
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)
                return sample

        return sample

    @staticmethod
    def _resize(resizer, sample, size, mode='short', context=None):
        self = resizer
        im = sample['image']
        target_size = size

        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(target_size)
            target_size_max = np.max(target_size)

            if mode == 'long':
                im_scale = min(target_size_min / im_size_min,
                               target_size_max / im_size_max)
            else:
                im_scale = max(target_size_min / im_size_min,
                               target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warn(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                    format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
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

        # apply gt_segm
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
class RandomSelect(BaseOperator):
    """
    Randomly choose a transformation between transforms1 and transforms2,
    and the probability of choosing transforms1 is p.

    The code is based on https://github.com/facebookresearch/detr/blob/main/datasets/transforms.py

    """

    def __init__(self, transforms1, transforms2, p=0.5):
        super(RandomSelect, self).__init__()
        self.transforms1 = Compose(transforms1)
        self.transforms2 = Compose(transforms2)
        self.p = p

    def apply(self, sample, context=None):
        if random.random() < self.p:
            return self.transforms1(sample)
        return self.transforms2(sample)


@register_op
class RandomShortSideResize(BaseOperator):
    def __init__(self,
                 short_side_sizes,
                 max_size=None,
                 interp=cv2.INTER_LINEAR,
                 random_interp=False):
        """
        Resize the image randomly according to the short side. If max_size is not None,
        the long side is scaled according to max_size. The whole process will be keep ratio.
        Args:
            short_side_sizes (list|tuple): Image target short side size.
            max_size (int): The size of the longest side of image after resize.
            interp (int): The interpolation method.
            random_interp (bool): Whether random select interpolation method.
        """
        super(RandomShortSideResize, self).__init__()

        assert isinstance(short_side_sizes,
                          Sequence), "short_side_sizes must be List or Tuple"

        self.short_side_sizes = short_side_sizes
        self.max_size = max_size
        self.interp = interp
        self.random_interp = random_interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

    def get_size_with_aspect_ratio(self, image_shape, size, max_size=None):
        h, w = image_shape
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def resize(self,
               sample,
               target_size,
               max_size=None,
               interp=cv2.INTER_LINEAR):
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        target_size = self.get_size_with_aspect_ratio(im.shape[:2], target_size,
                                                      max_size)
        im_scale_y, im_scale_x = target_size[1] / im.shape[0], target_size[
            0] / im.shape[1]

        sample['image'] = cv2.resize(im, target_size, interpolation=interp)
        sample['im_shape'] = np.asarray(target_size[::-1], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)
        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im.shape[:2],
                                                [im_scale_x, im_scale_y])
        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                target_size,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm, target_size, interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox.astype('float32')

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        target_size = random.choice(self.short_side_sizes)
        interp = random.choice(
            self.interps) if self.random_interp else self.interp

        return self.resize(sample, target_size, self.max_size, interp)


@register_op
class RandomSizeCrop(BaseOperator):
    """
    Cut the image randomly according to `min_size` and `max_size`
    """

    def __init__(self, min_size, max_size):
        super(RandomSizeCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

        from paddle.vision.transforms.functional import crop as paddle_crop
        self.paddle_crop = paddle_crop

    @staticmethod
    def get_crop_params(img_shape, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img_shape (list|tuple): Image's height and width.
            output_size (list|tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img_shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".
                format((th, tw), (h, w)))

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th + 1)
        j = random.randint(0, w - tw + 1)
        return i, j, th, tw

    def crop(self, sample, region):
        image_shape = sample['image'].shape[:2]
        sample['image'] = self.paddle_crop(sample['image'], *region)

        keep_index = None
        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], region)
            bbox = sample['gt_bbox'].reshape([-1, 2, 2])
            area = (bbox[:, 1, :] - bbox[:, 0, :]).prod(axis=1)
            keep_index = np.where(area > 0)[0]
            sample['gt_bbox'] = sample['gt_bbox'][keep_index] if len(
                keep_index) > 0 else np.zeros(
                    [0, 4], dtype=np.float32)
            sample['gt_class'] = sample['gt_class'][keep_index] if len(
                keep_index) > 0 else np.zeros(
                    [0, 1], dtype=np.float32)
            if 'gt_score' in sample:
                sample['gt_score'] = sample['gt_score'][keep_index] if len(
                    keep_index) > 0 else np.zeros(
                        [0, 1], dtype=np.float32)
            if 'is_crowd' in sample:
                sample['is_crowd'] = sample['is_crowd'][keep_index] if len(
                    keep_index) > 0 else np.zeros(
                        [0, 1], dtype=np.float32)

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], region,
                                                image_shape)
            if keep_index is not None:
                sample['gt_poly'] = sample['gt_poly'][keep_index]
        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            i, j, h, w = region
            sample['gt_segm'] = sample['gt_segm'][:, i:i + h, j:j + w]
            if keep_index is not None:
                sample['gt_segm'] = sample['gt_segm'][keep_index]

        return sample

    def apply_bbox(self, bbox, region):
        i, j, h, w = region
        region_size = np.asarray([w, h])
        crop_bbox = bbox - np.asarray([j, i, j, i])
        crop_bbox = np.minimum(crop_bbox.reshape([-1, 2, 2]), region_size)
        crop_bbox = crop_bbox.clip(min=0)
        return crop_bbox.reshape([-1, 4]).astype('float32')

    def apply_segm(self, segms, region, image_shape):
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

        i, j, h, w = region
        crop = [j, i, j + w, i + h]
        height, width = image_shape
        crop_segms = []
        for segm in segms:
            if is_poly(segm):
                import copy
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def apply(self, sample, context=None):
        h = random.randint(self.min_size,
                           min(sample['image'].shape[0], self.max_size))
        w = random.randint(self.min_size,
                           min(sample['image'].shape[1], self.max_size))

        region = self.get_crop_params(sample['image'].shape[:2], [h, w])
        return self.crop(sample, region)


@register_op
class WarpAffine(BaseOperator):
    def __init__(self,
                 keep_res=False,
                 pad=31,
                 input_h=512,
                 input_w=512,
                 scale=0.4,
                 shift=0.1):
        """WarpAffine
        Warp affine the image

        The code is based on https://github.com/xingyizhou/CenterNet/blob/master/src/lib/datasets/sample/ctdet.py


        """
        super(WarpAffine, self).__init__()
        self.keep_res = keep_res
        self.pad = pad
        self.input_h = input_h
        self.input_w = input_w
        self.scale = scale
        self.shift = shift

    def apply(self, sample, context=None):
        img = sample['image']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h, w = img.shape[:2]

        if self.keep_res:
            input_h = (h | self.pad) + 1
            input_w = (w | self.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
            c = np.array([w // 2, h // 2], dtype=np.float32)

        else:
            s = max(h, w) * 1.0
            input_h, input_w = self.input_h, self.input_w
            c = np.array([w / 2., h / 2.], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.resize(img, (w, h))
        inp = cv2.warpAffine(
            img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        sample['image'] = inp
        return sample


@register_op
class FlipWarpAffine(BaseOperator):
    def __init__(self,
                 keep_res=False,
                 pad=31,
                 input_h=512,
                 input_w=512,
                 not_rand_crop=False,
                 scale=0.4,
                 shift=0.1,
                 flip=0.5,
                 is_scale=True,
                 use_random=True):
        """FlipWarpAffine
        1. Random Crop
        2. Flip the image horizontal
        3. Warp affine the image 
        """
        super(FlipWarpAffine, self).__init__()
        self.keep_res = keep_res
        self.pad = pad
        self.input_h = input_h
        self.input_w = input_w
        self.not_rand_crop = not_rand_crop
        self.scale = scale
        self.shift = shift
        self.flip = flip
        self.is_scale = is_scale
        self.use_random = use_random

    def apply(self, sample, context=None):
        img = sample['image']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h, w = img.shape[:2]

        if self.keep_res:
            input_h = (h | self.pad) + 1
            input_w = (w | self.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
            c = np.array([w // 2, h // 2], dtype=np.float32)

        else:
            s = max(h, w) * 1.0
            input_h, input_w = self.input_h, self.input_w
            c = np.array([w / 2., h / 2.], dtype=np.float32)

        if self.use_random:
            gt_bbox = sample['gt_bbox']
            if not self.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = get_border(128, w)
                h_border = get_border(128, h)
                c[0] = np.random.randint(low=w_border, high=w - w_border)
                c[1] = np.random.randint(low=h_border, high=h - h_border)
            else:
                sf = self.scale
                cf = self.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.flip:
                img = img[:, ::-1, :]
                c[0] = w - c[0] - 1
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                gt_bbox[:, 0] = w - oldx2 - 1
                gt_bbox[:, 2] = w - oldx1 - 1
            sample['gt_bbox'] = gt_bbox

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        if not self.use_random:
            img = cv2.resize(img, (w, h))
        inp = cv2.warpAffine(
            img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        if self.is_scale:
            inp = (inp.astype(np.float32) / 255.)
        sample['image'] = inp
        sample['center'] = c
        sample['scale'] = s
        return sample


@register_op
class CenterRandColor(BaseOperator):
    """Random color for CenterNet series models.
    Args:
        saturation (float): saturation settings.
        contrast (float): contrast settings.
        brightness (float): brightness settings.
    """

    def __init__(self, saturation=0.4, contrast=0.4, brightness=0.4):
        super(CenterRandColor, self).__init__()
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness

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
class Mosaic(BaseOperator):
    """ Mosaic operator for image and gt_bboxes
    The code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/mosaicdetection.py

    1. get mosaic coords
    2. clip bbox and get mosaic_labels
    3. random_affine augment
    4. Mixup augment as copypaste (optinal), not used in tiny/nano

    Args:
        prob (float): probability of using Mosaic, 1.0 as default
        input_dim (list[int]): input shape
        degrees (list[2]): the rotate range to apply, transform range is [min, max]
        translate (list[2]): the translate range to apply, transform range is [min, max]
        scale (list[2]): the scale range to apply, transform range is [min, max]
        shear (list[2]): the shear range to apply, transform range is [min, max]
        enable_mixup (bool): whether to enable Mixup or not
        mixup_prob (float): probability of using Mixup, 1.0 as default
        mixup_scale (list[int]): scale range of Mixup
        remove_outside_box (bool): whether remove outside boxes, False as
            default in COCO dataset, True in MOT dataset
    """

    def __init__(self,
                 prob=1.0,
                 input_dim=[640, 640],
                 degrees=[-10, 10],
                 translate=[-0.1, 0.1],
                 scale=[0.1, 2],
                 shear=[-2, 2],
                 enable_mixup=True,
                 mixup_prob=1.0,
                 mixup_scale=[0.5, 1.5],
                 remove_outside_box=False):
        super(Mosaic, self).__init__()
        self.prob = prob
        if isinstance(input_dim, Integral):
            input_dim = [input_dim, input_dim]
        self.input_dim = input_dim
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.enable_mixup = enable_mixup
        self.mixup_prob = mixup_prob
        self.mixup_scale = mixup_scale
        self.remove_outside_box = remove_outside_box

    def get_mosaic_coords(self, mosaic_idx, xc, yc, w, h, input_h, input_w):
        # (x1, y1, x2, y2) means coords in large image,
        # small_coords means coords in small image in mosaic aug.
        if mosaic_idx == 0:
            # top left
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coords = w - (x2 - x1), h - (y2 - y1), w, h
        elif mosaic_idx == 1:
            # top right
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coords = 0, h - (y2 - y1), min(w, x2 - x1), h
        elif mosaic_idx == 2:
            # bottom left
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coords = w - (x2 - x1), 0, w, min(y2 - y1, h)
        elif mosaic_idx == 3:
            # bottom right
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2,
                                                                   yc + h)
            small_coords = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

        return (x1, y1, x2, y2), small_coords

    def random_affine_augment(self,
                              img,
                              labels=[],
                              input_dim=[640, 640],
                              degrees=[-10, 10],
                              scales=[0.1, 2],
                              shears=[-2, 2],
                              translates=[-0.1, 0.1]):
        # random rotation and scale
        degree = random.uniform(degrees[0], degrees[1])
        scale = random.uniform(scales[0], scales[1])
        assert scale > 0, "Argument scale should be positive."
        R = cv2.getRotationMatrix2D(angle=degree, center=(0, 0), scale=scale)
        M = np.ones([2, 3])

        # random shear
        shear = random.uniform(shears[0], shears[1])
        shear_x = math.tan(shear * math.pi / 180)
        shear_y = math.tan(shear * math.pi / 180)
        M[0] = R[0] + shear_y * R[1]
        M[1] = R[1] + shear_x * R[0]

        # random translation
        translate = random.uniform(translates[0], translates[1])
        translation_x = translate * input_dim[0]
        translation_y = translate * input_dim[1]
        M[0, 2] = translation_x
        M[1, 2] = translation_y

        # warpAffine
        img = cv2.warpAffine(
            img, M, dsize=tuple(input_dim), borderValue=(114, 114, 114))

        num_gts = len(labels)
        if num_gts > 0:
            # warp corner points
            corner_points = np.ones((4 * num_gts, 3))
            corner_points[:, :2] = labels[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                4 * num_gts, 2)  # x1y1, x2y2, x1y2, x2y1
            # apply affine transform
            corner_points = corner_points @M.T
            corner_points = corner_points.reshape(num_gts, 8)

            # create new boxes
            corner_xs = corner_points[:, 0::2]
            corner_ys = corner_points[:, 1::2]
            new_bboxes = np.concatenate((corner_xs.min(1), corner_ys.min(1),
                                         corner_xs.max(1), corner_ys.max(1)))
            new_bboxes = new_bboxes.reshape(4, num_gts).T

            # clip boxes
            new_bboxes[:, 0::2] = np.clip(new_bboxes[:, 0::2], 0, input_dim[0])
            new_bboxes[:, 1::2] = np.clip(new_bboxes[:, 1::2], 0, input_dim[1])
            labels[:, :4] = new_bboxes

        return img, labels

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(
            sample) == 5, "Mosaic needs 5 samples, 4 for mosaic and 1 for mixup."
        if np.random.uniform(0., 1.) > self.prob:
            return sample[0]

        mosaic_gt_bbox, mosaic_gt_class, mosaic_is_crowd = [], [], []
        input_h, input_w = self.input_dim
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
        mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)

        # 1. get mosaic coords
        for mosaic_idx, sp in enumerate(sample[:4]):
            img = sp['image']
            gt_bbox = sp['gt_bbox']
            h0, w0 = img.shape[:2]
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)),
                interpolation=cv2.INTER_LINEAR)
            (h, w, c) = img.shape[:3]

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (
                s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coords(
                    mosaic_idx, xc, yc, w, h, input_h, input_w)

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            # Normalized xywh to pixel xyxy format
            _gt_bbox = gt_bbox.copy()
            if len(gt_bbox) > 0:
                _gt_bbox[:, 0] = scale * gt_bbox[:, 0] + padw
                _gt_bbox[:, 1] = scale * gt_bbox[:, 1] + padh
                _gt_bbox[:, 2] = scale * gt_bbox[:, 2] + padw
                _gt_bbox[:, 3] = scale * gt_bbox[:, 3] + padh

            mosaic_gt_bbox.append(_gt_bbox)
            mosaic_gt_class.append(sp['gt_class'])
            mosaic_is_crowd.append(sp['is_crowd'])

        # 2. clip bbox and get mosaic_labels([gt_bbox, gt_class, is_crowd])
        if len(mosaic_gt_bbox):
            mosaic_gt_bbox = np.concatenate(mosaic_gt_bbox, 0)
            mosaic_gt_class = np.concatenate(mosaic_gt_class, 0)
            mosaic_is_crowd = np.concatenate(mosaic_is_crowd, 0)
            mosaic_labels = np.concatenate([
                mosaic_gt_bbox, mosaic_gt_class.astype(mosaic_gt_bbox.dtype),
                mosaic_is_crowd.astype(mosaic_gt_bbox.dtype)
            ], 1)
            if self.remove_outside_box:
                # for MOT dataset
                flag1 = mosaic_gt_bbox[:, 0] < 2 * input_w
                flag2 = mosaic_gt_bbox[:, 2] > 0
                flag3 = mosaic_gt_bbox[:, 1] < 2 * input_h
                flag4 = mosaic_gt_bbox[:, 3] > 0
                flag_all = flag1 * flag2 * flag3 * flag4
                mosaic_labels = mosaic_labels[flag_all]
            else:
                mosaic_labels[:, 0] = np.clip(mosaic_labels[:, 0], 0,
                                              2 * input_w)
                mosaic_labels[:, 1] = np.clip(mosaic_labels[:, 1], 0,
                                              2 * input_h)
                mosaic_labels[:, 2] = np.clip(mosaic_labels[:, 2], 0,
                                              2 * input_w)
                mosaic_labels[:, 3] = np.clip(mosaic_labels[:, 3], 0,
                                              2 * input_h)
        else:
            mosaic_labels = np.zeros((1, 6))

        # 3. random_affine augment
        mosaic_img, mosaic_labels = self.random_affine_augment(
            mosaic_img,
            mosaic_labels,
            input_dim=self.input_dim,
            degrees=self.degrees,
            translates=self.translate,
            scales=self.scale,
            shears=self.shear)

        # 4. Mixup augment as copypaste, https://arxiv.org/abs/2012.07177
        # optinal, not used(enable_mixup=False) in tiny/nano
        if (self.enable_mixup and not len(mosaic_labels) == 0 and
                random.random() < self.mixup_prob):
            sample_mixup = sample[4]
            mixup_img = sample_mixup['image']
            cp_labels = np.concatenate([
                sample_mixup['gt_bbox'],
                sample_mixup['gt_class'].astype(mosaic_labels.dtype),
                sample_mixup['is_crowd'].astype(mosaic_labels.dtype)
            ], 1)
            mosaic_img, mosaic_labels = self.mixup_augment(
                mosaic_img, mosaic_labels, self.input_dim, cp_labels, mixup_img)

        sample0 = sample[0]
        sample0['image'] = mosaic_img.astype(np.uint8)  # can not be float32
        sample0['h'] = float(mosaic_img.shape[0])
        sample0['w'] = float(mosaic_img.shape[1])
        sample0['im_shape'][0] = sample0['h']
        sample0['im_shape'][1] = sample0['w']
        sample0['gt_bbox'] = mosaic_labels[:, :4].astype(np.float32)
        sample0['gt_class'] = mosaic_labels[:, 4:5].astype(np.float32)
        sample0['is_crowd'] = mosaic_labels[:, 5:6].astype(np.float32)
        return sample0

    def mixup_augment(self, origin_img, origin_labels, input_dim, cp_labels,
                      img):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        if len(img.shape) == 3:
            cp_img = np.ones(
                (input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0],
                             input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img, (int(img.shape[1] * cp_scale_ratio),
                  int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR)

        cp_img[:int(img.shape[0] * cp_scale_ratio), :int(img.shape[
            1] * cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(cp_img, (int(cp_img.shape[1] * jit_factor),
                                     int(cp_img.shape[0] * jit_factor)))
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3),
            dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h, x_offset:
                                        x_offset + target_w]

        # adjust boxes
        cp_bboxes_origin_np = cp_labels[:, :4].copy()
        cp_bboxes_origin_np[:, 0::2] = np.clip(cp_bboxes_origin_np[:, 0::2] *
                                               cp_scale_ratio, 0, origin_w)
        cp_bboxes_origin_np[:, 1::2] = np.clip(cp_bboxes_origin_np[:, 1::2] *
                                               cp_scale_ratio, 0, origin_h)

        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1])
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        if self.remove_outside_box:
            # for MOT dataset
            cp_bboxes_transformed_np[:, 0::2] -= x_offset
            cp_bboxes_transformed_np[:, 1::2] -= y_offset
        else:
            cp_bboxes_transformed_np[:, 0::2] = np.clip(
                cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
            cp_bboxes_transformed_np[:, 1::2] = np.clip(
                cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

        cls_labels = cp_labels[:, 4:5].copy()
        crd_labels = cp_labels[:, 5:6].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels, crd_labels))
        if self.remove_outside_box:
            labels = labels[labels[:, 0] < target_w]
            labels = labels[labels[:, 2] > 0]
            labels = labels[labels[:, 1] < target_h]
            labels = labels[labels[:, 3] > 0]

        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(
            np.float32)

        return origin_img.astype(np.uint8), origin_labels


@register_op
class PadResize(BaseOperator):
    """ PadResize for image and gt_bbbox

    Args:
        target_size (list[int]): input shape
        fill_value (float): pixel value of padded image
    """

    def __init__(self, target_size, fill_value=114):
        super(PadResize, self).__init__()
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.fill_value = fill_value

    def _resize(self, img, bboxes, labels):
        ratio = min(self.target_size[0] / img.shape[0],
                    self.target_size[1] / img.shape[1])
        w, h = int(img.shape[1] * ratio), int(img.shape[0] * ratio)
        resized_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        if len(bboxes) > 0:
            bboxes *= ratio
            mask = np.minimum(bboxes[:, 2] - bboxes[:, 0],
                              bboxes[:, 3] - bboxes[:, 1]) > 1
            bboxes = bboxes[mask]
            labels = labels[mask]
        return resized_img, bboxes, labels

    def _pad(self, img):
        h, w, _ = img.shape
        if h == self.target_size[0] and w == self.target_size[1]:
            return img
        padded_img = np.full(
            (self.target_size[0], self.target_size[1], 3),
            self.fill_value,
            dtype=np.uint8)
        padded_img[:h, :w] = img
        return padded_img

    def apply(self, sample, context=None):
        image = sample['image']
        bboxes = sample['gt_bbox']
        labels = sample['gt_class']
        image, bboxes, labels = self._resize(image, bboxes, labels)
        sample['image'] = self._pad(image).astype(np.float32)
        sample['gt_bbox'] = bboxes
        sample['gt_class'] = labels
        return sample
