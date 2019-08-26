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

import uuid
import logging
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from ppdet.core.workspace import serializable

from .op_helper import (satisfy_sample_constraint, filter_and_process,
                        generate_sample_bbox, clip_bbox)

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
    def __init__(self, to_rgb=True, with_mixup=False):
        """ Transform the image data to numpy format.

        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
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
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)
        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context)
        return sample


@register_op
class ResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True):
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
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
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

            resize_w = np.round(im_scale_x * float(im_shape[1]))
            resize_h = np.round(im_scale_y * float(im_shape[0]))

            sample['im_info'] = np.array(
                [resize_h, resize_w, im_scale], dtype=np.float32)
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
            im = Image.fromarray(im)
            im = im.resize((resize_w, resize_h), self.interp)
            im = np.array(im)

        sample['image'] = im
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
                rle = mask_util.frPyObjects([rle], height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1, :]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        def is_poly(segm):
            assert isinstance(segm, (list, dict)), \
                "Invalid segm type: {}".format(type(segm))
            return isinstance(segm, list)

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
            if gt_bbox.shape[0] != 0 and (gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                m = "{}: invalid box, x2 should be greater than x1".format(self)
                raise BboxError(m)
            sample['gt_bbox'] = gt_bbox
            if self.is_mask_flip and len(sample['gt_poly']) != 0:
                sample['gt_poly'] = self.flip_segms(sample['gt_poly'], height,
                                                    width)
            sample['flipped'] = True
            sample['image'] = im
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
        im = sample['image']
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
        sample['image'] = im
        return sample


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
            1. Scale the image weight and height.
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
                gt_bbox, gt_class, _ = filter_and_process(expand_bbox, gt_bbox,
                                                          gt_class)
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
            avoid_no_bbox (bool): whether to to avoid the 
                                  situation where the box does not appear.
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
        """
        super(CropImage, self).__init__()
        self.batch_sampler = batch_sampler
        self.satisfy_all = satisfy_all
        self.avoid_no_bbox = avoid_no_bbox

    def __call__(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image weight and height.
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
                filter_and_process(sample_bbox, gt_bbox, gt_class, gt_score)
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
        assert 'image' in sample, "image data not found"
        im = sample['image']
        if self.channel_first:
            im = np.swapaxes(im, 1, 2)
            im = np.swapaxes(im, 1, 0)
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        sample['image'] = im
        return sample


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
        gt_bbox1 = sample['gt_bbox']
        gt_bbox2 = sample['mixup']['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)

        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        sample['image'] = im
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample.pop('mixup')
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
