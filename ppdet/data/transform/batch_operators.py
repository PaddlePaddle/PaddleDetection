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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
import cv2
import numpy as np

from .operators import register_op, BaseOperator

logger = logging.getLogger(__name__)


@register_op
class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.

    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
        return samples


@register_op
class RandomShape(BaseOperator):
    """
    Randomly reshape a batch

    Args:
        sizes (list): list of int, random choose a size from these
    """

    def __init__(self, sizes=[]):
        super(RandomShape, self).__init__()
        self.sizes = sizes

    def __call__(self, samples, context=None):
        # For YOLO: gt_bbox is normalized, is scale invariant.
        shape = np.random.choice(self.sizes)
        h, w = samples[0]['image'].shape[1:3]
        scale_x = float(shape) / w
        scale_y = float(shape) / h
        out_samples = []
        for data in samples:
            im = cv2.resize(
                data['image'],
                None,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_NEAREST)
            out_samples.append(data)
        return out_samples


@register_op
class PadMultiScaleTest(BaseOperator):
    """
    Pad the image so they can be divisible by a stride for multi-scale testing.
 
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadMultiScaleTest, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples

        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        if len(samples) != 1:
            raise ValueError("Batch size must be 1 when using multiscale test, "
                             "but now batch size is {}".format(len(samples)))
        out_samples = []
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im_c, im_h, im_w = im.shape
                    max_h = int(
                        np.ceil(im_h / coarsest_stride) * coarsest_stride)
                    max_w = int(
                        np.ceil(im_w / coarsest_stride) * coarsest_stride)
                    padding_im = np.zeros(
                        (im_c, max_h, max_w), dtype=np.float32)

                    padding_im[:, :im_h, :im_w] = im
                    sample[k] = padding_im
                    info_name = 'im_info' if k == 'image' else 'im_info_' + k
                    # update im_info
                    sample[info_name][:2] = [max_h, max_w]
            out_samples.append(sample)
        if not batch_input:
            out_samples = out_samples[0]
        return out_samples
