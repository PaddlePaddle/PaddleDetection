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

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_post_map(coarsest_stride=1,
                   is_padding=False,
                   random_shapes=[],
                   multi_scales=[],
                   use_padded_im_info=False):
    """
    Build a mapper for post-processing batches

    Args:
        config (dict of parameters):
          {
            coarsest_stride (int): stride of the coarsest FPN level
            is_padding (bool): whether to padding in minibatch
            random_shapes: (list of int): resize to image to random
                                          shapes, [] for not resize.
            multi_scales: (list of int): resize image by random
                                          scales, [] for not resize.
          }
    Returns:
        a mapper function which accept one argument 'batch' and
        return the processed result
    """

    def padding_minibatch(batch_data):
        if len(batch_data) == 1 and coarsest_stride == 1:
            return batch_data
        max_shape = np.array([data[0].shape for data in batch_data]).max(axis=0)
        if coarsest_stride > 1:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)
        padding_batch = []
        for data in batch_data:
            im_c, im_h, im_w = data[0].shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = data[0]
            if use_padded_im_info:
                data[1][:2] = max_shape[1:3]
            padding_batch.append((padding_im, ) + data[1:])
        return padding_batch

    def random_shape(batch_data):
        # For YOLO: gt_bbox is normalized, is scale invariant.
        shape = np.random.choice(random_shapes)
        scaled_batch = []
        h, w = batch_data[0][0].shape[1:3]
        scale_x = float(shape) / w
        scale_y = float(shape) / h
        for data in batch_data:
            im = cv2.resize(
                data[0].transpose((1, 2, 0)),
                None,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_NEAREST)
            scaled_batch.append((im.transpose(2, 0, 1), ) + data[1:])
        return scaled_batch

    def multi_scale_resize(batch_data):
        # For RCNN: image shape in record in im_info.
        scale = np.random.choice(multi_scales)
        scaled_batch = []
        for data in batch_data:
            im = cv2.resize(
                data[0].transpose((1, 2, 0)),
                None,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST)
            im_info = [im.shape[:2], scale]
            scaled_batch.append((im.transpose(2, 0, 1), im_info) + data[2:])
        return scaled_batch

    def _mapper(batch_data):
        try:
            if is_padding:
                batch_data = padding_minibatch(batch_data)
            if len(random_shapes) > 0:
                batch_data = random_shape(batch_data)
            if len(multi_scales) > 0:
                batch_data = multi_scale_resize(batch_data)
        except Exception as e:
            errmsg = "post-process failed with error: " + str(e)
            logger.warn(errmsg)
            raise e

        return batch_data

    return _mapper
