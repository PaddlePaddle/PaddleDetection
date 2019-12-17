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
from .op_helper import jaccard_overlap

logger = logging.getLogger(__name__)


def build_post_map(coarsest_stride=1,
                   is_padding=False,
                   random_shapes=[],
                   anchors=[],
                   anchor_masks=[],
                   downsample_ratios=[],
                   num_classes=20,
                   multi_scales=[],
                   use_padded_im_info=False,
                   enable_multiscale_test=False,
                   num_scale=1):
    """
    Build a mapper for post-processing batches

    Args:
        config (dict of parameters):
          {
            coarsest_stride (int): stride of the coarsest FPN level
            is_padding (bool): whether to padding in minibatch
            random_shapes (list of int): resize to image to random shapes, 
                [] for not resize.
            anchors (list of list of int): height and width of yolo anchors.
            anchor_masks (list of list of int): anchor mask for yolo loss layers.
            downsample_ratios (list of int): downsample ratio from input to yolo
                loss layers.
            num_classes (int): class number of dataset
            multi_scales (list of int): resize image by random scales, 
                [] for not resize.
            use_padded_im_info (bool): whether to update im_info after padding
            enable_multiscale_test (bool): whether to use multiscale test.
            num_scale (int) : the number of scales for multiscale test.
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

    def padding_multiscale_test(batch_data):
        if len(batch_data) != 1:
            raise NotImplementedError(
                "Batch size must be 1 when using multiscale test, but now batch size is {}".
                format(len(batch_data)))
        if coarsest_stride > 1:
            padding_batch = []
            padding_images = []
            data = batch_data[0]
            for i, input in enumerate(data):
                if i < num_scale:
                    im_c, im_h, im_w = input.shape
                    max_h = int(
                        np.ceil(im_h / coarsest_stride) * coarsest_stride)
                    max_w = int(
                        np.ceil(im_w / coarsest_stride) * coarsest_stride)
                    padding_im = np.zeros(
                        (im_c, max_h, max_w), dtype=np.float32)
                    padding_im[:, :im_h, :im_w] = input
                    data[num_scale][3 * i:3 * i + 2] = [max_h, max_w]
                    padding_batch.append(padding_im)
                else:
                    padding_batch.append(input)
            return [tuple(padding_batch)]
        # no need to padding
        return batch_data

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

    def gtbbox2target(batch_data):
        assert len(anchor_masks) == len(downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = batch_data[0][0].shape[1:3]
        an_hw = np.array(anchors) / np.array([[w, h]])
        new_batch = []
        for data in batch_data:
            im, gt_bbox, gt_class, gt_score = data
            new_data = [im, gt_bbox, gt_class, gt_score]
            for mask, downsample_ratio in zip(anchor_masks, downsample_ratios):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    # gtbox should be regresed in this layes if best match 
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)
                        gi = int(gx * grid_w)
                        gj = int(gy * grid_h)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(gw * w /
                                                           anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(gh * h /
                                                           anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.
                new_data.append(target)
            new_batch.append(new_data)
        return new_batch

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
            if len(downsample_ratios):
                batch_data = gtbbox2target(batch_data)
            if len(multi_scales) > 0:
                batch_data = multi_scale_resize(batch_data)
            if enable_multiscale_test:
                batch_data = padding_multiscale_test(batch_data)
        except Exception as e:
            errmsg = "post-process failed with error: " + str(e)
            logger.warn(errmsg)
            raise e

        return batch_data

    return _mapper
