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
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from .operators import BaseOperator, register_op

logger = logging.getLogger(__name__)


@register_op
class ArrangeRCNN(BaseOperator):
    """
    Transform dict to tuple format needed for training.

    Args:
        is_mask (bool): whether to use include mask data
    """

    def __init__(self, is_mask=False):
        super(ArrangeRCNN, self).__init__()
        self.is_mask = is_mask
        assert isinstance(self.is_mask, bool), "wrong type for is_mask"

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing following items
                (image, im_info, im_id, gt_bbox, gt_class, is_crowd, gt_masks)
        """
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        keys = list(sample.keys())
        if 'is_crowd' in keys:
            is_crowd = sample['is_crowd']
        else:
            raise KeyError("The dataset doesn't have 'is_crowd' key.")
        if 'im_info' in keys:
            im_info = sample['im_info']
        else:
            raise KeyError("The dataset doesn't have 'im_info' key.")
        im_id = sample['im_id']

        outs = (im, im_info, im_id, gt_bbox, gt_class, is_crowd)
        gt_masks = []
        if self.is_mask and len(sample['gt_poly']) != 0 \
                and 'is_crowd' in keys:
            valid = True
            segms = sample['gt_poly']
            assert len(segms) == is_crowd.shape[0]
            for i in range(len(sample['gt_poly'])):
                segm, iscrowd = segms[i], is_crowd[i]
                gt_segm = []
                if iscrowd:
                    gt_segm.append([[0, 0]])
                else:
                    for poly in segm:
                        if len(poly) == 0:
                            valid = False
                            break
                        gt_segm.append(np.array(poly).reshape(-1, 2))
                if (not valid) or len(gt_segm) == 0:
                    break
                gt_masks.append(gt_segm)
            outs = outs + (gt_masks, )
        return outs


@register_op
class ArrangeTestRCNN(BaseOperator):
    """
    Transform dict to the tuple format needed for training.
    """

    def __init__(self):
        super(ArrangeTestRCNN, self).__init__()

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing the following items:
                    (image, im_info, im_id)
        """
        im = sample['image']
        keys = list(sample.keys())
        if 'im_info' in keys:
            im_info = sample['im_info']
        else:
            raise KeyError("The dataset doesn't have 'im_info' key.")
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        # For rcnn models in eval and infer stage, original image size
        # is needed to clip the bounding boxes. And box clip op in
        # bbox prediction needs im_info as input in format of [N, 3],
        # so im_shape is appended by 1 to match dimension.
        im_shape = np.array((h, w, 1), dtype=np.float32)
        outs = (im, im_info, im_id, im_shape)
        return outs


@register_op
class ArrangeSSD(BaseOperator):
    """
    Transform dict to tuple format needed for training.
    """

    def __init__(self):
        super(ArrangeSSD, self).__init__()

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing the following items:
                    (image, gt_bbox, gt_class, difficult)
        """
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        outs = (im, gt_bbox, gt_class)
        return outs

@register_op
class ArrangeEvalSSD(BaseOperator):
    """
    Transform dict to tuple format needed for training.
    """

    def __init__(self):
        super(ArrangeEvalSSD, self).__init__()

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing the following items: (image)
        """
        im = sample['image']
        if len(sample['gt_bbox']) != len(sample['gt_class']):
            raise ValueError("gt num mismatch: bbox and class.")
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        im_shape = np.array((h, w))
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        difficult = sample['difficult']
        outs = (im, im_shape, im_id, gt_bbox, gt_class, difficult)

        return outs

@register_op
class ArrangeTestSSD(BaseOperator):
    """
    Transform dict to tuple format needed for training.

    Args:
        is_mask (bool): whether to use include mask data
    """

    def __init__(self):
        super(ArrangeTestSSD, self).__init__()

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing the following items: (image)
        """
        im = sample['image']
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        im_shape = np.array((h, w))
        outs = (im, im_id, im_shape)
        return outs


@register_op
class ArrangeYOLO(BaseOperator):
    """
    Transform dict to the tuple format needed for training.
    """

    def __init__(self):
        super(ArrangeYOLO, self).__init__()

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing the following items:
                (image, gt_bbox, gt_class, gt_score,
                 is_crowd, im_info, gt_masks)
        """
        im = sample['image']
        if len(sample['gt_bbox']) != len(sample['gt_class']):
            raise ValueError("gt num mismatch: bbox and class.")
        if len(sample['gt_bbox']) != len(sample['gt_score']):
            raise ValueError("gt num mismatch: bbox and score.")
        gt_bbox = np.zeros((50, 4), dtype=im.dtype)
        gt_class = np.zeros((50, ), dtype=np.int32)
        gt_score = np.zeros((50, ), dtype=im.dtype)
        gt_num = min(50, len(sample['gt_bbox']))
        if gt_num > 0:
            gt_bbox[:gt_num, :] = sample['gt_bbox'][:gt_num, :]
            gt_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            gt_score[:gt_num] = sample['gt_score'][:gt_num, 0]
        # parse [x1, y1, x2, y2] to [x, y, w, h]
        gt_bbox[:, 2:4] = gt_bbox[:, 2:4] - gt_bbox[:, :2]
        gt_bbox[:, :2] = gt_bbox[:, :2] + gt_bbox[:, 2:4] / 2.
        outs = (im, gt_bbox, gt_class, gt_score)
        return outs


@register_op
class ArrangeEvalYOLO(BaseOperator):
    """
    Transform dict to the tuple format needed for evaluation.
    """

    def __init__(self):
        super(ArrangeEvalYOLO, self).__init__()

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing the following items:
                (image, im_shape, im_id, gt_bbox, gt_class,
                 difficult)
        """
        im = sample['image']
        if len(sample['gt_bbox']) != len(sample['gt_class']):
            raise ValueError("gt num mismatch: bbox and class.")
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        im_shape = np.array((h, w))
        gt_bbox = np.zeros((50, 4), dtype=im.dtype)
        gt_class = np.zeros((50, ), dtype=np.int32)
        difficult = np.zeros((50, ), dtype=np.int32)
        gt_num = min(50, len(sample['gt_bbox']))
        if gt_num > 0:
            gt_bbox[:gt_num, :] = sample['gt_bbox'][:gt_num, :]
            gt_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            difficult[:gt_num] = sample['difficult'][:gt_num, 0]
        outs = (im, im_shape, im_id, gt_bbox, gt_class, difficult)
        return outs


@register_op
class ArrangeTestYOLO(BaseOperator):
    """
    Transform dict to the tuple format needed for inference.
    """

    def __init__(self):
        super(ArrangeTestYOLO, self).__init__()

    def __call__(self, sample, context=None):
        """
        Args:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Returns:
            sample: a tuple containing the following items:
                (image, gt_bbox, gt_class, gt_score, is_crowd,
                 im_info, gt_masks)
        """
        im = sample['image']
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        im_shape = np.array((h, w))
        outs = (im, im_shape, im_id)
        return outs
