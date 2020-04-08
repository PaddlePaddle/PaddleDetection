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

from paddle import fluid

from ppdet.core.workspace import register
from ppdet.modeling.ops import BBoxAssigner, MaskAssigner

__all__ = [
    'BBoxAssigner',
    'MaskAssigner',
    'CascadeBBoxAssigner',
]


@register
class CascadeBBoxAssigner(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=[0.5, 0.6, 0.7],
                 bg_thresh_hi=[0.5, 0.6, 0.7],
                 bg_thresh_lo=[0., 0., 0.],
                 bbox_reg_weights=[10, 20, 30],
                 shuffle_before_sample=True,
                 num_classes=81,
                 class_aware=False):
        super(CascadeBBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.class_nums = num_classes
        self.use_random = shuffle_before_sample
        self.class_aware = class_aware

    def __call__(self, input_rois, feed_vars, curr_stage):

        curr_bbox_reg_w = [
            1. / self.bbox_reg_weights[curr_stage],
            1. / self.bbox_reg_weights[curr_stage],
            2. / self.bbox_reg_weights[curr_stage],
            2. / self.bbox_reg_weights[curr_stage],
        ]
        outs = fluid.layers.generate_proposal_labels(
            rpn_rois=input_rois,
            gt_classes=feed_vars['gt_class'],
            is_crowd=feed_vars['is_crowd'],
            gt_boxes=feed_vars['gt_bbox'],
            im_info=feed_vars['im_info'],
            batch_size_per_im=self.batch_size_per_im,
            fg_thresh=self.fg_thresh[curr_stage],
            bg_thresh_hi=self.bg_thresh_hi[curr_stage],
            bg_thresh_lo=self.bg_thresh_lo[curr_stage],
            bbox_reg_weights=curr_bbox_reg_w,
            use_random=self.use_random,
            class_nums=self.class_nums if self.class_aware else 2,
            is_cls_agnostic=not self.class_aware,
            is_cascade_rcnn=True
            if curr_stage > 0 and not self.class_aware else False)
        return outs
