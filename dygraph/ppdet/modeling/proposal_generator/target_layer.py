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

import paddle

from ppdet.core.workspace import register, serializable

from .target import rpn_anchor_target, generate_proposal_target, generate_mask_target


@register
@serializable
class RPNTargetAssign(object):
    def __init__(self,
                 batch_size_per_im=256,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 use_random=True):
        super(RPNTargetAssign, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.use_random = use_random

    def __call__(self, inputs, anchors):
        """
        inputs: ground-truth instances.
        anchor_box (Tensor): [num_anchors, 4], num_anchors are all anchors in all feature maps.
        """
        gt_boxes = inputs['gt_bbox']
        batch_size = gt_boxes.shape[0]
        tgt_labels, tgt_bboxes, tgt_deltas = rpn_anchor_target(
            anchors, gt_boxes, self.batch_size_per_im, self.positive_overlap,
            self.negative_overlap, self.fg_fraction, self.use_random,
            batch_size)
        norm = self.batch_size_per_im * batch_size

        return tgt_labels, tgt_bboxes, tgt_deltas, norm


@register
class BBoxAssigner(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh=.5,
                 use_random=True,
                 is_cls_agnostic=False,
                 cascade_iou=[0.5, 0.6, 0.7],
                 num_classes=80):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.use_random = use_random
        self.is_cls_agnostic = is_cls_agnostic
        self.cascade_iou = cascade_iou
        self.num_classes = num_classes

    def __call__(self,
                 rpn_rois,
                 rpn_rois_num,
                 inputs,
                 stage=0,
                 is_cascade=False):
        is_cascade = True if stage > 0 else False
        gt_classes = inputs['gt_class']
        gt_boxes = inputs['gt_bbox']
        # rois, tgt_labels, tgt_bboxes, tgt_gt_inds
        # new_rois_num
        outs = generate_proposal_target(
            rpn_rois, gt_classes, gt_boxes, self.batch_size_per_im,
            self.fg_fraction, self.fg_thresh, self.bg_thresh, self.num_classes,
            self.use_random, is_cascade, self.cascade_iou[stage])
        rois = outs[0]
        rois_num = outs[-1]
        # tgt_labels, tgt_bboxes, tgt_gt_inds
        targets = outs[1:4]
        return rois, rois_num, targets


@register
@serializable
class MaskAssigner(object):
    __shared__ = ['num_classes', 'mask_resolution']

    def __init__(self, num_classes=80, mask_resolution=14):
        super(MaskAssigner, self).__init__()
        self.num_classes = num_classes
        self.mask_resolution = mask_resolution

    def __call__(self, rois, tgt_labels, tgt_gt_inds, inputs):
        gt_segms = inputs['gt_poly']

        outs = generate_mask_target(gt_segms, rois, tgt_labels, tgt_gt_inds,
                                    self.num_classes, self.mask_resolution)

        # mask_rois, mask_rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights
        return outs
