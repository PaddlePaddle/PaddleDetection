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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['MaskRCNN']


@register
class MaskRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'anchor',
        'proposal',
        'mask',
        'backbone',
        'neck',
        'rpn_head',
        'bbox_head',
        'mask_head',
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 anchor,
                 proposal,
                 mask,
                 backbone,
                 rpn_head,
                 bbox_head,
                 mask_head,
                 bbox_post_process,
                 mask_post_process,
                 neck=None):
        super(MaskRCNN, self).__init__()
        self.anchor = anchor
        self.proposal = proposal
        self.mask = mask
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head
        self.bbox_post_process = bbox_post_process
        self.mask_post_process = mask_post_process

    def model_arch(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        spatial_scale = 1. / 16

        # Neck
        if self.neck is not None:
            body_feats, spatial_scale = self.neck(body_feats)

        # RPN
        # rpn_head returns two list: rpn_feat, rpn_head_out 
        # each element in rpn_feats contains rpn feature on each level,
        # and the length is 1 when the neck is not applied.
        # each element in rpn_head_out contains (rpn_rois_score, rpn_rois_delta)
        rpn_feat, self.rpn_head_out = self.rpn_head(self.inputs, body_feats)

        # Anchor
        # anchor_out returns a list,
        # each element contains (anchor, anchor_var)
        self.anchor_out = self.anchor(rpn_feat)

        # Proposal RoI 
        # compute targets here when training
        rois = self.proposal(self.inputs, self.rpn_head_out, self.anchor_out)
        # BBox Head
        bbox_feat, self.bbox_head_out, self.bbox_head_feat_func = self.bbox_head(
            body_feats, rois, spatial_scale)

        rois_has_mask_int32 = None
        if self.inputs['mode'] == 'infer':
            bbox_pred, bboxes = self.bbox_head.get_prediction(
                self.bbox_head_out, rois)
            # Refine bbox by the output from bbox_head at test stage
            self.bboxes = self.bbox_post_process(bbox_pred, bboxes,
                                                 self.inputs['im_shape'],
                                                 self.inputs['scale_factor'])
        else:
            # Proposal RoI for Mask branch
            # bboxes update at training stage only
            bbox_targets = self.proposal.get_targets()[0]
            self.bboxes, rois_has_mask_int32 = self.mask(self.inputs, rois,
                                                         bbox_targets)

        # Mask Head 
        self.mask_head_out = self.mask_head(
            self.inputs, body_feats, self.bboxes, bbox_feat,
            rois_has_mask_int32, spatial_scale, self.bbox_head_feat_func)

    def get_loss(self, ):
        loss = {}

        # RPN loss
        rpn_loss_inputs = self.anchor.generate_loss_inputs(
            self.inputs, self.rpn_head_out, self.anchor_out)
        loss_rpn = self.rpn_head.get_loss(rpn_loss_inputs)
        loss.update(loss_rpn)

        # BBox loss
        bbox_targets = self.proposal.get_targets()
        loss_bbox = self.bbox_head.get_loss([self.bbox_head_out], bbox_targets)
        loss.update(loss_bbox)

        # Mask loss
        mask_targets = self.mask.get_targets()
        loss_mask = self.mask_head.get_loss(self.mask_head_out, mask_targets)
        loss.update(loss_mask)

        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox, bbox_num = self.bboxes
        output = {
            'bbox': bbox,
            'bbox_num': bbox_num,
            'mask': self.mask_head_out
        }
        return output
