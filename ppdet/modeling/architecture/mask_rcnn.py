from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from collections import OrderedDict
import copy

from paddle import fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.base import to_variable

from ppdet.core.workspace import register
from ppdet.utils.data_structure import BufferDict

__all__ = ['MaskRCNN']


@register
class MaskRCNN(Layer):
    __category__ = 'architecture'
    __inject__ = [
        # stage 1
        'backbone',
        'rpn_neck',
        'rpn_head',
        'anchor',
        # stage 2 
        'proposal',
        'mask',
        'roi_extractor',
        'bbox_neck',
        'bbox_head',
        'mask_neck',
        'mask_head'
    ]

    def __init__(
            self,
            backbone,
            rpn_neck,
            rpn_head,
            anchor,
            proposal,
            mask,
            roi_extractor,
            bbox_neck='BBoxNeck',
            bbox_head='BBoxHead',
            mask_neck='MaskNeck',
            mask_head='MaskHead',
            rpn_only=False, ):
        super(MaskRCNN, self).__init__()

        self.backbone = backbone
        self.rpn_neck = rpn_neck
        self.rpn_head = rpn_head
        self.anchor = anchor
        self.proposal = proposal
        self.roi_extractor = roi_extractor
        self.bbox_neck = bbox_neck
        self.bbox_head = bbox_head
        self.mask = mask
        self.mask_neck = mask_neck
        self.mask_head = mask_head
        self.rpn_only = rpn_only

    def forward(self, inputs, mode='train'):
        self.gbd = self.build_inputs(inputs)
        self.gbd['mode'] = mode

        # Backbone
        bb_out = self.backbone(self.gbd)
        self.gbd.update(bb_out)

        # RPN
        rpn_neck_out = self.rpn_neck(self.gbd)
        self.gbd.update(rpn_neck_out)
        rpn_head_out = self.rpn_head(self.gbd)
        self.gbd.update(rpn_head_out)

        # Anchor
        anchor_out = self.anchor(self.gbd)
        self.gbd.update(anchor_out)

        # Proposal BBox
        proposal_out = self.proposal(self.gbd)
        self.gbd.update(proposal_out)

        # RoI Extractor
        roi_out = self.roi_extractor(self.gbd)
        self.gbd.update(roi_out)

        # BBox Head
        bbox_neck_out = self.bbox_neck(self.gbd)
        self.gbd.update(bbox_neck_out)
        bbox_head_out = self.bbox_head(self.gbd)
        self.gbd.update(bbox_head_out)

        # Mask 
        mask_out = self.mask(self.gbd)
        self.gbd.update(mask_out)

        # Mask Head 
        mask_neck_out = self.mask_neck(self.gbd)
        self.gbd.update(mask_neck_out)
        mask_head_out = self.mask_head(self.gbd)
        self.gbd.update(mask_head_out)

        # result  
        if self.gbd['mode'] == 'train':
            return self.loss(self.gbd)
        elif self.gbd['mode'] == 'infer':
            self.post_processing(self.gbd)
        else:
            raise "Now, only support train or infer mode!"

    def loss(self, inputs):
        # used in train
        losses = []
        # RPN loss
        rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(inputs)

        # BBox loss
        bbox_cls_loss, bbox_reg_loss = self.bbox_head.loss(inputs)

        # Mask loss 
        mask_loss = self.mask_head.loss(inputs)

        # Total loss 
        losses = [
            rpn_cls_loss, rpn_reg_loss, bbox_cls_loss, bbox_reg_loss, mask_loss
        ]
        loss = fluid.layers.sum(losses)

        out = {
            'loss': loss,
            'loss_rpn_cls': rpn_cls_loss,
            'loss_rpn_reg': rpn_reg_loss,
            'loss_bbox_cls': bbox_cls_loss,
            'loss_bbox_reg': bbox_reg_loss,
            'loss_mask': mask_loss
        }
        return out

    def post_processing(self, inputs):
        # used in infer 
        pass

    def build_inputs(self,
                     inputs,
                     fields=[
                         'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class',
                         'is_crowd', 'gt_mask'
                     ]):
        gbd = BufferDict()
        # init input 
        for k in fields:
            gbd[k] = []
        # make batch list 
        for batch in inputs:
            for i, (k, v) in enumerate(zip(fields, batch)):
                gbd[k].append(v)
        # make variable  
        for k, v in gbd.items():
            batch = np.asarray(v)
            if k == 'gt_mask':
                batch = np.squeeze(batch, axis=3)
            gbd[k] = to_variable(batch)
        return gbd
