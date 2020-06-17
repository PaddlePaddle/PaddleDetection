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

__all__ = ['FasterRCNN']


@register
class FasterRCNN(Layer):
    __category__ = 'architecture'
    __inject__ = [
        'anchor',
        'proposal',
        'backbone',
        'rpn_head',
        'bbox_head',
    ]

    def __init__(self,
                 anchor,
                 proposal,
                 backbone,
                 rpn_head,
                 bbox_head,
                 rpn_only=False):
        super(FasterRCNN, self).__init__()
        self.anchor = anchor
        self.proposal = proposal
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.rpn_only = rpn_only

    def forward(self, inputs, mode='train'):
        self.gbd = self.build_inputs(inputs)
        self.gbd['mode'] = mode

        # Backbone
        bb_out = self.backbone(self.gbd)
        self.gbd.update(bb_out)

        # RPN
        rpn_head_out = self.rpn_head(self.gbd)
        self.gbd.update(rpn_head_out)

        # Anchor
        anchor_out = self.anchor(self.gbd)
        self.gbd.update(anchor_out)

        # Proposal BBox
        proposal_out = self.proposal(self.gbd)
        self.gbd.update(proposal_out)

        # BBox Head
        bbox_head_out = self.bbox_head(self.gbd)
        self.gbd.update(bbox_head_out)

        if self.gbd['mode'] == 'infer':
            bbox_out = self.proposal.post_process(self.gbd)
            self.gbd.update(bbox_out)

        # result  
        if self.gbd['mode'] == 'train':
            return self.loss(self.gbd)
        elif self.gbd['mode'] == 'infer':
            return self.infer(self.gbd)
        else:
            raise "Now, only support train or infer mode!"

    def loss(self, inputs):
        # used in train
        losses = []
        # RPN loss
        rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(inputs)
        # BBox loss
        bbox_cls_loss, bbox_reg_loss = self.bbox_head.loss(inputs)
        # Total loss 
        losses = [rpn_cls_loss, rpn_reg_loss, bbox_cls_loss, bbox_reg_loss]
        loss = fluid.layers.sum(losses)
        out = {
            'loss': loss,
            'loss_rpn_cls': rpn_cls_loss,
            'loss_rpn_reg': rpn_reg_loss,
            'loss_bbox_cls': bbox_cls_loss,
            'loss_bbox_reg': bbox_reg_loss,
        }
        return out

    def infer(self, inputs):
        outs = {
            "bbox_nums": inputs['predicted_bbox_nums'].numpy(),
            "bbox": inputs['predicted_bbox'].numpy(),
        }
        return outs

    def build_inputs(
            self,
            inputs,
            #fields=['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd']
            fields=['image', 'im_info', 'im_id', 'im_shape']):
        gbd = BufferDict()
        for i, k in enumerate(fields):
            v = to_variable(np.array([x[i] for x in inputs]))
            gbd.set(k, v)
        return gbd
