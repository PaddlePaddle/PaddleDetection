from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from paddle import fluid
from paddle.fluid.dygraph.base import to_variable

from ppdet.core.workspace import register
from ppdet.utils.data_structure import BufferDict

__all__ = ['MaskRCNN']


@register
class MaskRCNN(fluid.dygraph.Layer):
    __category__ = 'architecture'
    __inject__ = [
        'anchor',
        'proposal',
        'mask',
        'backbone',
        'rpn_head',
        'bbox_head',
        'mask_head',
    ]

    def __init__(self,
                 anchor,
                 proposal,
                 mask,
                 backbone,
                 rpn_head,
                 bbox_head,
                 mask_head,
                 rpn_only=False):
        super(MaskRCNN, self).__init__()

        self.anchor = anchor
        self.proposal = proposal
        self.mask = mask
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head

    def forward(self, inputs, mode='train'):
        self.gbd = self.build_inputs(inputs, mode)
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

        # Mask 
        mask_out = self.mask(self.gbd)
        self.gbd.update(mask_out)

        # Mask Head 
        mask_head_out = self.mask_head(self.gbd)
        self.gbd.update(mask_head_out)

        if self.gbd['mode'] == 'infer':
            mask_out = self.mask.post_process(self.gbd)
            self.gbd.update(mask_out)

        # result  
        if self.gbd['mode'] == 'train':
            return self.loss(self.gbd)
        elif self.gbd['mode'] == 'infer':
            self.infer(self.gbd)
        else:
            raise "Now, only support train or infer mode!"

    def loss(self, inputs):
        losses = []
        rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(inputs)
        bbox_cls_loss, bbox_reg_loss = self.bbox_head.loss(inputs)
        mask_loss = self.mask_head.loss(inputs)
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

    def infer(self, inputs):
        outs = {
            'bbox_nums': inputs['predicted_bbox_nums'].numpy(),
            'bbox': inputs['predicted_bbox'].numpy(),
            'mask': inputs['predicted_mask'].numpy(),
            'im_id': inputs['im_id'].numpy(),
            'im_shape': inputs['im_shape'].numpy()
        }
        return inputs

    def build_inputs(self, inputs, mode='train'):
        input_keys = [
            'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd',
            'gt_mask'
        ]
        if mode == 'infer':
            input_keys = ['image', 'im_info', 'im_id', 'im_shape']
        gbd = BufferDict()
        for i, k in enumerate(input_keys):
            v = to_variable(np.array([x[i] for x in inputs]))
            gbd.set(k, v)

        return gbd
