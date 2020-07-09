from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['CascadeRCNN']


@register
class CascadeRCNN(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['num_stages']
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
                 num_stages=3,
                 *args,
                 **kwargs):
        super(CascadeRCNN, self).__init__(*args, **kwargs)
        self.anchor = anchor
        self.proposal = proposal
        self.mask = mask
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head
        self.num_stages = num_stages

    def model_arch(self, ):
        # Backbone
        bb_out = self.backbone(self.gbd)
        self.gbd.update(bb_out)

        # RPN
        rpn_head_out = self.rpn_head(self.gbd)
        self.gbd.update(rpn_head_out)

        # Anchor
        anchor_out = self.anchor(self.gbd)
        self.gbd.update(anchor_out)

        self.gbd['stage'] = 0
        for i in range(self.num_stages):
            self.gbd.update_v('stage', i)
            # Proposal BBox
            proposal_out = self.proposal(self.gbd)
            self.gbd.update({"proposal_" + str(i): proposal_out})

            # BBox Head
            bbox_head_out = self.bbox_head(self.gbd)
            self.gbd.update({'bbox_head_' + str(i): bbox_head_out})

            refine_bbox_out = self.proposal.refine_bbox(self.gbd)
            self.gbd['proposal_' + str(i)].update(refine_bbox_out)

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

    def loss(self, ):
        outs = {}
        losses = []

        rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(self.gbd)
        outs['loss_rpn_cls'] = rpn_cls_loss
        outs['loss_rpn_reg'] = rpn_reg_loss
        losses.extend([rpn_cls_loss, rpn_reg_loss])

        bbox_cls_loss_list = []
        bbox_reg_loss_list = []
        for i in range(self.num_stages):
            self.gbd.update_v('stage', i)
            bbox_cls_loss, bbox_reg_loss = self.bbox_head.loss(self.gbd)
            bbox_cls_loss_list.append(bbox_cls_loss)
            bbox_reg_loss_list.append(bbox_reg_loss)
            outs['loss_bbox_cls_' + str(i)] = bbox_cls_loss
            outs['loss_bbox_reg_' + str(i)] = bbox_reg_loss
        losses.extend(bbox_cls_loss_list)
        losses.extend(bbox_reg_loss_list)

        mask_loss = self.mask_head.loss(self.gbd)
        outs['mask_loss'] = mask_loss
        losses.append(mask_loss)

        loss = fluid.layers.sum(losses)
        outs['loss'] = loss
        return outs

    def infer(self, ):
        outs = {
            'bbox': self.gbd['predicted_bbox'].numpy(),
            'bbox_nums': self.gbd['predicted_bbox_nums'].numpy(),
            'mask': self.gbd['predicted_mask'].numpy(),
            'im_id': self.gbd['im_id'].numpy(),
            'im_shape': self.gbd['im_shape'].numpy()
        }
        return inputs
