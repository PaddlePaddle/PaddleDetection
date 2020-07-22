from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['FasterRCNN']


@register
class FasterRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'anchor',
        'proposal',
        'backbone',
        'rpn_head',
        'bbox_head',
    ]

    def __init__(self, anchor, proposal, backbone, rpn_head, bbox_head, *args,
                 **kwargs):
        super(FasterRCNN, self).__init__(*args, **kwargs)
        self.anchor = anchor
        self.proposal = proposal
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head

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

        # Proposal BBox
        self.gbd['stage'] = 0
        proposal_out = self.proposal(self.gbd)
        self.gbd.update({'proposal_0': proposal_out})

        # BBox Head
        bboxhead_out = self.bbox_head(self.gbd)
        self.gbd.update({'bbox_head_0': bboxhead_out})

        if self.gbd['mode'] == 'infer':
            bbox_out = self.proposal.post_process(self.gbd)
            self.gbd.update(bbox_out)

    def loss(self, ):
        rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(self.gbd)
        bbox_cls_loss, bbox_reg_loss = self.bbox_head.loss(self.gbd)
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

    def infer(self, ):
        outs = {
            "bbox": self.gbd['predicted_bbox'].numpy(),
            "bbox_nums": self.gbd['predicted_bbox_nums'].numpy(),
            'im_id': self.gbd['im_id'].numpy()
        }
        return outs
