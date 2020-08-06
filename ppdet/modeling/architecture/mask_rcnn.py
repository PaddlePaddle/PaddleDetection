from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
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
    ]

    def __init__(self, anchor, proposal, mask, backbone, neck, rpn_head,
                 bbox_head, mask_head, *args, **kwargs):
        super(MaskRCNN, self).__init__(*args, **kwargs)
        self.anchor = anchor
        self.proposal = proposal
        self.mask = mask
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head

    def model_arch(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        spatial_scale = None

        # Neck
        if self.neck is not None:
            body_feats, spatial_scale = self.neck(body_feats)

        # RPN
        # rpn_head returns two list: rpn_feat, rpn_head_out 
        # each element in rpn_feats contains 
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
        bbox_feat, self.bbox_head_out = self.bbox_head(body_feats, rois,
                                                       spatial_scale)

        rois_has_mask_int32 = None
        if self.inputs['mode'] == 'infer':
            # Refine bbox by the output from bbox_head at test stage
            self.bboxes = self.proposal.post_process(self.inputs,
                                                     self.bbox_head_out, rois)
        else:
            # Proposal RoI for Mask branch
            # bboxes update at training stage only
            bbox_targets = self.proposal.get_targets()[0]
            self.bboxes, rois_has_mask_int32 = self.mask(self.inputs, rois,
                                                         bbox_targets)

        # Mask Head 
        self.mask_head_out = self.mask_head(self.inputs, body_feats,
                                            self.bboxes, bbox_feat,
                                            rois_has_mask_int32, spatial_scale)

    def loss(self, ):
        loss = {}

        # RPN loss
        rpn_loss_inputs = self.anchor.generate_loss_inputs(
            self.inputs, self.rpn_head_out, self.anchor_out)
        loss_rpn = self.rpn_head.loss(rpn_loss_inputs)
        loss.update(loss_rpn)

        # BBox loss
        bbox_targets = self.proposal.get_targets()
        loss_bbox = self.bbox_head.loss(self.bbox_head_out, bbox_targets)
        loss.update(loss_bbox)

        # Mask loss
        mask_targets = self.mask.get_targets()
        loss_mask = self.mask_head.loss(self.mask_head_out, mask_targets)
        loss.update(loss_mask)

        total_loss = fluid.layers.sums(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def infer(self, ):
        mask = self.mask.post_process(self.bboxes, self.mask_head_out,
                                      self.inputs['im_info'])
        bbox, bbox_num = self.bboxes
        output = {
            'bbox': bbox.numpy(),
            'bbox_num': bbox_num.numpy(),
            'im_id': self.inputs['im_id'].numpy()
        }
        output.update(mask)
        return output
