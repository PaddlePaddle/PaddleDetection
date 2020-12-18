from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['FasterRCNN']


@register
class FasterRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'anchor', 'proposal', 'backbone', 'neck', 'rpn_head', 'bbox_head',
        'bbox_post_process'
    ]

    def __init__(self,
                 anchor,
                 proposal,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 neck=None):
        super(FasterRCNN, self).__init__()
        self.anchor = anchor
        self.proposal = proposal
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process
        self.neck = neck

    def model_arch(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        spatial_scale = 0.0625

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
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox, bbox_num = self.bboxes
        output = {
            'bbox': bbox,
            'bbox_num': bbox_num,
        }
        return output
