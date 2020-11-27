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
<<<<<<< HEAD
        body_feats = self.backbone(self.inputs)

        spatial_scale = None
=======
        body_feats = self.backbone(self.inputs, False)
        spatial_scale = [0.25, 0.125, 0.0625, 0.03125]
>>>>>>> 55de7f1178fe806386501fc1b24407f116622cfd

        # Neck
        if self.neck is not None:
            body_feats, spatial_scale = self.neck(body_feats)
<<<<<<< HEAD
=======
            rpn_feat, self.rpn_head_out = self.rpn_head(body_feats)
        else:
            rpn_feat, self.rpn_head_out = self.rpn_head([body_feats[-1]])    
>>>>>>> 55de7f1178fe806386501fc1b24407f116622cfd

        # RPN
        # rpn_head returns two list: rpn_feat, rpn_head_out
        # each element in rpn_feats contains rpn feature on each level,
        # and the length is 1 when the neck is not applied.
        # each element in rpn_head_out contains (rpn_rois_score, rpn_rois_delta)
<<<<<<< HEAD
        rpn_feat, self.rpn_head_out = self.rpn_head(self.inputs, body_feats)
=======
>>>>>>> 55de7f1178fe806386501fc1b24407f116622cfd

        # Anchor
        # anchor_out returns a list,
        # each element contains (anchor, anchor_var)
        self.anchor_out = self.anchor(rpn_feat)

        # Proposal RoI
        # compute targets here when training
        rois = self.proposal(self.inputs, self.rpn_head_out, self.anchor_out)
        # BBox Head
<<<<<<< HEAD
        bbox_feat, self.bbox_head_out = self.bbox_head(body_feats, rois,
                                                       spatial_scale)

        if self.inputs['mode'] == 'infer':
            bbox_pred, bboxes = self.bbox_head.get_prediction(
                self.bbox_head_out, rois)
            # Refine bbox by the output from bbox_head at test stage
            self.bboxes = self.bbox_post_process(bbox_pred, bboxes,
                                                 self.inputs['im_info'])
=======
        bbox_head_return_stage = 0
        if not self.bbox_head.use_resnetc5:
            bbox_feat = self.bbox_head.bbox_feat(body_feats, rois, spatial_scale, stage=bbox_head_return_stage)
            bbox_feat, self.bbox_head_out = self.bbox_head(bbox_feat, stage=bbox_head_return_stage)
        else:
            rois_feat = self.bbox_head.bbox_feat.roi_extractor(body_feats, rois, spatial_scale)
            bbox_feat = self.backbone(rois_feat, use_resnetc5=True)

            bbox_feat = paddle.fluid.layers.pool2d(bbox_feat, pool_type='avg', global_pooling=True)
            bbox_feat = paddle.reshape(bbox_feat, (bbox_feat.shape[0], bbox_feat.shape[1]))
            bbox_feat, self.bbox_head_out = self.bbox_head(bbox_feat, stage=bbox_head_return_stage)


        if self.inputs['mode'] == 'infer':
            bbox_pred, bboxes = self.bbox_head.get_prediction(
                self.bbox_head_out, rois)
            # Refine bbox by the output from bbox_head at test stage
            self.bboxes = self.bbox_post_process(bbox_pred, bboxes,
                                                 self.inputs['im_shape'],
                                                 self.inputs['scale_factor'])

>>>>>>> 55de7f1178fe806386501fc1b24407f116622cfd
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
        loss_bbox = self.bbox_head.get_loss(self.bbox_head_out, bbox_targets)
        loss.update(loss_bbox)
<<<<<<< HEAD

        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self, ):
=======

        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self, return_numpy=True):
>>>>>>> 55de7f1178fe806386501fc1b24407f116622cfd
        bbox, bbox_num = self.bboxes
        output = {
            'bbox': bbox.numpy(),
            'bbox_num': bbox_num.numpy(),
            'im_id': self.inputs['im_id'].numpy()
        }

        return output
