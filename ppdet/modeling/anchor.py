import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.base import to_variable

from ppdet.core.workspace import register
from ppdet.modeling.ops import AnchorGenerator, ProposalGenerator, ProposalTargetGenerator, MaskTargetGenerator


@register
class Anchor(Layer):
    __inject__ = ['anchor_generator']

    def __init__(self, anchor_generator=AnchorGenerator().__dict__):
        super(Anchor, self).__init__()
        self.anchor_generator = anchor_generator
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)

    def forward(self, inputs):
        self.inputs = inputs
        anchor_out = self.generate_anchors()
        if self.inputs['mode'] == 'train':
            anchor_target_out = self.generate_anchors_target()
            anchor_out.update(anchor_target_out)
        return anchor_out

    def generate_anchors(self, ):
        # TODO: update here to use int to specify featmap size
        outs = self.anchor_generator(self.inputs['rpn_neck'])
        outs = {'anchor': outs[0], 'var': outs[1]}
        return outs

    def generate_anchors_target(self, ):
        # TODO: mv from rpn_head to here 
        return {}


@register
class Proposal(Layer):
    __inject__ = ['proposal_generator', 'proposal_target_generator']

    def __init__(self,
                 proposal_generator=ProposalGenerator().__dict__,
                 proposal_target_generator=ProposalTargetGenerator()):
        super(Proposal, self).__init__()
        self.proposal_generator = proposal_generator
        self.proposal_target_generator = proposal_target_generator
        if isinstance(proposal_generator, dict):
            self.proposal_generator = ProposalGenerator(**proposal_generator)
        if isinstance(proposal_target_generator, dict):
            self.proposal_target_generator = ProposalTargetGenerator(
                **proposal_target_generator)

    def forward(self, inputs):
        self.inputs = inputs

        proposal_out = self.generate_proposal()
        if self.inputs['mode'] == 'train':
            proposal_target_out = self.generate_proposal_target(proposal_out)
            proposal_out.update(proposal_target_out)
        return proposal_out

    def generate_proposal(self, ):
        rpn_rois_prob = fluid.layers.sigmoid(
            self.inputs['rpn_rois_score'], name='rpn_rois_prob')
        outs = self.proposal_generator(
            scores=rpn_rois_prob,
            bbox_deltas=self.inputs['rpn_rois_delta'],
            anchors=self.inputs['anchor'],
            variances=self.inputs['var'],
            im_info=self.inputs['im_info'],
            mode=self.inputs['mode'])
        outs = {
            'rpn_rois': outs[0],
            'rpn_rois_probs': outs[1],
            'rpn_rois_nums': outs[2]
        }
        return outs

    def generate_proposal_target(self, proposal_out):
        outs = self.proposal_target_generator(
            rpn_rois=proposal_out['rpn_rois'],
            rpn_rois_nums=proposal_out['rpn_rois_nums'],
            gt_classes=self.inputs['gt_class'],
            is_crowd=self.inputs['is_crowd'],
            gt_boxes=self.inputs['gt_bbox'],
            im_info=self.inputs['im_info'])
        outs = {
            'rois': outs[0],
            'labels_int32': outs[1],
            'bbox_targets': outs[2],
            'bbox_inside_weights': outs[3],
            'bbox_outside_weights': outs[4],
            'rois_nums': outs[5]
        }
        return outs


@register
class Mask(Layer):
    __inject__ = ['mask_target_generator']

    def __init__(self, mask_target_generator=MaskTargetGenerator().__dict__):
        super(Mask, self).__init__()
        self.mask_target_generator = mask_target_generator
        if isinstance(mask_target_generator, dict):
            self.mask_target_generator = MaskTargetGenerator(
                **mask_target_generator)

    def forward(self, inputs):
        self.inputs = inputs
        if self.inputs['mode'] == 'train':
            outs = self.generate_mask_target()
            return outs

    def generate_mask_target(self, ):
        outs = self.mask_target_generator(
            im_info=self.inputs['im_info'],
            gt_classes=self.inputs['gt_class'],
            is_crowd=self.inputs['is_crowd'],
            gt_segms=self.inputs['gt_mask'],
            rois=self.inputs['rois'],
            rois_nums=self.inputs['rois_nums'],
            labels_int32=self.inputs['labels_int32'], )
        outs = {
            'mask_rois': outs[0],
            'rois_has_mask_int32': outs[1],
            'mask_int32': outs[2]
        }
        return outs
