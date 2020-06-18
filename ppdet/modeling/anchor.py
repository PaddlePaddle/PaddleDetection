import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.base import to_variable

from ppdet.core.workspace import register
from ppdet.modeling.ops import (AnchorGenerator, RPNAnchorTargetGenerator,
                                ProposalGenerator, ProposalTargetGenerator,
                                MaskTargetGenerator, DecodeClipNms)
# TODO: modify here into ppdet.modeling.ops like DecodeClipNms 
from ppdet.py_op.post_process import mask_post_process


@register
class BBoxPostProcess(Layer):
    def __init__(self,
                 decode=None,
                 clip=None,
                 nms=None,
                 decode_clip_nms=DecodeClipNms().__dict__):
        super(BBoxPostProcess, self).__init__()
        self.decode = decode
        self.clip = clip
        self.nms = nms
        self.decode_clip_nms = decode_clip_nms
        if isinstance(decode_clip_nms, dict):
            self.decode_clip_nms = DecodeClipNms(**decode_clip_nms)

    def __call__(self, inputs):
        # TODO: split into 3 steps
        # TODO: modify related ops for deploying
        # decode
        # clip
        # nms
        outs = self.decode_clip_nms(inputs['rpn_rois'], inputs['bbox_prob'],
                                    inputs['bbox_delta'], inputs['im_info'])
        outs = {"predicted_bbox_nums": outs[0], "predicted_bbox": outs[1]}
        return outs


@register
class MaskPostProcess(object):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=81):
        super(MaskPostProcess, self).__init__()
        self.num_classes = num_classes

    def __call__(self, inputs):
        # TODO: modify related ops for deploying
        outs = mask_post_process(inputs['predicted_bbox_nums'].numpy(),
                                 inputs['predicted_bbox'].numpy(),
                                 inputs['mask_logits'].numpy(),
                                 inputs['im_info'].numpy())
        outs = {'predicted_mask': outs}
        return outs


@register
class Anchor(Layer):
    __inject__ = ['anchor_generator', 'anchor_target_generator']

    def __init__(self,
                 anchor_type='rpn',
                 anchor_generator=AnchorGenerator().__dict__,
                 anchor_target_generator=RPNAnchorTargetGenerator().__dict__):
        super(Anchor, self).__init__()
        self.anchor_generator = anchor_generator
        self.anchor_target_generator = anchor_target_generator
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(anchor_target_generator, dict):
            self.anchor_target_generator = RPNAnchorTargetGenerator(
                **anchor_target_generator)

    def forward(self, inputs):
        self.inputs = inputs
        anchor_out = self.generate_anchors()
        return anchor_out

    def generate_anchors(self, ):
        # TODO: update here to use int to specify featmap size
        outs = self.anchor_generator(self.inputs['rpn_feat'])
        outs = {'anchor': outs[0], 'var': outs[1], 'anchor_module': self}
        return outs

    def generate_anchors_target(self, inputs):
        # TODO: add rpn anchor targets
        # TODO: add yolo anchor targets 
        rpn_rois_score = fluid.layers.transpose(
            inputs['rpn_rois_score'], perm=[0, 2, 3, 1])
        rpn_rois_delta = fluid.layers.transpose(
            inputs['rpn_rois_delta'], perm=[0, 2, 3, 1])
        rpn_rois_score = fluid.layers.reshape(
            x=rpn_rois_score, shape=(0, -1, 1))
        rpn_rois_delta = fluid.layers.reshape(
            x=rpn_rois_delta, shape=(0, -1, 4))

        anchor = fluid.layers.reshape(inputs['anchor'], shape=(-1, 4))
        #var = fluid.layers.reshape(inputs['var'], shape=(-1, 4))

        score_pred, roi_pred, score_tgt, roi_tgt, roi_weight = self.anchor_target_generator(
            bbox_pred=rpn_rois_delta,
            cls_logits=rpn_rois_score,
            anchor_box=anchor,
            gt_boxes=inputs['gt_bbox'],
            is_crowd=inputs['is_crowd'],
            im_info=inputs['im_info'])
        outs = {
            'rpn_score_pred': score_pred,
            'rpn_score_target': score_tgt,
            'rpn_rois_pred': roi_pred,
            'rpn_rois_target': roi_tgt,
            'rpn_rois_weight': roi_weight
        }
        return outs

    def post_process(self, ):
        # TODO: whether move bbox post process to here 
        pass


@register
class Proposal(Layer):
    __inject__ = [
        'proposal_generator', 'proposal_target_generator', 'bbox_post_process'
    ]

    def __init__(
            self,
            proposal_generator=ProposalGenerator().__dict__,
            proposal_target_generator=ProposalTargetGenerator().__dict__,
            bbox_post_process=BBoxPostProcess().__dict__, ):
        super(Proposal, self).__init__()
        self.proposal_generator = proposal_generator
        self.proposal_target_generator = proposal_target_generator
        self.bbox_post_process = bbox_post_process
        if isinstance(proposal_generator, dict):
            self.proposal_generator = ProposalGenerator(**proposal_generator)
        if isinstance(proposal_target_generator, dict):
            self.proposal_target_generator = ProposalTargetGenerator(
                **proposal_target_generator)
        if isinstance(bbox_post_process, dict):
            self.bbox_post_process = BBoxPostProcess(**bbox_post_process)

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

    def post_process(self, inputs):
        outs = self.bbox_post_process(inputs)
        return outs


@register
class Mask(Layer):
    __inject__ = ['mask_target_generator', 'mask_post_process']

    def __init__(self,
                 mask_target_generator=MaskTargetGenerator().__dict__,
                 mask_post_process=MaskPostProcess().__dict__):
        super(Mask, self).__init__()
        self.mask_target_generator = mask_target_generator
        self.mask_post_process = mask_post_process
        if isinstance(mask_target_generator, dict):
            self.mask_target_generator = MaskTargetGenerator(
                **mask_target_generator)
        if isinstance(mask_post_process, dict):
            self.mask_post_process = MaskPostProcess(**mask_post_process)

    def forward(self, inputs):
        self.inputs = inputs
        outs = {}
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

    def post_process(self, inputs):
        outs = self.mask_post_process(inputs)
        return outs
