import numpy as np
import paddle.fluid as fluid
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from . import ops


@register
class Anchor(object):
    __inject__ = ['anchor_generator', 'anchor_target_generator']

    def __init__(self, anchor_generator, anchor_target_generator):
        super(Anchor, self).__init__()
        self.anchor_generator = anchor_generator
        self.anchor_target_generator = anchor_target_generator

    def __call__(self, rpn_feats):
        anchors = []
        num_level = len(rpn_feats)
        for i, rpn_feat in enumerate(rpn_feats):
            anchor, var = self.anchor_generator(rpn_feat, i)
            anchors.append((anchor, var))
        return anchors

    def _get_target_input(self, rpn_feats, anchors):
        rpn_score_list = []
        rpn_delta_list = []
        anchor_list = []
        for (rpn_score, rpn_delta), (anchor, var) in zip(rpn_feats, anchors):
            rpn_score = fluid.layers.transpose(rpn_score, perm=[0, 2, 3, 1])
            rpn_delta = fluid.layers.transpose(rpn_delta, perm=[0, 2, 3, 1])
            rpn_score = fluid.layers.reshape(x=rpn_score, shape=(0, -1, 1))
            rpn_delta = fluid.layers.reshape(x=rpn_delta, shape=(0, -1, 4))

            anchor = fluid.layers.reshape(anchor, shape=(-1, 4))
            var = fluid.layers.reshape(var, shape=(-1, 4))
            rpn_score_list.append(rpn_score)
            rpn_delta_list.append(rpn_delta)
            anchor_list.append(anchor)

        rpn_scores = fluid.layers.concat(rpn_score_list, axis=1)
        rpn_deltas = fluid.layers.concat(rpn_delta_list, axis=1)
        anchors = fluid.layers.concat(anchor_list)
        return rpn_scores, rpn_deltas, anchors

    def generate_loss_inputs(self, inputs, rpn_head_out, anchors):
        assert len(rpn_head_out) == len(
            anchors
        ), "rpn_head_out and anchors should have same length, but received rpn_head_out' length is {} and anchors' length is {}".format(
            len(rpn_head_out), len(anchors))
        rpn_score, rpn_delta, anchors = self._get_target_input(rpn_head_out,
                                                               anchors)

        score_pred, roi_pred, score_tgt, roi_tgt, roi_weight = self.anchor_target_generator(
            bbox_pred=rpn_delta,
            cls_logits=rpn_score,
            anchor_box=anchors,
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


@register
class AnchorYOLO(object):
    __inject__ = ['anchor_generator']

    def __init__(self, anchor_generator):
        super(AnchorYOLO, self).__init__()
        self.anchor_generator = anchor_generator

    def __call__(self):
        return self.anchor_generator()


@register
class Proposal(object):
    __inject__ = ['proposal_generator', 'proposal_target_generator']

    def __init__(self, proposal_generator, proposal_target_generator):
        super(Proposal, self).__init__()
        self.proposal_generator = proposal_generator
        self.proposal_target_generator = proposal_target_generator

    def generate_proposal(self, inputs, rpn_head_out, anchor_out):
        # TODO: delete im_info 
        try:
            im_shape = inputs['im_info']
        except:
            im_shape = inputs['im_shape']
        rpn_rois_list = []
        rpn_prob_list = []
        rpn_rois_num_list = []
        for (rpn_score, rpn_delta), (anchor, var) in zip(rpn_head_out,
                                                         anchor_out):
            rpn_prob = fluid.layers.sigmoid(rpn_score)
            rpn_rois, rpn_rois_prob, rpn_rois_num, post_nms_top_n = self.proposal_generator(
                scores=rpn_prob,
                bbox_deltas=rpn_delta,
                anchors=anchor,
                variances=var,
                im_shape=im_shape,
                mode=inputs['mode'])
            if len(rpn_head_out) == 1:
                return rpn_rois, rpn_rois_num
            rpn_rois_list.append(rpn_rois)
            rpn_prob_list.append(rpn_rois_prob)
            rpn_rois_num_list.append(rpn_rois_num)

        start_level = 2
        end_level = start_level + len(rpn_head_out)
        rois_collect, rois_num_collect = ops.collect_fpn_proposals(
            rpn_rois_list,
            rpn_prob_list,
            start_level,
            end_level,
            post_nms_top_n,
            rois_num_per_level=rpn_rois_num_list)
        return rois_collect, rois_num_collect

    def generate_proposal_target(self, inputs, rois, rois_num, stage=0):
        outs = self.proposal_target_generator(
            rpn_rois=rois,
            rpn_rois_num=rois_num,
            gt_classes=inputs['gt_class'],
            is_crowd=inputs['is_crowd'],
            gt_boxes=inputs['gt_bbox'],
            im_info=inputs['im_info'],
            stage=stage)
        rois = outs[0]
        rois_num = outs[-1]
        targets = {
            'labels_int32': outs[1],
            'bbox_targets': outs[2],
            'bbox_inside_weights': outs[3],
            'bbox_outside_weights': outs[4]
        }
        return rois, rois_num, targets

    def refine_bbox(self, rois, bbox_delta, stage=0):
        out_dim = bbox_delta.shape[1] / 4
        bbox_delta_r = fluid.layers.reshape(bbox_delta, (-1, out_dim, 4))
        bbox_delta_s = fluid.layers.slice(
            bbox_delta_r, axes=[1], starts=[1], ends=[2])

        refined_bbox = ops.box_coder(
            prior_box=rois,
            prior_box_var=self.proposal_target_generator.bbox_reg_weights[
                stage],
            target_box=bbox_delta_s,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1)
        refined_bbox = fluid.layers.reshape(refined_bbox, shape=[-1, 4])
        return refined_bbox

    def __call__(self,
                 inputs,
                 rpn_head_out,
                 anchor_out,
                 stage=0,
                 proposal_out=None,
                 bbox_head_outs=None,
                 refined=False):
        if refined:
            assert proposal_out is not None, "If proposal has been refined, proposal_out should not be None."
            return proposal_out
        if stage == 0:
            roi, rois_num = self.generate_proposal(inputs, rpn_head_out,
                                                   anchor_out)
            self.proposals_list = []
            self.targets_list = []

        else:
            bbox_delta = bbox_head_outs[stage][0]
            roi = self.refine_bbox(proposal_out[0], bbox_delta, stage - 1)
            rois_num = proposal_out[1]
        if inputs['mode'] == 'train':
            roi, rois_num, targets = self.generate_proposal_target(
                inputs, roi, rois_num, stage)
            self.targets_list.append(targets)
        self.proposals_list.append((roi, rois_num))
        return roi, rois_num

    def get_targets(self):
        return self.targets_list

    def get_proposals(self):
        return self.proposals_list
