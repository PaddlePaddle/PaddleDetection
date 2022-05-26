# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling.ops import (AnchorGenerator, RPNTargetAssign,
                                GenerateProposals)

__all__ = ['RPNTargetAssign', 'GenerateProposals', 'RPNHead', 'FPNRPNHead']


@register
class RPNHead(object):
    """
    RPN Head

    Args:
        anchor_generator (object): `AnchorGenerator` instance
        rpn_target_assign (object): `RPNTargetAssign` instance
        train_proposal (object): `GenerateProposals` instance for training
        test_proposal (object): `GenerateProposals` instance for testing
        num_classes (int): number of classes in rpn output
    """
    __inject__ = [
        'anchor_generator', 'rpn_target_assign', 'train_proposal',
        'test_proposal'
    ]

    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 train_proposal=GenerateProposals(12000, 2000).__dict__,
                 test_proposal=GenerateProposals().__dict__,
                 num_classes=1):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        self.num_classes = num_classes
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = GenerateProposals(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = GenerateProposals(**test_proposal)

    def _get_output(self, input):
        """
        Get anchor and RPN head output.

        Args:
            input(Variable): feature map from backbone with shape of [N, C, H, W]

        Returns:
            rpn_cls_score(Variable): Output of rpn head with shape of
                [N, num_anchors, H, W].
            rpn_bbox_pred(Variable): Output of rpn head with shape of
                [N, num_anchors * 4, H, W].
        """
        dim_out = input.shape[1]
        rpn_conv = fluid.layers.conv2d(
            input=input,
            num_filters=dim_out,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            name='conv_rpn',
            param_attr=ParamAttr(
                name="conv_rpn_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))
        # Generate anchors
        self.anchor, self.anchor_var = self.anchor_generator(input=rpn_conv)
        num_anchor = self.anchor.shape[2]
        # Proposal classification scores
        self.rpn_cls_score = fluid.layers.conv2d(
            rpn_conv,
            num_filters=num_anchor * self.num_classes,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_cls_score',
            param_attr=ParamAttr(
                name="rpn_cls_logits_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_cls_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        # Proposal bbox regression deltas
        self.rpn_bbox_pred = fluid.layers.conv2d(
            rpn_conv,
            num_filters=4 * num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_bbox_pred',
            param_attr=ParamAttr(
                name="rpn_bbox_pred_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_bbox_pred_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return self.rpn_cls_score, self.rpn_bbox_pred

    def get_proposals(self, body_feats, im_info, mode='train'):
        """
        Get proposals according to the output of backbone.

        Args:
            body_feats (dict): The dictionary of feature maps from backbone.
            im_info(Variable): The information of image with shape [N, 3] with
                shape (height, width, scale).
            body_feat_names(list): A list of names of feature maps from
                backbone.

        Returns:
            rpn_rois(Variable): Output proposals with shape of (rois_num, 4).
        """

        # In RPN Heads, only the last feature map of backbone is used.
        # And body_feat_names[-1] represents the last level name of backbone.
        body_feat = list(body_feats.values())[-1]
        rpn_cls_score, rpn_bbox_pred = self._get_output(body_feat)

        if self.num_classes == 1:
            rpn_cls_prob = fluid.layers.sigmoid(
                rpn_cls_score, name='rpn_cls_prob')
        else:
            rpn_cls_score = fluid.layers.transpose(
                rpn_cls_score, perm=[0, 2, 3, 1])
            rpn_cls_score = fluid.layers.reshape(
                rpn_cls_score, shape=(0, 0, 0, -1, self.num_classes))
            rpn_cls_prob_tmp = fluid.layers.softmax(
                rpn_cls_score, use_cudnn=False, name='rpn_cls_prob')
            rpn_cls_prob_slice = fluid.layers.slice(
                rpn_cls_prob_tmp, axes=[4], starts=[1],
                ends=[self.num_classes])
            rpn_cls_prob, _ = fluid.layers.topk(rpn_cls_prob_slice, 1)
            rpn_cls_prob = fluid.layers.reshape(
                rpn_cls_prob, shape=(0, 0, 0, -1))
            rpn_cls_prob = fluid.layers.transpose(
                rpn_cls_prob, perm=[0, 3, 1, 2])
        prop_op = self.train_proposal if mode == 'train' else self.test_proposal
        rpn_rois, rpn_roi_probs = prop_op(
            scores=rpn_cls_prob,
            bbox_deltas=rpn_bbox_pred,
            im_info=im_info,
            anchors=self.anchor,
            variances=self.anchor_var)
        return rpn_rois

    def _transform_input(self, rpn_cls_score, rpn_bbox_pred, anchor,
                         anchor_var):
        rpn_cls_score = fluid.layers.transpose(rpn_cls_score, perm=[0, 2, 3, 1])
        rpn_bbox_pred = fluid.layers.transpose(rpn_bbox_pred, perm=[0, 2, 3, 1])
        anchor = fluid.layers.reshape(anchor, shape=(-1, 4))
        anchor_var = fluid.layers.reshape(anchor_var, shape=(-1, 4))
        rpn_cls_score = fluid.layers.reshape(
            x=rpn_cls_score, shape=(0, -1, self.num_classes))
        rpn_bbox_pred = fluid.layers.reshape(x=rpn_bbox_pred, shape=(0, -1, 4))
        return rpn_cls_score, rpn_bbox_pred, anchor, anchor_var

    def _get_loss_input(self):
        for attr in ['rpn_cls_score', 'rpn_bbox_pred', 'anchor', 'anchor_var']:
            if not getattr(self, attr, None):
                raise ValueError("self.{} should not be None,".format(attr),
                                 "call RPNHead.get_proposals first")
        return self._transform_input(self.rpn_cls_score, self.rpn_bbox_pred,
                                     self.anchor, self.anchor_var)

    def get_loss(self, im_info, gt_box, is_crowd, gt_label=None):
        """
        Sample proposals and Calculate rpn loss.

        Args:
            im_info(Variable): The information of image with shape [N, 3] with
                shape (height, width, scale).
            gt_box(Variable): The ground-truth bounding boxes with shape [M, 4].
                M is the number of groundtruth.
            is_crowd(Variable): Indicates groud-truth is crowd or not with
                shape [M, 1]. M is the number of groundtruth.

        Returns:
            Type: dict
                rpn_cls_loss(Variable): RPN classification loss.
                rpn_bbox_loss(Variable): RPN bounding box regression loss.

        """
        rpn_cls, rpn_bbox, anchor, anchor_var = self._get_loss_input()
        if self.num_classes == 1:
            score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = \
                self.rpn_target_assign(
                    bbox_pred=rpn_bbox,
                    cls_logits=rpn_cls,
                    anchor_box=anchor,
                    anchor_var=anchor_var,
                    gt_boxes=gt_box,
                    is_crowd=is_crowd,
                    im_info=im_info)
            score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
            score_tgt.stop_gradient = True
            rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=score_pred, label=score_tgt)
        else:
            score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = \
                self.rpn_target_assign(
                    bbox_pred=rpn_bbox,
                    cls_logits=rpn_cls,
                    anchor_box=anchor,
                    anchor_var=anchor_var,
                    gt_boxes=gt_box,
                    gt_labels=gt_label,
                    is_crowd=is_crowd,
                    num_classes=self.num_classes,
                    im_info=im_info)
            labels_int64 = fluid.layers.cast(x=score_tgt, dtype='int64')
            labels_int64.stop_gradient = True
            rpn_cls_loss = fluid.layers.softmax_with_cross_entropy(
                logits=score_pred, label=labels_int64, numeric_stable_mode=True)

        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        loc_tgt = fluid.layers.cast(x=loc_tgt, dtype='float32')
        loc_tgt.stop_gradient = True
        rpn_reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        rpn_reg_loss = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        rpn_reg_loss = rpn_reg_loss / norm

        return {'loss_rpn_cls': rpn_cls_loss, 'loss_rpn_bbox': rpn_reg_loss}


@register
class FPNRPNHead(RPNHead):
    """
    RPN Head that supports FPN input

    Args:
        anchor_generator (object): `AnchorGenerator` instance
        rpn_target_assign (object): `RPNTargetAssign` instance
        train_proposal (object): `GenerateProposals` instance for training
        test_proposal (object): `GenerateProposals` instance for testing
        anchor_start_size (int): size of anchor at the first scale
        num_chan (int): number of FPN output channels
        min_level (int): lowest level of FPN output
        max_level (int): highest level of FPN output
        num_classes (int): number of classes in rpn output
    """

    __inject__ = [
        'anchor_generator', 'rpn_target_assign', 'train_proposal',
        'test_proposal'
    ]

    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 train_proposal=GenerateProposals(12000, 2000).__dict__,
                 test_proposal=GenerateProposals().__dict__,
                 anchor_start_size=32,
                 num_chan=256,
                 min_level=2,
                 max_level=6,
                 num_classes=1):
        super(FPNRPNHead, self).__init__(anchor_generator, rpn_target_assign,
                                         train_proposal, test_proposal)
        self.anchor_start_size = anchor_start_size
        self.num_chan = num_chan
        self.min_level = min_level
        self.max_level = max_level
        self.num_classes = num_classes

        self.fpn_rpn_list = []
        self.anchors_list = []
        self.anchor_var_list = []

    def _get_output(self, input, feat_lvl):
        """
        Get anchor and FPN RPN head output at one level.

        Args:
            input(Variable): Body feature from backbone.
            feat_lvl(int): Indicate the level of rpn output corresponding
                to the level of feature map.

        Return:
            rpn_cls_score(Variable): Output of one level of fpn rpn head with
                shape of [N, num_anchors, H, W].
            rpn_bbox_pred(Variable): Output of one level of fpn rpn head with
                shape of [N, num_anchors * 4, H, W].
        """
        slvl = str(feat_lvl)
        conv_name = 'conv_rpn_fpn' + slvl
        cls_name = 'rpn_cls_logits_fpn' + slvl
        bbox_name = 'rpn_bbox_pred_fpn' + slvl
        conv_share_name = 'conv_rpn_fpn' + str(self.min_level)
        cls_share_name = 'rpn_cls_logits_fpn' + str(self.min_level)
        bbox_share_name = 'rpn_bbox_pred_fpn' + str(self.min_level)

        num_anchors = len(self.anchor_generator.aspect_ratios)
        conv_rpn_fpn = fluid.layers.conv2d(
            input=input,
            num_filters=self.num_chan,
            filter_size=3,
            padding=1,
            act='relu',
            name=conv_name,
            param_attr=ParamAttr(
                name=conv_share_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=conv_share_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        self.anchors, self.anchor_var = self.anchor_generator(
            input=conv_rpn_fpn,
            anchor_sizes=(self.anchor_start_size * 2.
                          **(feat_lvl - self.min_level), ),
            stride=(2.**feat_lvl, 2.**feat_lvl))

        cls_num_filters = num_anchors * self.num_classes
        self.rpn_cls_score = fluid.layers.conv2d(
            input=conv_rpn_fpn,
            num_filters=cls_num_filters,
            filter_size=1,
            act=None,
            name=cls_name,
            param_attr=ParamAttr(
                name=cls_share_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=cls_share_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        self.rpn_bbox_pred = fluid.layers.conv2d(
            input=conv_rpn_fpn,
            num_filters=num_anchors * 4,
            filter_size=1,
            act=None,
            name=bbox_name,
            param_attr=ParamAttr(
                name=bbox_share_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=bbox_share_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return self.rpn_cls_score, self.rpn_bbox_pred

    def _get_single_proposals(self, body_feat, im_info, feat_lvl, mode='train'):
        """
        Get proposals in one level according to the output of fpn rpn head

        Args:
            body_feat(Variable): the feature map from backone.
            im_info(Variable): The information of image with shape [N, 3] with
                format (height, width, scale).
            feat_lvl(int): Indicate the level of proposals corresponding to
                the feature maps.

        Returns:
            rpn_rois_fpn(Variable): Output proposals with shape of (rois_num, 4).
            rpn_roi_probs_fpn(Variable): Scores of proposals with
                shape of (rois_num, 1).
        """

        rpn_cls_score_fpn, rpn_bbox_pred_fpn = self._get_output(body_feat,
                                                                feat_lvl)

        prop_op = self.train_proposal if mode == 'train' else self.test_proposal
        if self.num_classes == 1:
            rpn_cls_prob_fpn = fluid.layers.sigmoid(
                rpn_cls_score_fpn, name='rpn_cls_prob_fpn' + str(feat_lvl))
        else:
            rpn_cls_score_fpn = fluid.layers.transpose(
                rpn_cls_score_fpn, perm=[0, 2, 3, 1])
            rpn_cls_score_fpn = fluid.layers.reshape(
                rpn_cls_score_fpn, shape=(0, 0, 0, -1, self.num_classes))
            rpn_cls_prob_fpn = fluid.layers.softmax(
                rpn_cls_score_fpn,
                use_cudnn=False,
                name='rpn_cls_prob_fpn' + str(feat_lvl))
            rpn_cls_prob_fpn = fluid.layers.slice(
                rpn_cls_prob_fpn, axes=[4], starts=[1],
                ends=[self.num_classes])
            rpn_cls_prob_fpn, _ = fluid.layers.topk(rpn_cls_prob_fpn, 1)
            rpn_cls_prob_fpn = fluid.layers.reshape(
                rpn_cls_prob_fpn, shape=(0, 0, 0, -1))
            rpn_cls_prob_fpn = fluid.layers.transpose(
                rpn_cls_prob_fpn, perm=[0, 3, 1, 2])
        rpn_rois_fpn, rpn_roi_prob_fpn = prop_op(
            scores=rpn_cls_prob_fpn,
            bbox_deltas=rpn_bbox_pred_fpn,
            im_info=im_info,
            anchors=self.anchors,
            variances=self.anchor_var)
        return rpn_rois_fpn, rpn_roi_prob_fpn

    def get_proposals(self, fpn_feats, im_info, mode='train'):
        """
        Get proposals in multiple levels according to the output of fpn
        rpn head

        Args:
            fpn_feats(dict): A dictionary represents the output feature map
                of FPN with their name.
            im_info(Variable): The information of image with shape [N, 3] with
                format (height, width, scale).

        Return:
            rois_list(Variable): Output proposals in shape of [rois_num, 4]
        """
        rois_list = []
        roi_probs_list = []
        fpn_feat_names = list(fpn_feats.keys())
        for lvl in range(self.min_level, self.max_level + 1):
            fpn_feat_name = fpn_feat_names[self.max_level - lvl]
            fpn_feat = fpn_feats[fpn_feat_name]
            rois_fpn, roi_probs_fpn = self._get_single_proposals(
                fpn_feat, im_info, lvl, mode)
            self.fpn_rpn_list.append((self.rpn_cls_score, self.rpn_bbox_pred))
            rois_list.append(rois_fpn)
            roi_probs_list.append(roi_probs_fpn)
            self.anchors_list.append(self.anchors)
            self.anchor_var_list.append(self.anchor_var)
        prop_op = self.train_proposal if mode == 'train' else self.test_proposal
        post_nms_top_n = prop_op.post_nms_top_n
        rois_collect = fluid.layers.collect_fpn_proposals(
            rois_list,
            roi_probs_list,
            self.min_level,
            self.max_level,
            post_nms_top_n,
            name='collect')
        return rois_collect

    def _get_loss_input(self):
        rpn_clses = []
        rpn_bboxes = []
        anchors = []
        anchor_vars = []
        for i in range(len(self.fpn_rpn_list)):
            single_input = self._transform_input(
                self.fpn_rpn_list[i][0], self.fpn_rpn_list[i][1],
                self.anchors_list[i], self.anchor_var_list[i])
            rpn_clses.append(single_input[0])
            rpn_bboxes.append(single_input[1])
            anchors.append(single_input[2])
            anchor_vars.append(single_input[3])

        rpn_cls = fluid.layers.concat(rpn_clses, axis=1)
        rpn_bbox = fluid.layers.concat(rpn_bboxes, axis=1)
        anchors = fluid.layers.concat(anchors)
        anchor_var = fluid.layers.concat(anchor_vars)
        return rpn_cls, rpn_bbox, anchors, anchor_var
