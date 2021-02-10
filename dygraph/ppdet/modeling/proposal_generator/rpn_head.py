# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling import ops

from .anchor_generator import AnchorGenerator
from .target_layer import RPNTargetAssign
from .proposal_generator import ProposalGenerator


class RPNFeat(nn.Layer):
    def __init__(self, feat_in=1024, feat_out=1024):
        super(RPNFeat, self).__init__()
        # rpn feat is shared with each level
        self.rpn_conv = nn.Conv2D(
            in_channels=feat_in,
            out_channels=feat_out,
            kernel_size=3,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))

    def forward(self, feats):
        rpn_feats = []
        for feat in feats:
            rpn_feats.append(F.relu(self.rpn_conv(feat)))
        return rpn_feats


@register
class RPNHead(nn.Layer):
    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 train_proposal=ProposalGenerator(12000, 2000).__dict__,
                 test_proposal=ProposalGenerator().__dict__,
                 in_channel=1024):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = ProposalGenerator(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = ProposalGenerator(**test_proposal)

        num_anchors = self.anchor_generator.num_anchors
        self.rpn_feat = RPNFeat(in_channel, in_channel)
        # rpn head is shared with each level
        # rpn roi classification scores
        self.rpn_rois_score = nn.Conv2D(
            in_channels=in_channel,
            out_channels=num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))

        # rpn roi bbox regression deltas
        self.rpn_rois_delta = nn.Conv2D(
            in_channels=in_channel,
            out_channels=4 * num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))

    @classmethod
    def from_config(cls, cfg, input_shape):
        # FPN share same rpn head
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def forward(self, feats, inputs):
        rpn_feats = self.rpn_feat(feats)
        scores = []
        deltas = []

        for rpn_feat in rpn_feats:
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            scores.append(rrs)
            deltas.append(rrd)

        anchors = self.anchor_generator(rpn_feats)

        # TODO: Fix batch_size > 1 when testing.
        if self.training:
            batch_size = inputs['im_shape'].shape[0]
        else:
            batch_size = 1

        rois, rois_num = self._gen_proposal(scores, deltas, anchors, inputs,
                                            batch_size)
        if self.training:
            loss = self.get_loss(scores, deltas, anchors, inputs)
            return rois, rois_num, loss
        else:
            return rois, rois_num, None

    def _gen_proposal(self, scores, bbox_deltas, anchors, inputs, batch_size):
        """
        scores (list[Tensor]): Multi-level scores prediction
        bbox_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info
        """
        prop_gen = self.train_proposal if self.training else self.test_proposal
        im_shape = inputs['im_shape']
        rpn_rois_list = [[] for i in range(batch_size)]
        rpn_prob_list = [[] for i in range(batch_size)]
        rpn_rois_num_list = [[] for i in range(batch_size)]
        # Generate proposals for each level and each batch.
        # Discard batch-computing to avoid sorting bbox cross different batches.
        for rpn_score, rpn_delta, anchor in zip(scores, bbox_deltas, anchors):
            for i in range(batch_size):
                rpn_rois, rpn_rois_prob, rpn_rois_num, post_nms_top_n = prop_gen(
                    scores=rpn_score[i:i + 1],
                    bbox_deltas=rpn_delta[i:i + 1],
                    anchors=anchor,
                    im_shape=im_shape[i:i + 1])
                if rpn_rois.shape[0] > 0:
                    rpn_rois_list[i].append(rpn_rois)
                    rpn_prob_list[i].append(rpn_rois_prob)
                    rpn_rois_num_list[i].append(rpn_rois_num)

        # Collect multi-level proposals for each batch 
        # Get 'topk' of them as final output 
        rois_collect = []
        rois_num_collect = []
        for i in range(batch_size):
            if len(scores) > 1:
                rpn_rois = paddle.concat(rpn_rois_list[i])
                rpn_prob = paddle.concat(rpn_prob_list[i]).flatten()
                if rpn_prob.shape[0] > post_nms_top_n:
                    topk_prob, topk_inds = paddle.topk(rpn_prob, post_nms_top_n)
                    topk_rois = paddle.gather(rpn_rois, topk_inds)
                else:
                    topk_rois = rpn_rois
                    topk_prob = rpn_prob
            else:
                topk_rois = rpn_rois_list[i][0]
                topk_prob = rpn_prob_list[i][0].flatten()
            rois_collect.append(topk_rois)
            rois_num_collect.append(paddle.shape(topk_rois)[0])
        rois_num_collect = paddle.concat(rois_num_collect)
        return rois_collect, rois_num_collect

    def get_loss(self, pred_scores, pred_deltas, anchors, inputs):
        """
        pred_scores (list[Tensor]): Multi-level scores prediction 
        pred_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info, including im, gt_bbox, gt_score
        """
        anchors = [paddle.reshape(a, shape=(-1, 4)) for a in anchors]
        anchors = paddle.concat(anchors)

        scores = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 1)) for v in pred_scores
        ]
        scores = paddle.concat(scores, axis=1)

        deltas = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 4)) for v in pred_deltas
        ]
        deltas = paddle.concat(deltas, axis=1)

        score_tgt, bbox_tgt, loc_tgt, norm = self.rpn_target_assign(inputs,
                                                                    anchors)

        scores = paddle.reshape(x=scores, shape=(-1, ))
        deltas = paddle.reshape(x=deltas, shape=(-1, 4))

        score_tgt = paddle.concat(score_tgt)
        score_tgt.stop_gradient = True

        pos_mask = score_tgt == 1
        pos_ind = paddle.nonzero(pos_mask)

        valid_mask = score_tgt >= 0
        valid_ind = paddle.nonzero(valid_mask)

        # cls loss
        score_pred = paddle.gather(scores, valid_ind)
        score_label = paddle.gather(score_tgt, valid_ind).cast('float32')
        score_label.stop_gradient = True
        loss_rpn_cls = F.binary_cross_entropy_with_logits(
            logit=score_pred, label=score_label, reduction="sum")

        # reg loss
        loc_pred = paddle.gather(deltas, pos_ind)
        loc_tgt = paddle.concat(loc_tgt)
        loc_tgt = paddle.gather(loc_tgt, pos_ind)
        loc_tgt.stop_gradient = True
        loss_rpn_reg = paddle.abs(loc_pred - loc_tgt).sum()
        return {
            'loss_rpn_cls': loss_rpn_cls / norm,
            'loss_rpn_reg': loss_rpn_reg / norm
        }
