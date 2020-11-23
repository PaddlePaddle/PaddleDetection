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
from paddle import ParamAttr
from paddle.nn.initializer import Normal
from paddle.regularizer import L2Decay
from paddle.nn import Conv2D

from ppdet.core.workspace import register
from ppdet.modeling import ops


@register
class RPNFeat(nn.Layer):
    def __init__(self, feat_in=1024, feat_out=1024):
        super(RPNFeat, self).__init__()
        # rpn feat is shared with each level
        self.rpn_conv = Conv2D(
            in_channels=feat_in,
            out_channels=feat_out,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.)))

    def forward(self, inputs, feats):
        rpn_feats = []
        for feat in feats:
            rpn_feats.append(F.relu(self.rpn_conv(feat)))
        return rpn_feats


@register
class RPNHead(nn.Layer):
    __inject__ = ['rpn_feat']

    def __init__(self, rpn_feat, anchor_per_position=15, rpn_channel=1024):
        super(RPNHead, self).__init__()
        self.rpn_feat = rpn_feat
        if isinstance(rpn_feat, dict):
            self.rpn_feat = RPNFeat(**rpn_feat)
        # rpn head is shared with each level
        # rpn roi classification scores
        self.rpn_rois_score = Conv2D(
            in_channels=rpn_channel,
            out_channels=anchor_per_position,
            kernel_size=1,
            padding=0,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.)))

        # rpn roi bbox regression deltas
        self.rpn_rois_delta = Conv2D(
            in_channels=rpn_channel,
            out_channels=4 * anchor_per_position,
            kernel_size=1,
            padding=0,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.)))

    def forward(self, inputs, feats):
        rpn_feats = self.rpn_feat(inputs, feats)
        rpn_head_out = []
        for rpn_feat in rpn_feats:
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            rpn_head_out.append((rrs, rrd))
        return rpn_feats, rpn_head_out

    def get_loss(self, loss_inputs):
        # cls loss
        score_tgt = paddle.cast(
            x=loss_inputs['rpn_score_target'], dtype='float32')
        score_tgt.stop_gradient = True
        loss_rpn_cls = ops.sigmoid_cross_entropy_with_logits(
            input=loss_inputs['rpn_score_pred'], label=score_tgt)
        loss_rpn_cls = paddle.mean(loss_rpn_cls, name='loss_rpn_cls')

        # reg loss
        loc_tgt = paddle.cast(x=loss_inputs['rpn_rois_target'], dtype='float32')
        loc_tgt.stop_gradient = True
        loss_rpn_reg = ops.smooth_l1(
            input=loss_inputs['rpn_rois_pred'],
            label=loc_tgt,
            inside_weight=loss_inputs['rpn_rois_weight'],
            outside_weight=loss_inputs['rpn_rois_weight'],
            sigma=3.0, )
        loss_rpn_reg = paddle.sum(loss_rpn_reg)
        score_shape = paddle.shape(score_tgt)
        score_shape = paddle.cast(score_shape, dtype='float32')
        norm = paddle.prod(score_shape)
        norm.stop_gradient = True
        loss_rpn_reg = loss_rpn_reg / norm

        return {'loss_rpn_cls': loss_rpn_cls, 'loss_rpn_reg': loss_rpn_reg}
