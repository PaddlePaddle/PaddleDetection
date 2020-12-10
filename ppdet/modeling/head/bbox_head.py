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
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import ReLU
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops

from ..backbone.name_adapter import NameAdapter
from ..backbone.resnet import Blocks


@register
class TwoFCHead(nn.Layer):

    __shared__ = ['roi_stages']

    def __init__(self, in_dim=256, mlp_dim=1024, resolution=7, roi_stages=1):
        super(TwoFCHead, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.roi_stages = roi_stages
        fan = in_dim * resolution * resolution
        self.fc6_list = []
        self.fc6_relu_list = []
        self.fc7_list = []
        self.fc7_relu_list = []
        for stage in range(roi_stages):
            fc6_name = 'fc6_{}'.format(stage)
            fc7_name = 'fc7_{}'.format(stage)
            lr_factor = 2**stage
            fc6 = self.add_sublayer(
                fc6_name,
                nn.Linear(
                    in_dim * resolution * resolution,
                    mlp_dim,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor,
                        initializer=XavierUniform(fan_out=fan)),
                    bias_attr=ParamAttr(
                        learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))
            fc6_relu = self.add_sublayer(fc6_name + 'act', ReLU())
            fc7 = self.add_sublayer(
                fc7_name,
                nn.Linear(
                    mlp_dim,
                    mlp_dim,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor, initializer=XavierUniform()),
                    bias_attr=ParamAttr(
                        learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))
            fc7_relu = self.add_sublayer(fc7_name + 'act', ReLU())
            self.fc6_list.append(fc6)
            self.fc6_relu_list.append(fc6_relu)
            self.fc7_list.append(fc7)
            self.fc7_relu_list.append(fc7_relu)

    def forward(self, rois_feat, stage=0):
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6_list[stage](rois_feat)
        fc6_relu = self.fc6_relu_list[stage](fc6)
        fc7 = self.fc7_list[stage](fc6_relu)
        fc7_relu = self.fc7_relu_list[stage](fc7)
        return fc7_relu


@register
class Res5Head(nn.Layer):
    def __init__(self, feat_in=1024, feat_out=512):
        super(Res5Head, self).__init__()
        na = NameAdapter(self)
        self.res5_conv = []
        self.res5 = self.add_sublayer(
            'res5_roi_feat',
            Blocks(
                feat_in, feat_out, count=3, name_adapter=na, stage_num=5))
        self.feat_out = feat_out * 4

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y


@register
class BBoxFeat(nn.Layer):
    __inject__ = ['roi_extractor', 'head_feat']

    def __init__(self, roi_extractor, head_feat):
        super(BBoxFeat, self).__init__()
        self.roi_extractor = roi_extractor
        self.head_feat = head_feat
        self.rois_feat_list = []

    def forward(self, body_feats, rois, spatial_scale, stage=0):
        rois_feat = self.roi_extractor(body_feats, rois, spatial_scale)
        bbox_feat = self.head_feat(rois_feat, stage)
        return rois_feat, bbox_feat


@register
class BBoxHead(nn.Layer):
    __shared__ = ['num_classes', 'roi_stages']
    __inject__ = ['bbox_feat']

    def __init__(self,
                 bbox_feat,
                 in_feat=1024,
                 num_classes=81,
                 cls_agnostic=False,
                 roi_stages=1,
                 with_pool=False,
                 score_stage=[0, 1, 2],
                 delta_stage=[2]):
        super(BBoxHead, self).__init__()
        self.num_classes = num_classes
        self.cls_agnostic = cls_agnostic
        self.delta_dim = 2 if cls_agnostic else num_classes
        self.bbox_feat = bbox_feat
        self.roi_stages = roi_stages
        self.bbox_score_list = []
        self.bbox_delta_list = []
        self.roi_feat_list = [[] for i in range(roi_stages)]
        self.with_pool = with_pool
        self.score_stage = score_stage
        self.delta_stage = delta_stage
        for stage in range(roi_stages):
            score_name = 'bbox_score_{}'.format(stage)
            delta_name = 'bbox_delta_{}'.format(stage)
            lr_factor = 2**stage
            bbox_score = self.add_sublayer(
                score_name,
                nn.Linear(
                    in_feat,
                    1 * self.num_classes,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor,
                        initializer=Normal(
                            mean=0.0, std=0.01)),
                    bias_attr=ParamAttr(
                        learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))

            bbox_delta = self.add_sublayer(
                delta_name,
                nn.Linear(
                    in_feat,
                    4 * self.delta_dim,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor,
                        initializer=Normal(
                            mean=0.0, std=0.001)),
                    bias_attr=ParamAttr(
                        learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))
            self.bbox_score_list.append(bbox_score)
            self.bbox_delta_list.append(bbox_delta)

    def forward(self,
                body_feats=None,
                rois=None,
                spatial_scale=None,
                stage=0,
                roi_stage=-1):
        if rois is not None:
            rois_feat, bbox_feat = self.bbox_feat(body_feats, rois,
                                                  spatial_scale, stage)
            self.roi_feat_list[stage] = rois_feat
        else:
            rois_feat = self.roi_feat_list[roi_stage]
            bbox_feat = self.bbox_feat.head_feat(rois_feat, stage)
        if self.with_pool:
            bbox_feat_ = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            bbox_feat_ = paddle.squeeze(bbox_feat_, axis=[2, 3])
            scores = self.bbox_score_list[stage](bbox_feat_)
            deltas = self.bbox_delta_list[stage](bbox_feat_)
        else:
            scores = self.bbox_score_list[stage](bbox_feat)
            deltas = self.bbox_delta_list[stage](bbox_feat)
        bbox_head_out = (scores, deltas)
        return bbox_feat, bbox_head_out, self.bbox_feat.head_feat

    def _get_head_loss(self, score, delta, target):
        # bbox cls  
        labels_int64 = paddle.cast(x=target['labels_int32'], dtype='int64')
        labels_int64.stop_gradient = True
        loss_bbox_cls = F.softmax_with_cross_entropy(
            logits=score, label=labels_int64)
        loss_bbox_cls = paddle.mean(loss_bbox_cls)
        # bbox reg
        loss_bbox_reg = ops.smooth_l1(
            input=delta,
            label=target['bbox_targets'],
            inside_weight=target['bbox_inside_weights'],
            outside_weight=target['bbox_outside_weights'],
            sigma=1.0)
        loss_bbox_reg = paddle.mean(loss_bbox_reg)
        return loss_bbox_cls, loss_bbox_reg

    def get_loss(self, bbox_head_out, targets):
        loss_bbox = {}
        for lvl, (bboxhead, target) in enumerate(zip(bbox_head_out, targets)):
            score, delta = bboxhead
            cls_name = 'loss_bbox_cls_{}'.format(lvl)
            reg_name = 'loss_bbox_reg_{}'.format(lvl)
            loss_bbox_cls, loss_bbox_reg = self._get_head_loss(score, delta,
                                                               target)
            loss_weight = 1. / 2**lvl
            loss_bbox[cls_name] = loss_bbox_cls * loss_weight
            loss_bbox[reg_name] = loss_bbox_reg * loss_weight
        return loss_bbox

    def get_prediction(self, bbox_head_out, rois):
        proposal, proposal_num = rois
        score, delta = bbox_head_out
        bbox_prob = F.softmax(score)
        delta = paddle.reshape(delta, (-1, self.delta_dim, 4))
        bbox_pred = (delta, bbox_prob)
        return bbox_pred, rois

    def get_cascade_prediction(self, bbox_head_out, rois):
        proposal_list = []
        prob_list = []
        delta_list = []
        for stage in range(len(rois)):
            proposals = rois[stage]
            bboxhead = bbox_head_out[stage]
            score, delta = bboxhead
            proposal, proposal_num = proposals
            if stage in self.score_stage:
                if stage < 2:
                    _, head_out, _ = self(stage=stage, roi_stage=-1)
                    score = head_out[0]

                bbox_prob = F.softmax(score)
                prob_list.append(bbox_prob)
            if stage in self.delta_stage:
                proposal_list.append(proposal)
                delta_list.append(delta)
        bbox_prob = paddle.mean(paddle.stack(prob_list), axis=0)
        delta = paddle.mean(paddle.stack(delta_list), axis=0)
        proposal = paddle.mean(paddle.stack(proposal_list), axis=0)
        delta = paddle.reshape(delta, (-1, self.delta_dim, 4))
        if self.cls_agnostic:
            N, C, M = delta.shape
            delta = delta[:, 1:2, :]
            delta = paddle.expand(delta, [N, self.num_classes, M])
        bboxes = (proposal, proposal_num)
        bbox_pred = (delta, bbox_prob)
        return bbox_pred, bboxes
