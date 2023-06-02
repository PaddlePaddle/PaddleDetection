# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is based on https://github.com/PeizeSun/SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py
Ths copyright of PeizeSun/SparseR-CNN is as follows:
MIT License [see LICENSE for details]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import copy
import paddle
import paddle.nn as nn

from ppdet.core.workspace import register
from ppdet.modeling.heads.roi_extractor import RoIAlign
from ppdet.modeling.bbox_utils import delta2bbox
from .. import initializer as init

_DEFAULT_SCALE_CLAMP = math.log(100000. / 16)


class DynamicConv(nn.Layer):
    def __init__(
            self,
            head_hidden_dim,
            head_dim_dynamic,
            head_num_dynamic, ):
        super().__init__()

        self.hidden_dim = head_hidden_dim
        self.dim_dynamic = head_dim_dynamic
        self.num_dynamic = head_num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim,
                                       self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU()

        pooler_resolution = 7
        num_output = self.hidden_dim * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.transpose(perm=[1, 0, 2])
        parameters = self.dynamic_layer(pro_features).transpose(perm=[1, 0, 2])

        param1 = parameters[:, :, :self.num_params].reshape(
            [-1, self.hidden_dim, self.dim_dynamic])
        param2 = parameters[:, :, self.num_params:].reshape(
            [-1, self.dim_dynamic, self.hidden_dim])

        features = paddle.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = paddle.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class RCNNHead(nn.Layer):
    def __init__(
            self,
            d_model,
            num_classes,
            dim_feedforward,
            nhead,
            dropout,
            head_cls,
            head_reg,
            head_dim_dynamic,
            head_num_dynamic,
            scale_clamp: float=_DEFAULT_SCALE_CLAMP,
            bbox_weights=(2.0, 2.0, 1.0, 1.0), ):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(d_model, head_dim_dynamic,
                                         head_num_dynamic)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # cls.
        num_cls = head_cls
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, bias_attr=False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU())
        self.cls_module = nn.LayerList(cls_module)

        # reg.
        num_reg = head_reg
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, bias_attr=False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU())
        self.reg_module = nn.LayerList(reg_module)

        # pred.
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(bboxes[b])
        roi_num = paddle.full([N], nr_boxes).astype("int32")

        roi_features = pooler(features, proposal_boxes, roi_num)
        roi_features = roi_features.reshape(
            [N * nr_boxes, self.d_model, -1]).transpose(perm=[2, 0, 1])

        # self_att.
        pro_features = pro_features.reshape([N, nr_boxes, self.d_model])
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features)
        pro_features = pro_features.transpose(perm=[1, 0, 2]) + self.dropout1(
            pro_features2.transpose(perm=[1, 0, 2]))
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.reshape(
            [nr_boxes, N, self.d_model]).transpose(perm=[1, 0, 2]).reshape(
                [1, N * nr_boxes, self.d_model])
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(perm=[1, 0, 2]).reshape(
            [N * nr_boxes, -1])
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = delta2bbox(bboxes_deltas,
                                 bboxes.reshape([-1, 4]), self.bbox_weights)

        return class_logits.reshape([N, nr_boxes, -1]), pred_bboxes.reshape(
            [N, nr_boxes, -1]), obj_features


@register
class SparseRCNNHead(nn.Layer):
    '''
    SparsercnnHead
    Args:
        roi_input_shape (list[ShapeSpec]): The output shape of fpn
        num_classes (int): Number of classes,
        head_hidden_dim (int): The param of MultiHeadAttention,
        head_dim_feedforward (int): The param of MultiHeadAttention,
        nhead (int): The param of MultiHeadAttention,
        head_dropout (float): The p of dropout,
        head_cls (int): The number of class head,
        head_reg (int): The number of regressionhead,
        head_num_dynamic (int): The number of DynamicConv's param,
        head_num_heads (int): The number of RCNNHead,
        deep_supervision (int): wheather supervise the intermediate results,
        num_proposals (int): the number of proposals boxes and features
    '''
    __inject__ = ['loss_func']
    __shared__ = ['num_classes']

    def __init__(
            self,
            head_hidden_dim,
            head_dim_feedforward,
            nhead,
            head_dropout,
            head_cls,
            head_reg,
            head_dim_dynamic,
            head_num_dynamic,
            head_num_heads,
            deep_supervision,
            num_proposals,
            num_classes=80,
            loss_func="SparseRCNNLoss",
            roi_input_shape=None, ):
        super().__init__()
        assert head_num_heads > 0, \
            f'At least one RoI Head is required, but {head_num_heads}.'

        # Build RoI.
        box_pooler = self._init_box_pooler(roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        rcnn_head = RCNNHead(
            head_hidden_dim,
            num_classes,
            head_dim_feedforward,
            nhead,
            head_dropout,
            head_cls,
            head_reg,
            head_dim_dynamic,
            head_num_dynamic, )
        self.head_series = nn.LayerList(
            [copy.deepcopy(rcnn_head) for i in range(head_num_heads)])
        self.return_intermediate = deep_supervision

        self.num_classes = num_classes

        # build init proposal
        self.init_proposal_features = nn.Embedding(num_proposals,
                                                   head_hidden_dim)
        self.init_proposal_boxes = nn.Embedding(num_proposals, 4)

        self.lossfunc = loss_func

        # Init parameters.
        init.reset_initialized_parameter(self)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, reverse=True)
            elif not isinstance(m, nn.Embedding) and hasattr(
                    m, "weight") and m.weight.dim() > 1:
                init.xavier_normal_(m.weight, reverse=False)

            if hasattr(m, "bias") and m.bias is not None and m.bias.shape[
                    -1] == self.num_classes:
                init.constant_(m.bias, bias_value)

        init_bboxes = paddle.empty_like(self.init_proposal_boxes.weight)
        init_bboxes[:, :2] = 0.5
        init_bboxes[:, 2:] = 1.0
        self.init_proposal_boxes.weight.set_value(init_bboxes)

    @staticmethod
    def _init_box_pooler(input_shape):

        pooler_resolution = 7
        sampling_ratio = 2

        if input_shape is not None:
            pooler_scales = tuple(1.0 / input_shape[k].stride
                                  for k in range(len(input_shape)))
            in_channels = [
                input_shape[f].channels for f in range(len(input_shape))
            ]
            end_level = len(input_shape) - 1
            # Check all channel counts are equal
            assert len(set(in_channels)) == 1, in_channels
        else:
            pooler_scales = [1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0]
            end_level = 3

        aligned = True
        if paddle.device.is_compiled_with_custom_device('npu'):
            aligned = False
        box_pooler = RoIAlign(
            resolution=pooler_resolution,
            spatial_scale=pooler_scales,
            sampling_ratio=sampling_ratio,
            end_level=end_level,
            aligned=aligned)
        return box_pooler

    def forward(self, features, input_whwh):

        bs = len(features[0])
        bboxes = box_cxcywh_to_xyxy(self.init_proposal_boxes.weight.clone(
        )).unsqueeze(0)
        bboxes = bboxes * input_whwh.unsqueeze(-2)

        init_features = self.init_proposal_features.weight.unsqueeze(0).tile(
            [1, bs, 1])
        proposal_features = init_features.clone()

        inter_class_logits = []
        inter_pred_bboxes = []

        for stage, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler)

            if self.return_intermediate or stage == len(self.head_series) - 1:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        output = {
            'pred_logits': inter_class_logits[-1],
            'pred_boxes': inter_pred_bboxes[-1]
        }
        if self.return_intermediate:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(inter_class_logits[:-1], inter_pred_bboxes[:-1])]

        return output

    def get_loss(self, outputs, targets):
        losses = self.lossfunc(outputs, targets)
        weight_dict = self.lossfunc.weight_dict

        for k in losses.keys():
            if k in weight_dict:
                losses[k] *= weight_dict[k]

        return losses


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)
