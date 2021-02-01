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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from paddle.nn.initializer import Normal, Constant
from IPython import embed


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


class LossParam(nn.Layer):
    def __init__(self, init_value=0.):
        super(LossParam, self).__init__()
        self.loss_param = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=init_value)),
            dtype="float32")

    def forward(self, inputs):
        out = paddle.exp(-self.loss_param) * inputs + self.loss_param
        return out


@register
class JDEHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['jde_loss']
    """
    JDEHead
    Args:
        anchors(list): Anchor parameters.
        anchor_masks(list): Anchor parameters.
        num_classes(int): Number of classes. Only support one class tracking.
        num_identifiers(int): Number of identifiers.
        embedding_dim(int): Embedding dimension. Default: 512.
        jde_loss    : 
        img_size(list): Input size of JDE network.
        ide_thresh  : Identification positive threshold. Default: 0.5.
        obj_thresh  : Objectness positive threshold. Default: 0.5.
        bkg_thresh  : Background positive threshold. Default: 0.4.
        s_box       : Weight for the box regression task.
        s_cls       : Weight for the classification task.
        s_ide       : Weight for the identifier classification task.
    """

    def __init__(
            self,
            anchors=[[8, 24], [11, 34], [16, 48], [23, 68], [32, 96],
                     [45, 135], [64, 192], [90, 271], [128, 384], [180, 540],
                     [256, 640], [512, 640]],
            anchor_masks=[[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]],
            num_classes=1,
            num_identifiers=14455,
            embedding_dim=512,
            # jde_loss='JDELoss',
            jde_loss='YOLOv3Loss',  # todo
            img_size=[1888, 608],
            test_emb=False,
            iou_aware=False,
            iou_aware_factor=0.4):
        super(JDEHead, self).__init__()
        self.num_classes = num_classes
        self.num_identifiers = num_identifiers
        self.embedding_dim = embedding_dim
        self.jde_loss = jde_loss
        self.img_size = img_size
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.test_emb = test_emb

        self.shift = [1, 3, 5]
        self.emb_scale = math.sqrt(2) * math.log(
            self.num_identifiers - 1) if self.num_identifiers > 1 else 1

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)

        self.yolo_outputs = []
        self.identify_outputs = []
        self.loss_params_cls = []
        self.loss_params_reg = []
        self.loss_params_ide = []
        for i in range(len(self.anchors)):
            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            yolo_output = self.add_sublayer(
                name,
                nn.Conv2D(
                    in_channels=128 * (2**self.num_outputs) // (2**i),
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(name=name + '.conv.weights'),
                    bias_attr=ParamAttr(
                        name=name + '.conv.bias', regularizer=L2Decay(0.))))
            self.yolo_outputs.append(yolo_output)

            name = 'identify_output.{}'.format(i)
            identify_output = self.add_sublayer(
                name,
                nn.Conv2D(
                    in_channels=64 * (2**self.num_outputs) // (2**i),
                    out_channels=embedding_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(name=name + '.conv.weights'),
                    bias_attr=ParamAttr(
                        name=name + '.conv.bias', regularizer=L2Decay(0.))))
            self.identify_outputs.append(identify_output)

            loss_p_cls = self.add_sublayer('cls.{}'.format(i), LossParam(-4.15))
            self.loss_params_cls.append(loss_p_cls)
            loss_p_reg = self.add_sublayer('reg.{}'.format(i), LossParam(-4.85))
            self.loss_params_reg.append(loss_p_reg)
            loss_p_ide = self.add_sublayer('ide.{}'.format(i), LossParam(-2.3))
            self.loss_params_ide.append(loss_p_ide)

        self.classifier = self.add_sublayer(
            'classifier',
            nn.Linear(
                embedding_dim,
                num_identifiers,
                weight_attr=ParamAttr(
                    learning_rate=1., initializer=Normal(
                        mean=0.0, std=0.01)),
                bias_attr=ParamAttr(
                    learning_rate=2., regularizer=L2Decay(0.))))

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, yolo_feats, identify_feats):
        assert len(yolo_feats) == len(identify_feats) == len(self.anchors)
        det_outs = []
        ide_outs = []
        for yolo_head, ide_head, yolo_feat, ide_feat in zip(
                self.yolo_outputs, self.identify_outputs, yolo_feats,
                identify_feats):
            det_out = yolo_head(yolo_feat)
            ide_out = ide_head(ide_feat)
            det_outs.append(det_out)
            ide_outs.append(ide_out)
        return det_outs, ide_outs

    def get_loss(self, det_outs, ide_outs, targets):
        #jde_loss = self.jde_loss(det_outs, ide_outs, targets, self.anchors, self.emb_scale, self.test_emb) # todo jde_loss
        jde_loss = {}
        yolo_loss = self.jde_loss(det_outs, targets, self.anchors)
        return yolo_loss

    def get_outputs(self, det_outs, ide_outs):
        if self.iou_aware:
            y = []
            for i, out in enumerate(det_outs):
                na = len(self.anchors[i])
                ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                b, c, h, w = x.shape
                no = c // na
                x = x.reshape((b, na, no, h * w))
                ioup = ioup.reshape((b, na, 1, h * w))
                obj = x[:, :, 4:5, :]
                ioup = F.sigmoid(ioup)
                obj = F.sigmoid(obj)
                obj_t = (obj**(1 - self.iou_aware_factor)) * (
                    ioup**self.iou_aware_factor)
                obj_t = _de_sigmoid(obj_t)
                loc_t = x[:, :, :4, :]
                cls_t = x[:, :, 5:, :]
                y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                y_t = y_t.reshape((b, c, h, w))
                y.append(y_t)
            return y
        else:
            # ide_outpus =  # todo
            return det_outs
