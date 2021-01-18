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
from paddle.nn import Conv2D, ReLU, Sequential
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Uniform, Normal
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
import numpy as np


@register
class HMHead(nn.Layer):

    __shared__ = ['num_classes']

    def __init__(self, ch_in, ch_out=128, num_classes=80, conv_num=2):
        super(HMHead, self).__init__()
        head_conv = Sequential()
        for i in range(conv_num):
            name = 'conv.{}'.format(i)
            head_conv.add_sublayer(
                name,
                Conv2D(
                    in_channels=ch_in if i == 0 else ch_out,
                    out_channels=ch_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0, 0.01)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            head_conv.add_sublayer(name + '.act', ReLU())
        self.feat = self.add_sublayer('hm_feat', head_conv)
        bias_init = float(-np.log((1 - 0.01) / 0.01))
        self.head = self.add_sublayer(
            'hm_head',
            Conv2D(
                in_channels=ch_out,
                out_channels=num_classes,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=Normal(0, 0.01)),
                bias_attr=ParamAttr(
                    learning_rate=2.,
                    regularizer=L2Decay(0.),
                    initializer=Constant(bias_init))))

    def forward(self, feat):
        out = self.feat(feat)
        out = self.head(out)
        return out


@register
class WHHead(nn.Layer):
    def __init__(self, ch_in, ch_out=64, conv_num=2):
        super(WHHead, self).__init__()
        head_conv = Sequential()
        for i in range(conv_num):
            name = 'conv.{}'.format(i)
            head_conv.add_sublayer(
                name,
                Conv2D(
                    in_channels=ch_in if i == 0 else ch_out,
                    out_channels=ch_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0, 0.001)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            head_conv.add_sublayer(name + '.act', ReLU())
        self.feat = self.add_sublayer('wh_feat', head_conv)
        self.head = self.add_sublayer(
            'wh_head',
            Conv2D(
                in_channels=ch_out,
                out_channels=4,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=Normal(0, 0.001)),
                bias_attr=ParamAttr(
                    learning_rate=2., regularizer=L2Decay(0.))))

    def forward(self, feat):
        out = self.feat(feat)
        out = self.head(out)
        out = F.relu(out)
        return out


@register
class TTFHead(nn.Layer):
    __shared__ = ['down_ratio']
    __inject__ = ['hm_head', 'wh_head', 'hm_loss', 'wh_loss']

    def __init__(self,
                 hm_head='HMHead',
                 wh_head='WHHead',
                 hm_loss='CTFocalLoss',
                 wh_loss='GIoULoss',
                 wh_offset_base=16.,
                 down_ratio=4):
        super(TTFHead, self).__init__()
        self.hm_head = hm_head
        self.wh_head = wh_head
        self.hm_loss = hm_loss
        self.wh_loss = wh_loss

        self.wh_offset_base = wh_offset_base
        self.down_ratio = down_ratio

    def forward(self, feats):
        hm = self.hm_head(feats)
        wh = self.wh_head(feats) * self.wh_offset_base
        return hm, wh

    def filter_box_by_weight(self, pred, target, weight):
        index = paddle.nonzero(weight > 0)
        index.stop_gradient = True
        weight = paddle.gather_nd(weight, index)
        pred = paddle.gather_nd(pred, index)
        target = paddle.gather_nd(target, index)
        return pred, target, weight

    def get_loss(self, pred_hm, pred_wh, target_hm, box_target, target_weight):
        pred_hm = paddle.clip(F.sigmoid(pred_hm), 1e-4, 1 - 1e-4)
        hm_loss = self.hm_loss(pred_hm, target_hm)
        H, W = target_hm.shape[2:]
        mask = paddle.reshape(target_weight, [-1, H, W])
        avg_factor = paddle.sum(mask) + 1e-4

        base_step = self.down_ratio
        shifts_x = paddle.arange(0, W * base_step, base_step, dtype='int32')
        shifts_y = paddle.arange(0, H * base_step, base_step, dtype='int32')
        shift_y, shift_x = paddle.tensor.meshgrid([shifts_y, shifts_x])
        base_loc = paddle.stack([shift_x, shift_y], axis=0)
        base_loc.stop_gradient = True

        pred_boxes = paddle.concat(
            [0 - pred_wh[:, 0:2, :, :] + base_loc, pred_wh[:, 2:4] + base_loc],
            axis=1)
        pred_boxes = paddle.transpose(pred_boxes, [0, 2, 3, 1])
        boxes = paddle.transpose(box_target, [0, 2, 3, 1])
        boxes.stop_gradient = True

        pred_boxes, boxes, mask = self.filter_box_by_weight(pred_boxes, boxes,
                                                            mask)
        mask.stop_gradient = True
        wh_loss = self.wh_loss(pred_boxes, boxes, iou_weight=mask.unsqueeze(1))
        wh_loss = wh_loss / avg_factor

        ttf_loss = {'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return ttf_loss
