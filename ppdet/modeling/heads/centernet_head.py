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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingUniform
from ppdet.core.workspace import register
from ppdet.modeling.losses import CTFocalLoss


class ConvLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(ConvLayer, self).__init__()
        bias_attr = False
        fan_in = ch_in * kernel_size**2
        bound = 1 / math.sqrt(fan_in)
        param_attr = paddle.ParamAttr(initializer=KaimingUniform())
        if bias:
            bias_attr = paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-bound, bound))
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=param_attr,
            bias_attr=bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)

        return out


@register
class CenterNetHead(nn.Layer):
    """
    Args:
        in_channels (int): the channel number of input to CenterNetHead.
        num_classes (int): the number of classes, 80 by default.
        head_planes (int): the channel number in all head, 256 by default.
        heatmap_weight (float): the weight of heatmap loss, 1 by default.
        regress_ltrb (bool): whether to regress left/top/right/bottom or
            width/height for a box, true by default
        size_weight (float): the weight of box size loss, 0.1 by default.
        offset_weight (float): the weight of center offset loss, 1 by default.

    """

    __shared__ = ['num_classes']

    def __init__(self,
                 in_channels,
                 num_classes=80,
                 head_planes=256,
                 heatmap_weight=1,
                 regress_ltrb=True,
                 size_weight=0.1,
                 offset_weight=1):
        super(CenterNetHead, self).__init__()
        self.weights = {
            'heatmap': heatmap_weight,
            'size': size_weight,
            'offset': offset_weight
        }
        self.heatmap = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        self.heatmap[2].conv.bias[:] = -2.19
        self.size = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                4 if regress_ltrb else 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        self.offset = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                head_planes, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.focal_loss = CTFocalLoss()

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self, feat, inputs):
        heatmap = self.heatmap(feat)
        size = self.size(feat)
        offset = self.offset(feat)
        if self.training:
            loss = self.get_loss(heatmap, size, offset, self.weights, inputs)
            return loss
        else:
            heatmap = F.sigmoid(heatmap)
            return {'heatmap': heatmap, 'size': size, 'offset': offset}

    def get_loss(self, heatmap, size, offset, weights, inputs):
        heatmap_target = inputs['heatmap']
        size_target = inputs['size']
        offset_target = inputs['offset']
        index = inputs['index']
        mask = inputs['index_mask']
        heatmap = paddle.clip(F.sigmoid(heatmap), 1e-4, 1 - 1e-4)
        heatmap_loss = self.focal_loss(heatmap, heatmap_target)

        size = paddle.transpose(size, perm=[0, 2, 3, 1])
        size_n, size_h, size_w, size_c = size.shape
        size = paddle.reshape(size, shape=[size_n, -1, size_c])
        index = paddle.unsqueeze(index, 2)
        batch_inds = list()
        for i in range(size_n):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)
        pos_size = paddle.gather_nd(size, index=index)
        mask = paddle.unsqueeze(mask, axis=2)
        size_mask = paddle.expand_as(mask, pos_size)
        size_mask = paddle.cast(size_mask, dtype=pos_size.dtype)
        pos_num = size_mask.sum()
        size_mask.stop_gradient = True
        size_target.stop_gradient = True
        size_loss = F.l1_loss(
            pos_size * size_mask, size_target * size_mask, reduction='sum')
        size_loss = size_loss / (pos_num + 1e-4)

        offset = paddle.transpose(offset, perm=[0, 2, 3, 1])
        offset_n, offset_h, offset_w, offset_c = offset.shape
        offset = paddle.reshape(offset, shape=[offset_n, -1, offset_c])
        pos_offset = paddle.gather_nd(offset, index=index)
        offset_mask = paddle.expand_as(mask, pos_offset)
        offset_mask = paddle.cast(offset_mask, dtype=pos_offset.dtype)
        pos_num = offset_mask.sum()
        offset_mask.stop_gradient = True
        offset_target.stop_gradient = True
        offset_loss = F.l1_loss(
            pos_offset * offset_mask,
            offset_target * offset_mask,
            reduction='sum')
        offset_loss = offset_loss / (pos_num + 1e-4)

        det_loss = weights['heatmap'] * heatmap_loss + weights[
            'size'] * size_loss + weights['offset'] * offset_loss

        return {
            'det_loss': det_loss,
            'heatmap_loss': heatmap_loss,
            'size_loss': size_loss,
            'offset_loss': offset_loss
        }
