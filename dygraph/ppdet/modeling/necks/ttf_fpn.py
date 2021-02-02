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
from paddle.nn.initializer import Constant, Uniform, Normal
from paddle.nn import Conv2D, ReLU, Sequential
from paddle import ParamAttr
from ppdet.core.workspace import register, serializable
from paddle.regularizer import L2Decay
from ppdet.modeling.layers import DeformableConvV2
import math
from ppdet.modeling.ops import batch_norm
from ..shape_spec import ShapeSpec

__all__ = ['TTFFPN']

__all__ = ['TTFFPN']


class Upsample(nn.Layer):
    def __init__(self, ch_in, ch_out, name=None):
        super(Upsample, self).__init__()
        fan_in = ch_in * 3 * 3
        stdv = 1. / math.sqrt(fan_in)
        self.dcn = DeformableConvV2(
            ch_in,
            ch_out,
            kernel_size=3,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(
                initializer=Constant(0),
                regularizer=L2Decay(0.),
                learning_rate=2.),
            lr_scale=2.,
            regularizer=L2Decay(0.),
            name=name)

        self.bn = batch_norm(
            ch_out, norm_type='bn', initializer=Constant(1.), name=name)

    def forward(self, feat):
        dcn = self.dcn(feat)
        bn = self.bn(dcn)
        relu = F.relu(bn)
        out = F.interpolate(relu, scale_factor=2., mode='bilinear')
        return out


class ShortCut(nn.Layer):
    def __init__(self, layer_num, ch_out, name=None):
        super(ShortCut, self).__init__()
        shortcut_conv = Sequential()
        ch_in = ch_out * 2
        for i in range(layer_num):
            fan_out = 3 * 3 * ch_out
            std = math.sqrt(2. / fan_out)
            in_channels = ch_in if i == 0 else ch_out
            shortcut_name = name + '.conv.{}'.format(i)
            shortcut_conv.add_sublayer(
                shortcut_name,
                Conv2D(
                    in_channels=in_channels,
                    out_channels=ch_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0, std)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            if i < layer_num - 1:
                shortcut_conv.add_sublayer(shortcut_name + '.act', ReLU())
        self.shortcut = self.add_sublayer('short', shortcut_conv)

    def forward(self, feat):
        out = self.shortcut(feat)
        return out


@register
@serializable
class TTFFPN(nn.Layer):
    """
    Args:
        in_channels (list): number of input feature channels from backbone.
            [128,256,512,1024] by default, means the channels of DarkNet53
            backbone return_idx [1,2,3,4].
        shortcut_num (list): the number of convolution layers in each shortcut.
            [3,2,1] by default, means DarkNet53 backbone return_idx_1 has 3 convs
            in its shortcut, return_idx_2 has 2 convs and return_idx_3 has 1 conv.
    """

    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 shortcut_num=[3, 2, 1]):
        super(TTFFPN, self).__init__()
        self.planes = [c // 2 for c in in_channels[:-1]][::-1]
        self.shortcut_num = shortcut_num[::-1]
        self.shortcut_len = len(shortcut_num)
        self.ch_in = in_channels[::-1]

        self.upsample_list = []
        self.shortcut_list = []
        for i, out_c in enumerate(self.planes):
            in_c = self.ch_in[i] if i == 0 else self.ch_in[i] // 2
            upsample = self.add_sublayer(
                'upsample.' + str(i),
                Upsample(
                    in_c, out_c, name='upsample.' + str(i)))
            self.upsample_list.append(upsample)
            if i < self.shortcut_len:
                shortcut = self.add_sublayer(
                    'shortcut.' + str(i),
                    ShortCut(
                        self.shortcut_num[i], out_c, name='shortcut.' + str(i)))
                self.shortcut_list.append(shortcut)

    def forward(self, inputs):
        feat = inputs[-1]
        for i, out_c in enumerate(self.planes):
            feat = self.upsample_list[i](feat)
            if i < self.shortcut_len:
                shortcut = self.shortcut_list[i](inputs[-i - 2])
                feat = feat + shortcut
        return feat

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.planes[-1], )]
