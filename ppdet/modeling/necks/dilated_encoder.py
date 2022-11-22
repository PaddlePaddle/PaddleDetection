# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingUniform, Constant, Normal
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['DilatedEncoder']


class Bottleneck(nn.Layer):
    def __init__(self, in_channels, mid_channels, dilation):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(* [
            nn.Conv2D(
                in_channels,
                mid_channels,
                1,
                padding=0,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(0.0))),
            nn.BatchNorm2D(
                mid_channels,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            nn.ReLU(),
        ])
        self.conv2 = nn.Sequential(* [
            nn.Conv2D(
                mid_channels,
                mid_channels,
                3,
                padding=dilation,
                dilation=dilation,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(0.0))),
            nn.BatchNorm2D(
                mid_channels,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            nn.ReLU(),
        ])
        self.conv3 = nn.Sequential(* [
            nn.Conv2D(
                mid_channels,
                in_channels,
                1,
                padding=0,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(0.0))),
            nn.BatchNorm2D(
                in_channels,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            nn.ReLU(),
        ])

    def forward(self, x):
        identity = x
        y = self.conv3(self.conv2(self.conv1(x)))
        return y + identity


@register
class DilatedEncoder(nn.Layer):
    """
    DilatedEncoder used in YOLOF
    """

    def __init__(self,
                 in_channels=[2048],
                 out_channels=[512],
                 block_mid_channels=128,
                 num_residual_blocks=4,
                 block_dilations=[2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert len(self.in_channels) == 1, "YOLOF only has one level feature."
        assert len(self.out_channels) == 1, "YOLOF only has one level feature."

        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations

        out_ch = self.out_channels[0]
        self.lateral_conv = nn.Conv2D(
            self.in_channels[0],
            out_ch,
            1,
            weight_attr=ParamAttr(initializer=KaimingUniform(
                negative_slope=1, nonlinearity='leaky_relu')),
            bias_attr=ParamAttr(initializer=Constant(value=0.0)))
        self.lateral_norm = nn.BatchNorm2D(
            out_ch,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        self.fpn_conv = nn.Conv2D(
            out_ch,
            out_ch,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=KaimingUniform(
                negative_slope=1, nonlinearity='leaky_relu')))
        self.fpn_norm = nn.BatchNorm2D(
            out_ch,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            encoder_blocks.append(
                Bottleneck(
                    out_ch,
                    self.block_mid_channels,
                    dilation=block_dilations[i]))
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def forward(self, inputs, for_mot=False):
        out = self.lateral_norm(self.lateral_conv(inputs[0]))
        out = self.fpn_norm(self.fpn_conv(out))
        out = self.dilated_encoder_blocks(out)
        return [out]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self.out_channels]
