# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import KaimingNormal
from ppdet.core.workspace import register, serializable
from .name_adapter import NameAdapter
from numbers import Integral


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_groups=1,
                 act='relu',
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = nn.BatchNorm2D(out_channels)
        '''
        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name + "_bn_scale"),
            bias_attr=ParamAttr(name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")
        '''

        if act == 'relu':
            self._act = nn.ReLU()
        elif act == 'relu6':
            self._act = nn.ReLU6()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._act(x)
        return x


class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels1,
                 out_channels2,
                 num_groups,
                 stride,
                 scale,
                 name=None):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            in_channels,
            int(out_channels1 * scale),
            kernel_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            name=name + "_dw")

        self._pointwise_conv = ConvBNLayer(
            int(out_channels1 * scale),
            int(out_channels2 * scale),
            kernel_size=1,
            stride=1,
            padding=0,
            name=name + "_sep")

    def forward(self, x):
        x = self._depthwise_conv(x)
        x = self._pointwise_conv(x)
        return x


class ExtraBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels1,
                 out_channels2,
                 num_groups=1,
                 stride=2,
                 name=None):
        super(ExtraBlock, self).__init__()

        self.pointwise_conv = ConvBNLayer(
            in_channels,
            int(out_channels1),
            kernel_size=1,
            stride=1,
            padding=0,
            num_groups=int(num_groups),
            act='relu6',
            name=name + "_extra1")

        self.normal_conv = ConvBNLayer(
            int(out_channels1),
            int(out_channels2),
            kernel_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups),
            act='relu6',
            name=name + "_extra2")

    def forward(self, x):
        x = self.pointwise_conv(x)
        x = self.normal_conv(x)
        return x


@register
@serializable
class MobileNet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 norm_type='bn',
                 scale=1,
                 feature_maps=[4, 6, 13],
                 with_extra_blocks=False,
                 extra_block_filters=[[256, 512], [128, 256], [128, 256], [64, 128]]):
        super(MobileNet, self).__init__()
        self.scale = scale
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        self.with_extra_blocks = with_extra_blocks
        self.extra_block_filters = extra_block_filters

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            name="conv1")

        self.dwsl = []
        dws21 = self.add_sublayer(
            "conv2_1",
            sublayer=DepthwiseSeparable(
                in_channels=int(32 * scale),
                out_channels1=32,
                out_channels2=64,
                num_groups=32,
                stride=1,
                scale=scale,
                name="conv2_1"))
        self.dwsl.append(dws21)
        dws22 = self.add_sublayer(
            "conv2_2",
            sublayer=DepthwiseSeparable(
                in_channels=int(64 * scale),
                out_channels1=64,
                out_channels2=128,
                num_groups=64,
                stride=2,
                scale=scale,
                name="conv2_2"))
        self.dwsl.append(dws22)
        # 1/4
        dws31 = self.add_sublayer(
            "conv3_1",
            sublayer=DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=128,
                num_groups=128,
                stride=1,
                scale=scale,
                name="conv3_1"))
        self.dwsl.append(dws31)
        dws32 = self.add_sublayer(
            "conv3_2",
            sublayer=DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=256,
                num_groups=128,
                stride=2,
                scale=scale,
                name="conv3_2"))
        self.dwsl.append(dws32)
        # 1/8
        dws41 = self.add_sublayer(
            "conv4_1",
            sublayer=DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=256,
                num_groups=256,
                stride=1,
                scale=scale,
                name="conv4_1"))
        self.dwsl.append(dws41)
        dws42 = self.add_sublayer(
            "conv4_2",
            sublayer=DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=512,
                num_groups=256,
                stride=2,
                scale=scale,
                name="conv4_2"))
        self.dwsl.append(dws42)
        # 1/16
        for i in range(5):
            tmp = self.add_sublayer(
                "conv5_" + str(i + 1),
                sublayer=DepthwiseSeparable(
                    in_channels=int(512 * scale),
                    out_channels1=512,
                    out_channels2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale,
                    name="conv5_" + str(i + 1)))
            self.dwsl.append(tmp)
        dws56 = self.add_sublayer(
            "conv5_6",
            sublayer=DepthwiseSeparable(
                in_channels=int(512 * scale),
                out_channels1=512,
                out_channels2=1024,
                num_groups=512,
                stride=2,
                scale=scale,
                name="conv5_6"))
        self.dwsl.append(dws56)
        # 1/32
        dws6 = self.add_sublayer(
            "conv6",
            sublayer=DepthwiseSeparable(
                in_channels=int(1024 * scale),
                out_channels1=1024,
                out_channels2=1024,
                num_groups=1024,
                stride=1,
                scale=scale,
                name="conv6"))
        self.dwsl.append(dws6)

        # todo
        if self.with_extra_blocks:
            self.extra_blocks = []
            num_filters = self.extra_block_filters

            conv7_1 = self.add_sublayer(
                "conv7_1",
                sublayer=ExtraBlock(
                    1024,
                    num_filters[0][0],
                    num_filters[0][1],
                    name="conv7_1"))
            self.extra_blocks.append(conv7_1)

            conv7_2 = self.add_sublayer(
                "conv7_2",
                sublayer=ExtraBlock(
                    num_filters[0][1],
                    num_filters[1][0],
                    num_filters[1][1],
                    name="conv7_2"))
            self.extra_blocks.append(conv7_2)

            conv7_3 = self.add_sublayer(
                "conv7_3",
                sublayer=ExtraBlock(
                    num_filters[1][1],
                    num_filters[2][0],
                    num_filters[2][1],
                    name="conv7_3"))
            self.extra_blocks.append(conv7_3)

            conv7_4 = self.add_sublayer(
                "conv7_4",
                sublayer=ExtraBlock(
                    num_filters[2][1],
                    num_filters[3][0],
                    num_filters[3][1],
                    name="conv7_4"))
            self.extra_blocks.append(conv7_4)

    def forward(self, inputs):
        outs = []
        y = self.conv1(inputs['image'])
        for i, block in enumerate(self.dwsl):
            y = block(y)
            if i + 1 in self.feature_maps:
                outs.append(y)

        if not self.with_extra_blocks:
            return outs

        y = outs[-1]
        for i, block in enumerate(self.extra_blocks):
            idx = i + len(self.dwsl)
            y = block(y)
            if idx + 1 in self.feature_maps:
                outs.append(y)
        return outs

