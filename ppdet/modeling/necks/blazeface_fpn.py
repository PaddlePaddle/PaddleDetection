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

import paddle
import paddle.nn.functional as F
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn.initializer import KaimingNormal
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['BlazeNeck']


def hard_swish(x):
    return x * F.relu6(x + 3) / 6.


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_groups=1,
                 act='relu',
                 conv_lr=0.1,
                 conv_decay=0.,
                 norm_decay=0.,
                 norm_type='bn',
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(
                learning_rate=conv_lr, initializer=KaimingNormal()),
            bias_attr=False)

        if norm_type in ['sync_bn', 'bn']:
            self._batch_norm = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self.act == "relu":
            x = F.relu(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        elif self.act == 'leaky':
            x = F.leaky_relu(x)
        elif self.act == 'hard_swish':
            x = hard_swish(x)
        return x


class FPN(nn.Layer):
    def __init__(self, in_channels, out_channels, name=None):
        super(FPN, self).__init__()
        self.conv1_fpn = ConvBNLayer(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            act='leaky',
            name=name + '_output1')
        self.conv2_fpn = ConvBNLayer(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            act='leaky',
            name=name + '_output2')
        self.conv3_fpn = ConvBNLayer(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            act='leaky',
            name=name + '_merge')

    def forward(self, input):
        output1 = self.conv1_fpn(input[0])
        output2 = self.conv2_fpn(input[1])
        up2 = F.upsample(
            output2, size=output1.shape[-2:], mode='nearest')
        output1 = paddle.add(output1, up2)
        output1 = self.conv3_fpn(output1)
        return output1, output2


class SSH(nn.Layer):
    def __init__(self, in_channels, out_channels, name=None):
        super(SSH, self).__init__()
        assert out_channels % 4 == 0
        self.conv0_ssh = ConvBNLayer(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            act=None,
            name=name + 'ssh_conv3')
        self.conv1_ssh = ConvBNLayer(
            out_channels // 2,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act='leaky',
            name=name + 'ssh_conv5_1')
        self.conv2_ssh = ConvBNLayer(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act=None,
            name=name + 'ssh_conv5_2')
        self.conv3_ssh = ConvBNLayer(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act='leaky',
            name=name + 'ssh_conv7_1')
        self.conv4_ssh = ConvBNLayer(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act=None,
            name=name + 'ssh_conv7_2')

    def forward(self, x):
        conv0 = self.conv0_ssh(x)
        conv1 = self.conv1_ssh(conv0)
        conv2 = self.conv2_ssh(conv1)
        conv3 = self.conv3_ssh(conv2)
        conv4 = self.conv4_ssh(conv3)
        concat = paddle.concat([conv0, conv2, conv4], axis=1)
        return F.relu(concat)


@register
@serializable
class BlazeNeck(nn.Layer):
    def __init__(self, in_channel, neck_type="None", data_format='NCHW'):
        super(BlazeNeck, self).__init__()
        self.neck_type = neck_type
        self.reture_input = False
        self._out_channels = in_channel
        if self.neck_type == 'None':
            self.reture_input = True
        if "fpn" in self.neck_type:
            self.fpn = FPN(self._out_channels[0],
                           self._out_channels[1],
                           name='fpn')
            self._out_channels = [
                self._out_channels[0] // 2, self._out_channels[1] // 2
            ]
        if "ssh" in self.neck_type:
            self.ssh1 = SSH(self._out_channels[0],
                            self._out_channels[0],
                            name='ssh1')
            self.ssh2 = SSH(self._out_channels[1],
                            self._out_channels[1],
                            name='ssh2')
            self._out_channels = [self._out_channels[0], self._out_channels[1]]

    def forward(self, inputs):
        if self.reture_input:
            return inputs
        output1, output2 = None, None
        if "fpn" in self.neck_type:
            backout_4, backout_1 = inputs
            output1, output2 = self.fpn([backout_4, backout_1])
        if self.neck_type == "only_fpn":
            return [output1, output2]
        if self.neck_type == "only_ssh":
            output1, output2 = inputs
        feature1 = self.ssh1(output1)
        feature2 = self.ssh2(output2)
        return [feature1, feature2]

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=c)
            for c in [self._out_channels[0], self._out_channels[1]]
        ]
