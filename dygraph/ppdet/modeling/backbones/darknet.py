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
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import batch_norm
from ..shape_spec import ShapeSpec

__all__ = ['DarkNet', 'ConvBNLayer']


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 norm_type='bn',
                 norm_decay=0.,
                 act="leaky",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '.conv.weights'),
            bias_attr=False)
        self.batch_norm = batch_norm(
            ch_out, norm_type=norm_type, norm_decay=norm_decay, name=name)
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(out, 0.1)
        return out


class DownSample(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=2,
                 padding=1,
                 norm_type='bn',
                 norm_decay=0.,
                 name=None):

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            norm_type=norm_type,
            norm_decay=norm_decay,
            name=name)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type='bn', norm_decay=0., name=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            norm_decay=norm_decay,
            name=name + '.0')
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            name=name + '.1')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = paddle.add(x=inputs, y=conv2)
        return out


class Blocks(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 count,
                 norm_type='bn',
                 norm_decay=0.,
                 name=None):
        super(Blocks, self).__init__()

        self.basicblock0 = BasicBlock(
            ch_in,
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            name=name + '.0')
        self.res_out_list = []
        for i in range(1, count):
            block_name = '{}.{}'.format(name, i)
            res_out = self.add_sublayer(
                block_name,
                BasicBlock(
                    ch_out * 2,
                    ch_out,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    name=block_name))
            self.res_out_list.append(res_out)
        self.ch_out = ch_out

    def forward(self, inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


@register
@serializable
class DarkNet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=53,
                 freeze_at=-1,
                 return_idx=[2, 3, 4],
                 num_stages=5,
                 norm_type='bn',
                 norm_decay=0.):
        super(DarkNet, self).__init__()
        self.depth = depth
        self.freeze_at = freeze_at
        self.return_idx = return_idx
        self.num_stages = num_stages
        self.stages = DarkNet_cfg[self.depth][0:num_stages]

        self.conv0 = ConvBNLayer(
            ch_in=3,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            name='yolo_input')

        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32 * 2,
            norm_type=norm_type,
            norm_decay=norm_decay,
            name='yolo_input.downsample')

        self._out_channels = []
        self.darknet_conv_block_list = []
        self.downsample_list = []
        ch_in = [64, 128, 256, 512, 1024]
        for i, stage in enumerate(self.stages):
            name = 'stage.{}'.format(i)
            conv_block = self.add_sublayer(
                name,
                Blocks(
                    int(ch_in[i]),
                    32 * (2**i),
                    stage,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    name=name))
            self.darknet_conv_block_list.append(conv_block)
            if i in return_idx:
                self._out_channels.append(64 * (2**i))
        for i in range(num_stages - 1):
            down_name = 'stage.{}.downsample'.format(i)
            downsample = self.add_sublayer(
                down_name,
                DownSample(
                    ch_in=32 * (2**(i + 1)),
                    ch_out=32 * (2**(i + 2)),
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    name=down_name))
            self.downsample_list.append(downsample)

    def forward(self, inputs):
        x = inputs['image']

        out = self.conv0(x)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet_conv_block_list):
            out = conv_block_i(out)
            if i == self.freeze_at:
                out.stop_gradient = True
            if i in self.return_idx:
                blocks.append(out)
            if i < self.num_stages - 1:
                out = self.downsample_list[i](out)
        return blocks

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
