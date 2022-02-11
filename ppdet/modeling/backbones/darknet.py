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

from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import batch_norm, mish
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
                 freeze_norm=False,
                 data_format='NCHW',
                 name=''):
        """
        conv + bn + activation layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            stride (int): stride, default 1
            groups (int): number of groups of conv layer, default 1
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            act (str): activation function type, default 'leaky', which means leaky_relu
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            data_format=data_format,
            bias_attr=False)
        self.batch_norm = batch_norm(
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(out, 0.1)
        else:
            out = getattr(F, self.act)(out)
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
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        downsample layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            stride (int): stride, default 2
            padding (int): padding size, default 1
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        BasicBlock layer of DarkNet

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """

        super(BasicBlock, self).__init__()

        assert ch_in == ch_out and (ch_in % 2) == 0, \
            f"ch_in and ch_out should be the same even int, but the input \'ch_in is {ch_in}, \'ch_out is {ch_out}"
        # example:
        # --------------{conv1} --> {conv2}
        # channel route: 10-->5 --> 5-->10
        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=int(ch_out / 2),
            filter_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            ch_in=int(ch_out / 2),
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)

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
                 freeze_norm=False,
                 name=None,
                 data_format='NCHW'):
        """
        Blocks layer, which consist of some BaickBlock layers

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            count (int): number of BasicBlock layer
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(Blocks, self).__init__()

        self.basicblock0 = BasicBlock(
            ch_in,
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.res_out_list = []
        for i in range(1, count):
            block_name = '{}.{}'.format(name, i)
            res_out = self.add_sublayer(
                block_name,
                BasicBlock(
                    ch_out,
                    ch_out,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    data_format=data_format))
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
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 depth=53,
                 freeze_at=-1,
                 return_idx=[2, 3, 4],
                 num_stages=5,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        Darknet, see https://pjreddie.com/darknet/yolo/

        Args:
            depth (int): depth of network
            freeze_at (int): freeze the backbone at which stage
            filter_size (int): filter size, default 3
            return_idx (list): index of stages whose feature maps are returned
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            data_format (str): data format, NCHW or NHWC
        """
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
            freeze_norm=freeze_norm,
            data_format=data_format)

        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32 * 2,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)

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
                    int(ch_in[i]),
                    stage,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    data_format=data_format,
                    name=name))
            self.darknet_conv_block_list.append(conv_block)
            if i in return_idx:
                self._out_channels.append(int(ch_in[i]))
        for i in range(num_stages - 1):
            down_name = 'stage.{}.downsample'.format(i)
            downsample = self.add_sublayer(
                down_name,
                DownSample(
                    ch_in=int(ch_in[i]),
                    ch_out=int(ch_in[i + 1]),
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    data_format=data_format))
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
