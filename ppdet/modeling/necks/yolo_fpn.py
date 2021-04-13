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
from ppdet.core.workspace import register, serializable
from ..backbones.darknet import ConvBNLayer
import numpy as np

from ..shape_spec import ShapeSpec

__all__ = ['YOLOv3FPN', 'PPYOLOFPN']


def add_coord(x):
    b = x.shape[0]
    if self.data_format == 'NCHW':
        h = x.shape[2]
        w = x.shape[3]
    else:
        h = x.shape[1]
        w = x.shape[2]

    gx = paddle.arange(w, dtype='float32') / (w - 1.) * 2.0 - 1.
    if self.data_format == 'NCHW':
        gx = gx.reshape([1, 1, 1, w]).expand([b, 1, h, w])
    else:
        gx = gx.reshape([1, 1, w, 1]).expand([b, h, w, 1])
    gx.stop_gradient = True

    gy = paddle.arange(h, dtype='float32') / (h - 1.) * 2.0 - 1.
    if self.data_format == 'NCHW':
        gy = gy.reshape([1, 1, h, 1]).expand([b, 1, h, w])
    else:
        gy = gy.reshape([1, h, 1, 1]).expand([b, h, w, 1])
    gy.stop_gradient = True

    return gx, gy


class YoloDetBlock(nn.Layer):
    def __init__(self, ch_in, channel, norm_type, name, data_format='NCHW'):
        """
        YOLODetBlock layer for yolov3, see https://arxiv.org/abs/1804.02767

        Args:
            ch_in (int): input channel
            channel (int): base channel
            norm_type (str): batch norm type
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
        conv_def = [
            ['conv0', ch_in, channel, 1, '.0.0'],
            ['conv1', channel, channel * 2, 3, '.0.1'],
            ['conv2', channel * 2, channel, 1, '.1.0'],
            ['conv3', channel, channel * 2, 3, '.1.1'],
            ['route', channel * 2, channel, 1, '.2'],
        ]

        self.conv_module = nn.Sequential()
        for idx, (conv_name, ch_in, ch_out, filter_size,
                  post_name) in enumerate(conv_def):
            self.conv_module.add_sublayer(
                conv_name,
                ConvBNLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=filter_size,
                    padding=(filter_size - 1) // 2,
                    norm_type=norm_type,
                    data_format=data_format,
                    name=name + post_name))

        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            padding=1,
            norm_type=norm_type,
            data_format=data_format,
            name=name + '.tip')

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class SPP(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 norm_type,
                 name,
                 act='leaky',
                 data_format='NCHW'):
        """
        SPP layer, which consist of four pooling layer follwed by conv layer

        Args:
            ch_in (int): input channel of conv layer
            ch_out (int): output channel of conv layer
            k (int): kernel size of conv layer
            norm_type (str): batch norm type
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for size in pool_size:
            pool = self.add_sublayer(
                '{}.pool1'.format(name),
                nn.MaxPool2D(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    data_format=data_format,
                    ceil_mode=False))
            self.pool.append(pool)
        self.conv = ConvBNLayer(
            ch_in,
            ch_out,
            k,
            padding=k // 2,
            norm_type=norm_type,
            name=name,
            act=act,
            data_format=data_format)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == "NCHW":
            y = paddle.concat(outs, axis=1)
        else:
            y = paddle.concat(outs, axis=-1)

        y = self.conv(y)
        return y


class DropBlock(nn.Layer):
    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = paddle.cast(paddle.rand(x.shape, x.dtype) < gamma, x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class CoordConv(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 padding,
                 norm_type,
                 name,
                 data_format='NCHW'):
        """
        CoordConv layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            name (str): layer name
            data_format (str): data format, NCHW or NHWC

        """
        super(CoordConv, self).__init__()
        self.conv = ConvBNLayer(
            ch_in + 2,
            ch_out,
            filter_size=filter_size,
            padding=padding,
            norm_type=norm_type,
            data_format=data_format,
            name=name)
        self.data_format = data_format

    def forward(self, x):
        gx, gy = add_coord(x)
        if self.data_format == 'NCHW':
            y = paddle.concat([x, gx, gy], axis=1)
        else:
            y = paddle.concat([x, gx, gy], axis=-1)
        y = self.conv(y)
        return y


class PPYOLODetBlock(nn.Layer):
    def __init__(self, cfg, name, data_format='NCHW'):
        """
        PPYOLODetBlock layer

        Args:
            cfg (list): layer configs for this block
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlock, self).__init__()
        self.conv_module = nn.Sequential()
        for idx, (conv_name, layer, args, kwargs) in enumerate(cfg[:-1]):
            kwargs.update(
                name='{}.{}'.format(name, conv_name), data_format=data_format)
            self.conv_module.add_sublayer(conv_name, layer(*args, **kwargs))

        conv_name, layer, args, kwargs = cfg[-1]
        kwargs.update(
            name='{}.{}'.format(name, conv_name), data_format=data_format)
        self.tip = layer(*args, **kwargs)

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class PPYOLOTinyDetBlock(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 name,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 data_format='NCHW'):
        """
        PPYOLO Tiny DetBlock layer
        Args:
            ch_in (list): input channel number
            ch_out (list): output channel number
            name (str): block name
            drop_block: whether user DropBlock
            block_size: drop block size
            keep_prob: probability to keep block in DropBlock
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLOTinyDetBlock, self).__init__()
        self.drop_block_ = drop_block
        self.conv_module = nn.Sequential()

        cfgs = [
            # name, in channels, out channels, filter_size, 
            # stride, padding, groups
            ['.0', ch_in, ch_out, 1, 1, 0, 1],
            ['.1', ch_out, ch_out, 5, 1, 2, ch_out],
            ['.2', ch_out, ch_out, 1, 1, 0, 1],
            ['.route', ch_out, ch_out, 5, 1, 2, ch_out],
        ]
        for cfg in cfgs:
            conv_name, conv_ch_in, conv_ch_out, filter_size, stride, padding, \
                    groups = cfg
            self.conv_module.add_sublayer(
                name + conv_name,
                ConvBNLayer(
                    ch_in=conv_ch_in,
                    ch_out=conv_ch_out,
                    filter_size=filter_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    name=name + conv_name))

        self.tip = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            name=name + conv_name)

        if self.drop_block_:
            self.drop_block = DropBlock(
                block_size=block_size,
                keep_prob=keep_prob,
                data_format=data_format,
                name=name + '.dropblock')

    def forward(self, inputs):
        if self.drop_block_:
            inputs = self.drop_block(inputs)
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class PPYOLODetBlockCSP(nn.Layer):
    def __init__(self,
                 cfg,
                 ch_in,
                 ch_out,
                 act,
                 norm_type,
                 name,
                 data_format='NCHW'):
        """
        PPYOLODetBlockCSP layer

        Args:
            cfg (list): layer configs for this block
            ch_in (int): input channel
            ch_out (int): output channel
            act (str): default mish
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlockCSP, self).__init__()
        self.data_format = data_format
        self.conv1 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name + '.left',
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name + '.right',
            data_format=data_format)
        self.conv3 = ConvBNLayer(
            ch_out * 2,
            ch_out * 2,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name,
            data_format=data_format)
        self.conv_module = nn.Sequential()
        for idx, (layer_name, layer, args, kwargs) in enumerate(cfg):
            kwargs.update(name=name + layer_name, data_format=data_format)
            self.conv_module.add_sublayer(layer_name, layer(*args, **kwargs))

    def forward(self, inputs):
        conv_left = self.conv1(inputs)
        conv_right = self.conv2(inputs)
        conv_left = self.conv_module(conv_left)
        if self.data_format == 'NCHW':
            conv = paddle.concat([conv_left, conv_right], axis=1)
        else:
            conv = paddle.concat([conv_left, conv_right], axis=-1)

        conv = self.conv3(conv)
        return conv, conv


@register
@serializable
class YOLOv3FPN(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 norm_type='bn',
                 data_format='NCHW'):
        """
        YOLOv3FPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC

        """
        super(YOLOv3FPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)

        self._out_channels = []
        self.yolo_blocks = []
        self.routes = []
        self.data_format = data_format
        for i in range(self.num_blocks):
            name = 'yolo_block.{}'.format(i)
            in_channel = in_channels[-i - 1]
            if i > 0:
                in_channel += 512 // (2**i)
            yolo_block = self.add_sublayer(
                name,
                YoloDetBlock(
                    in_channel,
                    channel=512 // (2**i),
                    norm_type=norm_type,
                    data_format=data_format,
                    name=name))
            self.yolo_blocks.append(yolo_block)
            # tip layer output channel doubled
            self._out_channels.append(1024 // (2**i))

            if i < self.num_blocks - 1:
                name = 'yolo_transition.{}'.format(i)
                route = self.add_sublayer(
                    name,
                    ConvBNLayer(
                        ch_in=512 // (2**i),
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        data_format=data_format,
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = paddle.concat([route, block], axis=1)
                else:
                    block = paddle.concat([route, block], axis=-1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2., data_format=self.data_format)

        return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class PPYOLOFPN(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 norm_type='bn',
                 data_format='NCHW',
                 coord_conv=False,
                 conv_block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False):
        """
        PPYOLOFPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            coord_conv (bool): whether use CoordConv or not
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not

        """
        super(PPYOLOFPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.coord_conv = coord_conv
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.coord_conv:
            ConvLayer = CoordConv
        else:
            ConvLayer = ConvBNLayer

        if self.drop_block:
            dropblock_cfg = [[
                'dropblock', DropBlock, [self.block_size, self.keep_prob],
                dict()
            ]]
        else:
            dropblock_cfg = []

        self._out_channels = []
        self.yolo_blocks = []
        self.routes = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // (2**i)
            channel = 64 * (2**self.num_blocks) // (2**i)
            base_cfg = []
            c_in, c_out = ch_in, channel
            for j in range(self.conv_block_num):
                base_cfg += [
                    [
                        'conv{}'.format(2 * j), ConvLayer, [c_in, c_out, 1],
                        dict(
                            padding=0, norm_type=norm_type)
                    ],
                    [
                        'conv{}'.format(2 * j + 1), ConvBNLayer,
                        [c_out, c_out * 2, 3], dict(
                            padding=1, norm_type=norm_type)
                    ],
                ]
                c_in, c_out = c_out * 2, c_out

            base_cfg += [[
                'route', ConvLayer, [c_in, c_out, 1], dict(
                    padding=0, norm_type=norm_type)
            ], [
                'tip', ConvLayer, [c_out, c_out * 2, 3], dict(
                    padding=1, norm_type=norm_type)
            ]]

            if self.conv_block_num == 2:
                if i == 0:
                    if self.spp:
                        spp_cfg = [[
                            'spp', SPP, [channel * 4, channel, 1], dict(
                                pool_size=[5, 9, 13], norm_type=norm_type)
                        ]]
                    else:
                        spp_cfg = []
                    cfg = base_cfg[0:3] + spp_cfg + base_cfg[
                        3:4] + dropblock_cfg + base_cfg[4:6]
                else:
                    cfg = base_cfg[0:2] + dropblock_cfg + base_cfg[2:6]
            elif self.conv_block_num == 0:
                if self.spp and i == 0:
                    spp_cfg = [[
                        'spp', SPP, [c_in * 4, c_in, 1], dict(
                            pool_size=[5, 9, 13], norm_type=norm_type)
                    ]]
                else:
                    spp_cfg = []
                cfg = spp_cfg + dropblock_cfg + base_cfg
            name = 'yolo_block.{}'.format(i)
            yolo_block = self.add_sublayer(name, PPYOLODetBlock(cfg, name))
            self.yolo_blocks.append(yolo_block)
            self._out_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = 'yolo_transition.{}'.format(i)
                route = self.add_sublayer(
                    name,
                    ConvBNLayer(
                        ch_in=channel,
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        data_format=data_format,
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = paddle.concat([route, block], axis=1)
                else:
                    block = paddle.concat([route, block], axis=-1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2., data_format=self.data_format)

        return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class PPYOLOTinyFPN(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[80, 56, 34],
                 detection_block_channels=[160, 128, 96],
                 norm_type='bn',
                 data_format='NCHW',
                 **kwargs):
        """
        PPYOLO Tiny FPN layer
        Args:
            in_channels (list): input channels for fpn
            detection_block_channels (list): channels in fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            kwargs: extra key-value pairs, such as parameter of DropBlock and spp 
        """
        super(PPYOLOTinyFPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels[::-1]
        assert len(detection_block_channels
                   ) > 0, "detection_block_channelslength should > 0"
        self.detection_block_channels = detection_block_channels
        self.data_format = data_format
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.drop_block = kwargs.get('drop_block', False)
        self.block_size = kwargs.get('block_size', 3)
        self.keep_prob = kwargs.get('keep_prob', 0.9)

        self.spp_ = kwargs.get('spp', False)
        if self.spp_:
            self.spp = SPP(self.in_channels[0] * 4,
                           self.in_channels[0],
                           k=1,
                           pool_size=[5, 9, 13],
                           norm_type=norm_type,
                           name='spp')

        self._out_channels = []
        self.yolo_blocks = []
        self.routes = []
        for i, (
                ch_in, ch_out
        ) in enumerate(zip(self.in_channels, self.detection_block_channels)):
            name = 'yolo_block.{}'.format(i)
            if i > 0:
                ch_in += self.detection_block_channels[i - 1]
            yolo_block = self.add_sublayer(
                name,
                PPYOLOTinyDetBlock(
                    ch_in,
                    ch_out,
                    name,
                    drop_block=self.drop_block,
                    block_size=self.block_size,
                    keep_prob=self.keep_prob))
            self.yolo_blocks.append(yolo_block)
            self._out_channels.append(ch_out)

            if i < self.num_blocks - 1:
                name = 'yolo_transition.{}'.format(i)
                route = self.add_sublayer(
                    name,
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        data_format=data_format,
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]

        yolo_feats = []
        for i, block in enumerate(blocks):
            if i == 0 and self.spp_:
                block = self.spp(block)

            if i > 0:
                if self.data_format == 'NCHW':
                    block = paddle.concat([route, block], axis=1)
                else:
                    block = paddle.concat([route, block], axis=-1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2., data_format=self.data_format)

        return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class PPYOLOPAN(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 norm_type='bn',
                 data_format='NCHW',
                 act='mish',
                 conv_block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False):
        """
        PPYOLOPAN layer with SPP, DropBlock and CSP connection.

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            act (str): activation function, default mish
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not

        """
        super(PPYOLOPAN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.drop_block:
            dropblock_cfg = [[
                'dropblock', DropBlock, [self.block_size, self.keep_prob],
                dict()
            ]]
        else:
            dropblock_cfg = []

        # fpn
        self.fpn_blocks = []
        self.fpn_routes = []
        fpn_channels = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // (2**(i - 1))
            channel = 512 // (2**i)
            base_cfg = []
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    [
                        '{}.0'.format(j), ConvBNLayer, [channel, channel, 1],
                        dict(
                            padding=0, act=act, norm_type=norm_type)
                    ],
                    [
                        '{}.1'.format(j), ConvBNLayer, [channel, channel, 3],
                        dict(
                            padding=1, act=act, norm_type=norm_type)
                    ]
                ]

            if i == 0 and self.spp:
                base_cfg[3] = [
                    'spp', SPP, [channel * 4, channel, 1], dict(
                        pool_size=[5, 9, 13], act=act, norm_type=norm_type)
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'fpn.{}'.format(i)
            fpn_block = self.add_sublayer(
                name,
                PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name,
                                  data_format))
            self.fpn_blocks.append(fpn_block)
            fpn_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = 'fpn_transition.{}'.format(i)
                route = self.add_sublayer(
                    name,
                    ConvBNLayer(
                        ch_in=channel * 2,
                        ch_out=channel,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act,
                        norm_type=norm_type,
                        data_format=data_format,
                        name=name))
                self.fpn_routes.append(route)
        # pan
        self.pan_blocks = []
        self.pan_routes = []
        self._out_channels = []
        for i in reversed(range(self.num_blocks - 1)):
            name = 'pan_transition.{}'.format(i)
            route = self.add_sublayer(
                name,
                ConvBNLayer(
                    ch_in=fpn_channels[i + 1],
                    ch_out=fpn_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act,
                    norm_type=norm_type,
                    data_format=data_format,
                    name=name))
            self.pan_routes = [route, ] + self.pan_routes
            base_cfg = []
            ch_in = fpn_channels[i] + fpn_channels[i + 1]
            channel = 512 // (2**i)
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    [
                        '{}.0'.format(j), ConvBNLayer, [channel, channel, 1],
                        dict(
                            padding=0, act=act, norm_type=norm_type)
                    ],
                    [
                        '{}.1'.format(j), ConvBNLayer, [channel, channel, 3],
                        dict(
                            padding=1, act=act, norm_type=norm_type)
                    ]
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'pan.{}'.format(i)
            pan_block = self.add_sublayer(
                name,
                PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name,
                                  data_format))

            self.pan_blocks = [pan_block, ] + self.pan_blocks
            self._out_channels.append(channel * 2)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        # fpn
        fpn_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = paddle.concat([route, block], axis=1)
                else:
                    block = paddle.concat([route, block], axis=-1)
            route, tip = self.fpn_blocks[i](block)
            fpn_feats.append(tip)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2., data_format=self.data_format)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[self.num_blocks - 1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            if self.data_format == 'NCHW':
                block = paddle.concat([route, block], axis=1)
            else:
                block = paddle.concat([route, block], axis=-1)

            route, tip = self.pan_blocks[i](block)
            pan_feats.append(tip)

        return pan_feats[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
