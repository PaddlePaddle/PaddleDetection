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
from ..backbone.darknet import ConvBNLayer


class YoloDetBlock(nn.Layer):
    def __init__(self, ch_in, channel, norm_type, name):
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
                    name=name + post_name))

        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            padding=1,
            norm_type=norm_type,
            name=name + '.tip')

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class SPP(nn.Layer):
    def __init__(self, ch_in, ch_out, k, pool_size, norm_type, name):
        super(SPP, self).__init__()
        self.pool = []
        for size in pool_size:
            pool = self.add_sublayer(
                '{}.spp.pool1'.format(name),
                nn.Pool2D(
                    pool_size=size, pool_padding=size // 2))
            self.pool.append(pool)
        self.conv = ConvBNLayer(
            ch_in,
            ch_out,
            k,
            padding=k // 2,
            norm_type=norm_type,
            name='{}.spp.conv'.format(name))

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = paddle.concat(outs, axis=0)
        y = self.conv(y)
        return y


class DropBlock(nn.Layer):
    def __init__(self, block_size, keep_prob, name):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name

    def forward(self, x):
        if not self.training and sell.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            for s in x.shape[2:]:
                gamma *= s / (s - self.block_size + 1)

            matrix = paddle.bernoulli(paddle.full_like(x, gamma))
            mask_inv = F.max_pool2d(
                matrix, self.block_size, stride=1, padding=self.block_size // 2)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class CoordConv(nn.Layer):
    def __init__(self, ch_in, ch_out, filter_size, padding, norm_type, name):
        super(CoordConv, self).__init__()
        self.conv = ConvBNLayer(
            ch_in,
            ch_out,
            filter_size=filter_size,
            padding=padding,
            norm_type=norm_type,
            name='{}.conv'.format(name))

    def forward(self, x):
        b, _, h, w = x.shape

        gx = paddle.arange(w, dtype=x.dtype) / (w - 1) * 2.0 - 1
        gx = gx.unsqueeze([0, 1, 2]).expand([b, 1, h, 1])
        gx.stop_gradient = True

        gy = paddle.arange(h, dtype=x.dtype) / (h - 1) * 2.0 - 1
        gy = gy.unsqueeze([0, 1, 3]).expand([b, 1, 1, w])
        gy.stop_gradient = True

        y = paddle.concat([x, gx, gy], axis=1)
        y = self.conv(y)
        return y


class PPYOLODetBlock(nn.Layer):
    def __init__(self, cfg, name):
        super(PPYOLODetBlock, self).__init__()
        self.conv_module = nn.Sequential()
        for idx, (conv_name, layer, args, kwargs) in enumerate(cfg[:-1]):
            self.conv_module.add_sublayer(
                conv_name,
                layer(
                    *args, **kwargs, name='{}.{}'.format(name + conv_name,
                                                         idx)))

        name, layer, args, kwargs = cfg[-1]
        self.tip = layer(*args, **kwargs, name='{}.tip'.format(name))

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


@register
@serializable
class YOLOv3FPN(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self, feat_channels=[1024, 768, 384], norm_type='bn'):
        super(YOLOv3FPN, self).__init__()
        assert len(feat_channels) > 0, "feat_channels length should > 0"
        self.feat_channels = feat_channels
        self.num_blocks = len(feat_channels)
        self.yolo_blocks = []
        self.routes = []
        for i in range(self.num_blocks):
            name = 'yolo_block.{}'.format(i)
            yolo_block = self.add_sublayer(
                name,
                YoloDetBlock(
                    feat_channels[i],
                    channel=512 // (2**i),
                    norm_type=norm_type,
                    name=name))
            self.yolo_blocks.append(yolo_block)

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
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        return yolo_feats


@register
@serializable
class PPYOLOFPN(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self, feat_channels=[1024, 768, 384], norm_type='bn',
                 **kwargs):
        super(PPYOLOFPN, self).__init__()
        assert len(feat_channels) > 0, "feat_channels length should > 0"
        self.feat_channels = feat_channels
        self.num_blocks = len(feat_channels)
        self.yolo_blocks = []
        self.routes = []
        # parse kwargs
        self.coord_conv = kwargs.get('coord_conv', False)
        self.drop_block = kwargs.get('drop_block', False)
        if self.drop_block:
            self.block_size = kwargs.get('block_size', 3)
            self.keep_prob = kwargs.get('keep_prob', 0.9)

        self.spp = kwargs.get('spp', False)
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

        self.yolo_blocks = []
        self.routes = []
        for i, ch_in in enumerate(self.feat_channels):
            channel = 64 * (2**self.num_blocks) // (2**i)
            base_cfg = [
                # name of layer, Layer, args
                ['conv0', ConvLayer, [ch_in, channel, 1]],
                ['conv1', ConvBNLayer, [channel, channel * 2, 3]],
                ['conv2', ConvLayer, [channel * 2, channel, 1]],
                ['conv3', ConvBNLayer, [channel, channel * 2, 3]],
                ['route', ConvLayer, [channel * 2, channel, 1]],
                ['tip', ConvLayer, [channel, channel * 2, 3]]
            ]
            for conf in base_cfg:
                filter_size = conf[-1][-1]
                conf.append(dict(padding=filter_size // 2, norm_type=norm_type))
            if i == 0:
                if self.spp:
                    spp_cfg = [[
                        'spp', SPP, [channel, channel, 1], dict(
                            pool_size=[5, 9, 13], norm_type=norm_type)
                    ]]
                else:
                    spp_cfg = []
                cfg = base_cfg[0:3] + spp_cfg + base_cfg[
                    3:4] + dropblock_cfg + base_cfg[4:6]
            else:
                cfg = base_cfg[0:2] + dropblock_cfg + dropblock_cfg[2:6]
            name = 'yolo_block.{}'.format(i)
            yolo_block = self.add_sublayer(name, PPYOLODetBlock(cfg, name))
            if i < self.num_blocks - 1:
                name = 'yolo_transition.{}'.format(i)
                route = self.add_sublayer(
                    name,
                    ConvBNLayer(
                        ch_in=channel // (2**i),
                        ch_out=channel // (2**(i + 1)),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        return yolo_feats
