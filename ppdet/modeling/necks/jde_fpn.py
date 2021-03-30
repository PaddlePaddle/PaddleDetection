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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppdet.core.workspace import register, serializable
from ..backbones.darknet import ConvBNLayer
import numpy as np
from ..shape_spec import ShapeSpec
from .yolo_fpn import YoloDetBlock, SPP, DropBlock, CoordConv, PPYOLODetBlock

__all__ = ['JDEFPN', 'PPJDEFPN']


@register
@serializable
class JDEFPN(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[1024, 768, 384],
                 norm_type='bn',
                 freeze_norm=True,
                 data_format='NCHW'):
        super(JDEFPN, self).__init__()
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
                    freeze_norm=freeze_norm,
                    name=name,
                    data_format=data_format))
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
                        freeze_norm=freeze_norm,
                        name=name,
                        data_format=data_format))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        det_feats = []
        emb_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = paddle.concat([route, block], axis=1)
                else:
                    block = paddle.concat([route, block], axis=-1)
            route, tip = self.yolo_blocks[i](block)
            det_feats.append(tip)

            # add emb_feats output
            emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2., data_format=self.data_format)

        return det_feats, emb_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class PPJDEFPN(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[1024, 768, 384],
                 norm_type='bn',
                 freeze_norm=True,
                 data_format='NCHW',
                 **kwargs):
        super(PPJDEFPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.coord_conv = kwargs.get('coord_conv', False)
        self.drop_block = kwargs.get('drop_block', False)
        if self.drop_block:
            self.block_size = kwargs.get('block_size', 3)
            self.keep_prob = kwargs.get('keep_prob', 0.9)

        self.spp = kwargs.get('spp', False)
        self.conv_block_num = kwargs.get('conv_block_num', 2)
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
                            padding=0,
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
                    ],
                    [
                        'conv{}'.format(2 * j + 1), ConvBNLayer,
                        [c_out, c_out * 2, 3], dict(
                            padding=1,
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
                    ],
                ]
                c_in, c_out = c_out * 2, c_out

            base_cfg += [[
                'route', ConvLayer, [c_in, c_out, 1], dict(
                    padding=0, norm_type=norm_type, freeze_norm=freeze_norm)
            ], [
                'tip', ConvLayer, [c_out, c_out * 2, 3], dict(
                    padding=1, norm_type=norm_type, freeze_norm=freeze_norm)
            ]]

            if self.conv_block_num == 2:
                if i == 0:
                    if self.spp:
                        spp_cfg = [[
                            'spp', SPP, [channel * 4, channel, 1], dict(
                                pool_size=[5, 9, 13],
                                norm_type=norm_type,
                                freeze_norm=freeze_norm)
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
                            pool_size=[5, 9, 13],
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
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
                        freeze_norm=freeze_norm,
                        data_format=data_format,
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        det_feats = []
        emb_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = paddle.concat([route, block], axis=1)
                else:
                    block = paddle.concat([route, block], axis=-1)
            route, tip = self.yolo_blocks[i](block)
            det_feats.append(tip)

            # add emb_feats output
            emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2., data_format=self.data_format)

        return det_feats, emb_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
