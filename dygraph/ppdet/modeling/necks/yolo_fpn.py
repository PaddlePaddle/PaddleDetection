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
