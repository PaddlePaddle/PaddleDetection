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
from .yolo_fpn import YoloDetBlock

__all__ = ['JDEFPN']


@register
@serializable
class JDEFPN(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 in_channels=[1024, 768, 384],
                 norm_type='bn',
                 freeze_norm=True):
        super(JDEFPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)

        self._out_channels = []
        self.yolo_blocks = []
        self.routes = []
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
                        freeze_norm=freeze_norm,
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        det_feats = []
        emb_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route, tip = self.yolo_blocks[i](block)
            det_feats.append(tip)

            # add emb_feats output
            emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        return det_feats, emb_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]

