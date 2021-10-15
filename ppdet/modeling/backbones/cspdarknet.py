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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['CSPDarkNet']

MODEL_CFGS = {
    53: {
        'ch_out': (64, 128, 256, 512, 1024),
        'depth': (1, 2, 8, 8, 4),
        'stride': (2, ) * 5,
        'exp_ratio': (2., ) + (1., ) * 4,
        'bottle_ratio': (0.5, ) + (1.0, ) * 4,
        'block_ratio': (1., ) + (0.5, ) * 4
    }
}


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act='relu'):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == 'leaky':
            x = F.leaky_relu(x, 0.1)
        else:
            x = getattr(F, self.act)(x)

        return x


class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, ratio=0.5, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()

        c_ = int(ch_out * ratio)
        self.conv1 = ConvBNLayer(ch_in, c_, filter_size=1, padding=0, act=act)
        self.conv2 = ConvBNLayer(c_, ch_out, filter_size=3, padding=1, act=act)
        self.shortcut = shortcut and ch_in == ch_out

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class CSPStage(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 n=1,
                 exp_ratio=1.,
                 bottle_ratio=1.,
                 block_ratio=1.,
                 act='relu',
                 shortcut=True):
        super(CSPStage, self).__init__()

        if stride != 1:
            self.conv_down = ConvBNLayer(
                ch_in, ch_out, filter_size=3, stride=stride, padding=1, act=act)
            pre_ch = ch_out
        else:
            self.conv_down = None
            pre_ch = ch_in

        exp_ch = int(ch_out * exp_ratio)
        block_out_ch = int(ch_out * block_ratio)
        self.conv_exp = ConvBNLayer(
            pre_ch, exp_ch, filter_size=1, padding=0, act=act)
        self.blocks = nn.Sequential()
        pre_ch = exp_ch // 2
        for i in range(n):
            self.blocks.add_sublayer(
                str(i),
                BasicBlock(
                    pre_ch,
                    block_out_ch,
                    ratio=bottle_ratio,
                    act=act,
                    shortcut=shortcut))
            pre_ch = block_out_ch

        self.conv_transition_b = ConvBNLayer(
            pre_ch, block_out_ch, filter_size=1, padding=0, act=act)
        self.conv_transition = ConvBNLayer(
            exp_ch, ch_out, filter_size=1, padding=0, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        x = self.conv_exp(x)
        xs, xb = paddle.split(x, 2, axis=1)
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb)
        out = self.conv_transition(paddle.concat([xs, xb], axis=1))
        return out


@register
@serializable
class CSPDarkNet(nn.Layer):
    def __init__(self, layers, act='leaky', return_idx=[0, 1, 2, 3, 4]):
        super(CSPDarkNet, self).__init__()
        cfg = MODEL_CFGS[layers]

        self.stem = nn.Sequential(('conv1', ConvBNLayer(
            3, 32, 3, 1, padding=0, act=act)))

        self.stages = nn.Sequential()
        ch_outs = cfg['ch_out']
        depths = cfg['depth']
        strides = cfg['stride']
        exp_ratios = cfg['exp_ratio']
        bottle_ratios = cfg['bottle_ratio']
        block_ratios = cfg['block_ratio']
        ch_in = 32
        for i, (ch_out, stride, n, exp_ratio, bottle_ratio,
                block_ratio) in enumerate(
                    zip(ch_outs, strides, depths, exp_ratios, bottle_ratios,
                        block_ratios)):
            self.stages.add_sublayer(
                str(i),
                CSPStage(
                    ch_in,
                    ch_out,
                    stride,
                    n,
                    exp_ratio,
                    bottle_ratio,
                    block_ratio,
                    act=act))
            ch_in = ch_out

        self._out_channels = ch_outs
        self._out_strides = [2, 4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]
