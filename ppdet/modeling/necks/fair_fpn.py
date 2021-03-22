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

import numpy as np
import math
import paddle
import paddle.nn.functional as F
from paddle import ParamAttr
import paddle.nn as nn
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingUniform
from ppdet.core.workspace import register, serializable
from ppdet.modeling.backbones.dla import ConvLayer, NormLayer
from ..shape_spec import ShapeSpec


class IDAUp(nn.Layer):
    def __init__(self, ch_ins, ch_out, up_strides, name=None):
        super(IDAUp, self).__init__()
        for i in range(1, len(ch_ins)):
            ch_in = ch_ins[i]
            up_s = int(up_strides[i])
            proj = nn.Sequential(
                ConvLayer(
                    ch_in,
                    ch_out,
                    kernel_size=3,
                    padding=1,
                    dcn_v2=True,
                    bias=True,
                    name=name + ".proj_{}.conv".format(i)),
                NormLayer(
                    ch_out, name=name + ".proj_{}.actf.0".format(i)),
                nn.ReLU())
            node = nn.Sequential(
                ConvLayer(
                    ch_out,
                    ch_out,
                    kernel_size=3,
                    padding=1,
                    dcn_v2=True,
                    bias=True,
                    name=name + ".node_{}.conv".format(i)),
                NormLayer(
                    ch_out, name=name + ".node_{}.actf.0".format(i)),
                nn.ReLU())

            param_attr = paddle.ParamAttr(
                initializer=KaimingUniform(),
                name=name + ".up_{}.weight".format(i))
            up = nn.Conv2DTranspose(
                ch_out,
                ch_out,
                kernel_size=up_s * 2,
                weight_attr=param_attr,
                stride=up_s,
                padding=up_s // 2,
                groups=ch_out,
                bias_attr=False)
            self.fill_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def fill_weights(self, up):
        weight = up.weight
        f = math.ceil(weight.shape[2] / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(weight.shape[2]):
            for j in range(weight.shape[3]):
                weight[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, weight.shape[0]):
            weight[c, 0, :, :] = weight[0, 0, :, :]

    def forward(self, inputs, start_level, end_level):
        for i in range(start_level + 1, end_level):
            upsample = getattr(self, 'up_' + str(i - start_level))
            project = getattr(self, 'proj_' + str(i - start_level))
            inputs[i] = upsample(project(inputs[i]))
            node = getattr(self, 'node_' + str(i - start_level))
            inputs[i] = node(paddle.add(inputs[i], inputs[i - 1]))


class DLAUp(nn.Layer):
    def __init__(self, start_level, channels, scales, ch_in=None, name=None):
        super(DLAUp, self).__init__()
        self.start_level = start_level
        if ch_in is None:
            ch_in = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                'ida_{}'.format(i),
                IDAUp(
                    ch_in[j:],
                    channels[j],
                    scales[j:] // scales[j],
                    name=name + '.ida_{}'.format(i)))
            scales[j + 1:] = scales[j]
            ch_in[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, inputs):
        out = [inputs[-1]]  # start with 32
        for i in range(len(inputs) - self.start_level - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(inputs, len(inputs) - i - 2, len(inputs))
            out.insert(0, inputs[-1])
        return out


@register
@serializable
class FairDLAFPN(nn.Layer):
    def __init__(self, in_channels, down_ratio=4, last_level=5, out_channel=0):
        super(FairDLAFPN, self).__init__()
        self.first_level = int(np.log2(down_ratio))
        self.down_ratio = down_ratio
        self.last_level = last_level
        scales = [2**i for i in range(len(in_channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level,
            in_channels[self.first_level:],
            scales,
            name="dla_up")
        self.out_channel = out_channel
        if out_channel == 0:
            self.out_channel = in_channels[self.first_level]
        self.ida_up = IDAUp(
            in_channels[self.first_level:self.last_level],
            self.out_channel,
            [2**i for i in range(self.last_level - self.first_level)],
            name="ida_up")

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    def forward(self, body_feats):
        dla_up_feats = self.dla_up(body_feats)
        #for i in range(len(dla_up_feats)):
        #    print('-----------------dla up {}'.format(i), np.mean(dla_up_feats[i].numpy()))

        ida_up_feats = []
        for i in range(self.last_level - self.first_level):
            ida_up_feats.append(dla_up_feats[i].clone())
        self.ida_up(ida_up_feats, 0, len(ida_up_feats))
        #for i in range(len(ida_up_feats)):
        #    print('-----------------ida up {}'.format(i), np.mean(ida_up_feats[i].numpy()))

        return ida_up_feats

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel, stride=self.down_ratio)]
