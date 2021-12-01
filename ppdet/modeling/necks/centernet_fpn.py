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

import numpy as np
import math
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Uniform
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ppdet.modeling.backbones.hardnet import ConvLayer, HarDBlock
from ..shape_spec import ShapeSpec

__all__ = ['CenterNetDLAFPN', 'CenterNetHarDNetFPN']


# SGE attention
class BasicConv(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias_attr=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(
            out_planes,
            epsilon=1e-5,
            momentum=0.01,
            weight_attr=False,
            bias_attr=False) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Layer):
    def forward(self, x):
        return paddle.concat(
            (paddle.max(x, 1).unsqueeze(1), paddle.mean(x, 1).unsqueeze(1)),
            axis=1)


class SpatialGate(nn.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2,
            1,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


def fill_up_weights(up):
    weight = up.weight.numpy()
    f = math.ceil(weight.shape[2] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(weight.shape[2]):
        for j in range(weight.shape[3]):
            weight[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, weight.shape[0]):
        weight[c, 0, :, :] = weight[0, 0, :, :]
    up.weight.set_value(weight)


class IDAUp(nn.Layer):
    def __init__(self, ch_ins, ch_out, up_strides, dcn_v2=True):
        super(IDAUp, self).__init__()
        for i in range(1, len(ch_ins)):
            ch_in = ch_ins[i]
            up_s = int(up_strides[i])
            fan_in = ch_in * 3 * 3
            stdv = 1. / math.sqrt(fan_in)
            proj = nn.Sequential(
                ConvNormLayer(
                    ch_in,
                    ch_out,
                    filter_size=3,
                    stride=1,
                    use_dcn=dcn_v2,
                    bias_on=dcn_v2,
                    norm_decay=None,
                    dcn_lr_scale=1.,
                    dcn_regularizer=None,
                    initializer=Uniform(-stdv, stdv)),
                nn.ReLU())
            node = nn.Sequential(
                ConvNormLayer(
                    ch_out,
                    ch_out,
                    filter_size=3,
                    stride=1,
                    use_dcn=dcn_v2,
                    bias_on=dcn_v2,
                    norm_decay=None,
                    dcn_lr_scale=1.,
                    dcn_regularizer=None,
                    initializer=Uniform(-stdv, stdv)),
                nn.ReLU())

            kernel_size = up_s * 2
            fan_in = ch_out * kernel_size * kernel_size
            stdv = 1. / math.sqrt(fan_in)
            up = nn.Conv2DTranspose(
                ch_out,
                ch_out,
                kernel_size=up_s * 2,
                stride=up_s,
                padding=up_s // 2,
                groups=ch_out,
                weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
                bias_attr=False)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, inputs, start_level, end_level):
        for i in range(start_level + 1, end_level):
            upsample = getattr(self, 'up_' + str(i - start_level))
            project = getattr(self, 'proj_' + str(i - start_level))

            inputs[i] = project(inputs[i])
            inputs[i] = upsample(inputs[i])
            node = getattr(self, 'node_' + str(i - start_level))
            inputs[i] = node(paddle.add(inputs[i], inputs[i - 1]))


class DLAUp(nn.Layer):
    def __init__(self, start_level, channels, scales, ch_in=None, dcn_v2=True):
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
                    dcn_v2=dcn_v2))
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
class CenterNetDLAFPN(nn.Layer):
    """
    Args:
        in_channels (list): number of input feature channels from backbone.
            [16, 32, 64, 128, 256, 512] by default, means the channels of DLA-34
        down_ratio (int): the down ratio from images to heatmap, 4 by default
        last_level (int): the last level of input feature fed into the upsamplng block
        out_channel (int): the channel of the output feature, 0 by default means
            the channel of the input feature whose down ratio is `down_ratio`
        first_level (None): the first level of input feature fed into the upsamplng block.
            if None, the first level stands for logs(down_ratio)
        dcn_v2 (bool): whether use the DCNv2, True by default
        with_sge (bool): whether use SGE attention, False by default
    """

    def __init__(self,
                 in_channels,
                 down_ratio=4,
                 last_level=5,
                 out_channel=0,
                 first_level=None,
                 dcn_v2=True,
                 with_sge=False):
        super(CenterNetDLAFPN, self).__init__()
        self.first_level = int(np.log2(
            down_ratio)) if first_level is None else first_level
        assert self.first_level >= 0, "first level in CenterNetDLAFPN should be greater or equal to 0, but received {}".format(
            self.first_level)
        self.down_ratio = down_ratio
        self.last_level = last_level
        scales = [2**i for i in range(len(in_channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level,
            in_channels[self.first_level:],
            scales,
            dcn_v2=dcn_v2)
        self.out_channel = out_channel
        if out_channel == 0:
            self.out_channel = in_channels[self.first_level]
        self.ida_up = IDAUp(
            in_channels[self.first_level:self.last_level],
            self.out_channel,
            [2**i for i in range(self.last_level - self.first_level)],
            dcn_v2=dcn_v2)

        self.with_sge = with_sge
        if self.with_sge:
            self.sge_attention = SpatialGate()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    def forward(self, body_feats):

        dla_up_feats = self.dla_up(body_feats)

        ida_up_feats = []
        for i in range(self.last_level - self.first_level):
            ida_up_feats.append(dla_up_feats[i].clone())

        self.ida_up(ida_up_feats, 0, len(ida_up_feats))

        feat = ida_up_feats[-1]
        if self.with_sge:
            feat = self.sge_attention(feat)
        if self.down_ratio != 4:
            feat = F.interpolate(feat, scale_factor=self.down_ratio // 4, mode="bilinear", align_corners=True)
        return feat

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel, stride=self.down_ratio)]


class TransitionUp(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip):
        w, h = skip.shape[2], skip.shape[3]
        out = F.interpolate(x, size=(w, h), mode="bilinear", align_corners=True)
        out = paddle.concat([out, skip], 1)
        return out


@register
@serializable
class CenterNetHarDNetFPN(nn.Layer):
    """
    Args:
        in_channels (list): number of input feature channels from backbone.
            [96, 214, 458, 784] by default, means the channels of HarDNet85
        num_layers (int): HarDNet laters, 85 by default
        down_ratio (int): the down ratio from images to heatmap, 4 by default
        first_level (int|None): the first level of input feature fed into the upsamplng block.
            if None, the first level stands for logs(down_ratio) - 1

        last_level (int): the last level of input feature fed into the upsamplng block
        out_channel (int): the channel of the output feature, 0 by default means
            the channel of the input feature whose down ratio is `down_ratio`
    """

    def __init__(self,
                 in_channels,
                 num_layers=85,
                 down_ratio=4,
                 first_level=None,
                 last_level=4,
                 out_channel=0):
        super(CenterNetHarDNetFPN, self).__init__()
        self.first_level = int(np.log2(
            down_ratio)) - 1 if first_level is None else first_level
        assert self.first_level >= 0, "first level in CenterNetDLAFPN should be greater or equal to 0, but received {}".format(
            self.first_level)
        self.down_ratio = down_ratio
        self.last_level = last_level
        self.last_pool = nn.AvgPool2D(kernel_size=2, stride=2)

        assert num_layers in [68, 85], "HarDNet-{} not support.".format(
            num_layers)
        if num_layers == 85:
            self.last_proj = ConvLayer(784, 256, kernel_size=1)
            self.last_blk = HarDBlock(768, 80, 1.7, 8)
            self.skip_nodes = [1, 3, 8, 13]
            self.SC = [32, 32, 0]
            gr = [64, 48, 28]
            layers = [8, 8, 4]
            ch_list2 = [224 + self.SC[0], 160 + self.SC[1], 96 + self.SC[2]]
            channels = [96, 214, 458, 784]
            self.skip_lv = 3

        elif num_layers == 68:
            self.last_proj = ConvLayer(654, 192, kernel_size=1)
            self.last_blk = HarDBlock(576, 72, 1.7, 8)
            self.skip_nodes = [1, 3, 8, 11]
            self.SC = [32, 32, 0]
            gr = [48, 32, 20]
            layers = [8, 8, 4]
            ch_list2 = [224 + self.SC[0], 96 + self.SC[1], 64 + self.SC[2]]
            channels = [64, 124, 328, 654]
            self.skip_lv = 2

        self.transUpBlocks = nn.LayerList([])
        self.denseBlocksUp = nn.LayerList([])
        self.conv1x1_up = nn.LayerList([])
        self.avg9x9 = nn.AvgPool2D(kernel_size=(9, 9), stride=1, padding=(4, 4))
        prev_ch = self.last_blk.get_out_ch()

        for i in range(3):
            skip_ch = channels[3 - i]
            self.transUpBlocks.append(TransitionUp(prev_ch, prev_ch))
            if i < self.skip_lv:
                cur_ch = prev_ch + skip_ch
            else:
                cur_ch = prev_ch
            self.conv1x1_up.append(
                ConvLayer(
                    cur_ch, ch_list2[i], kernel_size=1))
            cur_ch = ch_list2[i]
            cur_ch -= self.SC[i]
            cur_ch *= 3

            blk = HarDBlock(cur_ch, gr[i], 1.7, layers[i])
            self.denseBlocksUp.append(blk)
            prev_ch = blk.get_out_ch()

        prev_ch += self.SC[0] + self.SC[1] + self.SC[2]
        self.out_channel = prev_ch

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    def forward(self, body_feats):
        x = body_feats[-1]
        x_sc = []
        x = self.last_proj(x)
        x = self.last_pool(x)
        x2 = self.avg9x9(x)
        x3 = x / (x.sum((2, 3), keepdim=True) + 0.1)
        x = paddle.concat([x, x2, x3], 1)
        x = self.last_blk(x)

        for i in range(3):
            skip_x = body_feats[3 - i]
            x_up = self.transUpBlocks[i](x, skip_x)
            x_ch = self.conv1x1_up[i](x_up)
            if self.SC[i] > 0:
                end = x_ch.shape[1]
                new_st = end - self.SC[i]
                x_sc.append(x_ch[:, new_st:, :, :])
                x_ch = x_ch[:, :new_st, :, :]
            x2 = self.avg9x9(x_ch)
            x3 = x_ch / (x_ch.sum((2, 3), keepdim=True) + 0.1)
            x_new = paddle.concat([x_ch, x2, x3], 1)
            x = self.denseBlocksUp[i](x_new)

        scs = [x]
        for i in range(3):
            if self.SC[i] > 0:
                scs.insert(
                    0,
                    F.interpolate(
                        x_sc[i],
                        size=(x.shape[2], x.shape[3]),
                        mode="bilinear",
                        align_corners=True))
        neck_feat = paddle.concat(scs, 1)
        return neck_feat

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel, stride=self.down_ratio)]
