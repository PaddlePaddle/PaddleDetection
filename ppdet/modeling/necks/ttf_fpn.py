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
from paddle.nn.initializer import Constant, Uniform, Normal, XavierUniform
from ppdet.core.workspace import register, serializable
from paddle.regularizer import L2Decay
from ppdet.modeling.layers import DeformableConvV2, ConvNormLayer, LiteConv
import math
from ppdet.modeling.ops import batch_norm
from ..shape_spec import ShapeSpec

__all__ = ['TTFFPN']


class Upsample(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type='bn'):
        super(Upsample, self).__init__()
        fan_in = ch_in * 3 * 3
        stdv = 1. / math.sqrt(fan_in)
        self.dcn = DeformableConvV2(
            ch_in,
            ch_out,
            kernel_size=3,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(
                initializer=Constant(0),
                regularizer=L2Decay(0.),
                learning_rate=2.),
            lr_scale=2.,
            regularizer=L2Decay(0.))

        self.bn = batch_norm(
            ch_out, norm_type=norm_type, initializer=Constant(1.))

    def forward(self, feat):
        dcn = self.dcn(feat)
        bn = self.bn(dcn)
        relu = F.relu(bn)
        out = F.interpolate(relu, scale_factor=2., mode='bilinear')
        return out


class DeConv(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type='bn'):
        super(DeConv, self).__init__()
        self.deconv = nn.Sequential()
        conv1 = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            stride=1,
            filter_size=1,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv2 = nn.Conv2DTranspose(
            in_channels=ch_out,
            out_channels=ch_out,
            kernel_size=4,
            padding=1,
            stride=2,
            groups=ch_out,
            weight_attr=ParamAttr(initializer=XavierUniform()),
            bias_attr=False)
        bn = batch_norm(ch_out, norm_type=norm_type, norm_decay=0.)
        conv3 = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            stride=1,
            filter_size=1,
            norm_type=norm_type,
            initializer=XavierUniform())

        self.deconv.add_sublayer('conv1', conv1)
        self.deconv.add_sublayer('relu6_1', nn.ReLU6())
        self.deconv.add_sublayer('conv2', conv2)
        self.deconv.add_sublayer('bn', bn)
        self.deconv.add_sublayer('relu6_2', nn.ReLU6())
        self.deconv.add_sublayer('conv3', conv3)
        self.deconv.add_sublayer('relu6_3', nn.ReLU6())

    def forward(self, inputs):
        return self.deconv(inputs)


class LiteUpsample(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type='bn'):
        super(LiteUpsample, self).__init__()
        self.deconv = DeConv(ch_in, ch_out, norm_type=norm_type)
        self.conv = LiteConv(ch_in, ch_out, norm_type=norm_type)

    def forward(self, inputs):
        deconv_up = self.deconv(inputs)
        conv = self.conv(inputs)
        interp_up = F.interpolate(conv, scale_factor=2., mode='bilinear')
        return deconv_up + interp_up


class ShortCut(nn.Layer):
    def __init__(self,
                 layer_num,
                 ch_in,
                 ch_out,
                 norm_type='bn',
                 lite_neck=False,
                 name=None):
        super(ShortCut, self).__init__()
        shortcut_conv = nn.Sequential()
        for i in range(layer_num):
            fan_out = 3 * 3 * ch_out
            std = math.sqrt(2. / fan_out)
            in_channels = ch_in if i == 0 else ch_out
            shortcut_name = name + '.conv.{}'.format(i)
            if lite_neck:
                shortcut_conv.add_sublayer(
                    shortcut_name,
                    LiteConv(
                        in_channels=in_channels,
                        out_channels=ch_out,
                        with_act=i < layer_num - 1,
                        norm_type=norm_type))
            else:
                shortcut_conv.add_sublayer(
                    shortcut_name,
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=ch_out,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(initializer=Normal(0, std)),
                        bias_attr=ParamAttr(
                            learning_rate=2., regularizer=L2Decay(0.))))
                if i < layer_num - 1:
                    shortcut_conv.add_sublayer(shortcut_name + '.act',
                                               nn.ReLU())
        self.shortcut = self.add_sublayer('shortcut', shortcut_conv)

    def forward(self, feat):
        out = self.shortcut(feat)
        return out


@register
@serializable
class TTFFPN(nn.Layer):
    """
    Args:
        in_channels (list): number of input feature channels from backbone.
            [128,256,512,1024] by default, means the channels of DarkNet53
            backbone return_idx [1,2,3,4].
        planes (list): the number of output feature channels of FPN.
            [256, 128, 64] by default
        shortcut_num (list): the number of convolution layers in each shortcut.
            [3,2,1] by default, means DarkNet53 backbone return_idx_1 has 3 convs
            in its shortcut, return_idx_2 has 2 convs and return_idx_3 has 1 conv.
        norm_type (string): norm type, 'sync_bn', 'bn', 'gn' are optional. 
            bn by default
        lite_neck (bool): whether to use lite conv in TTFNet FPN, 
            False by default
        fusion_method (string): the method to fusion upsample and lateral layer.
            'add' and 'concat' are optional, add by default
    """

    __shared__ = ['norm_type']

    def __init__(self,
                 in_channels,
                 planes=[256, 128, 64],
                 shortcut_num=[3, 2, 1],
                 norm_type='bn',
                 lite_neck=False,
                 fusion_method='add'):
        super(TTFFPN, self).__init__()
        self.planes = planes
        self.shortcut_num = shortcut_num[::-1]
        self.shortcut_len = len(shortcut_num)
        self.ch_in = in_channels[::-1]
        self.fusion_method = fusion_method

        self.upsample_list = []
        self.shortcut_list = []
        self.upper_list = []
        for i, out_c in enumerate(self.planes):
            in_c = self.ch_in[i] if i == 0 else self.upper_list[-1]
            upsample_module = LiteUpsample if lite_neck else Upsample
            upsample = self.add_sublayer(
                'upsample.' + str(i),
                upsample_module(
                    in_c, out_c, norm_type=norm_type))
            self.upsample_list.append(upsample)
            if i < self.shortcut_len:
                shortcut = self.add_sublayer(
                    'shortcut.' + str(i),
                    ShortCut(
                        self.shortcut_num[i],
                        self.ch_in[i + 1],
                        out_c,
                        norm_type=norm_type,
                        lite_neck=lite_neck,
                        name='shortcut.' + str(i)))
                self.shortcut_list.append(shortcut)
                if self.fusion_method == 'add':
                    upper_c = out_c
                elif self.fusion_method == 'concat':
                    upper_c = out_c * 2
                else:
                    raise ValueError('Illegal fusion method. Expected add or\
                        concat, but received {}'.format(self.fusion_method))
                self.upper_list.append(upper_c)

    def forward(self, inputs):
        feat = inputs[-1]
        for i, out_c in enumerate(self.planes):
            feat = self.upsample_list[i](feat)
            if i < self.shortcut_len:
                shortcut = self.shortcut_list[i](inputs[-i - 2])
                if self.fusion_method == 'add':
                    feat = feat + shortcut
                else:
                    feat = paddle.concat([feat, shortcut], axis=1)
        return feat

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.upper_list[-1], )]
