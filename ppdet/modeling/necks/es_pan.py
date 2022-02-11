# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable

from ..shape_spec import ShapeSpec
from ..backbones.esnet import SEModule
from .csp_pan import ConvBNLayer, Channel_T, DPModule

__all__ = ['ESPAN']


class ES_Block(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 act='leaky_relu'):
        super(ES_Block, self).__init__()
        self._residual = ConvBNLayer(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=act)
        self._conv_pw = ConvBNLayer(
            in_channel=in_channels,
            out_channel=mid_channels // 2,
            kernel_size=1,
            stride=1,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channel=mid_channels // 2,
            out_channel=mid_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels)

        self._conv_linear = ConvBNLayer(
            in_channel=mid_channels,
            out_channel=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=act)

        self._out_conv = ConvBNLayer(
            in_channel=out_channels * 2,
            out_channel=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1 = self._residual(inputs)
        x2 = self._conv_pw(inputs)
        x3 = self._conv_dw(x2)
        x3 = paddle.concat([x2, x3], axis=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = paddle.concat([x1, x3], axis=1)
        out = self._out_conv(out)
        return out


@register
@serializable
class ESPAN(nn.Layer):
    """Path Aggregation Network with ES module.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_features (int): Number of output features of CSPPAN module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 num_features=3,
                 use_depthwise=True,
                 act='hard_swish',
                 spatial_scales=[0.125, 0.0625, 0.03125]):
        super(ESPAN, self).__init__()
        self.conv_t = Channel_T(in_channels, out_channels, act=act)
        in_channels = [out_channels] * len(spatial_scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        conv_func = DPModule if use_depthwise else ConvBNLayer

        if self.num_features == 4:
            self.first_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act)
            self.second_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act)
            self.spatial_scales.append(self.spatial_scales[-1] / 2)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                ES_Block(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    in_channels[idx - 1],
                    kernel_size=kernel_size,
                    stride=1,
                    act=act))

        # build bottom-up blocks
        self.downsamples = nn.LayerList()
        self.bottom_up_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    in_channels[idx],
                    in_channels[idx],
                    kernel_size=kernel_size,
                    stride=2,
                    act=act))
            self.bottom_up_blocks.append(
                ES_Block(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    in_channels[idx + 1],
                    kernel_size=kernel_size,
                    stride=1,
                    act=act))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], 1))
            outs.append(out)

        top_features = None
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)

        return tuple(outs)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channels, stride=1. / s)
            for s in self.spatial_scales
        ]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }
