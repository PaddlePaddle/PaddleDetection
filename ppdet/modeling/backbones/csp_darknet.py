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
from ppdet.modeling.initializer import conv_init_
from ..shape_spec import ShapeSpec

__all__ = [
    'CSPDarkNet', 'BaseConv', 'DWConv', 'BottleNeck', 'SPPLayer', 'SPPFLayer'
]


class BaseConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = x * F.sigmoid(x)
        return y


class DWConv(nn.Layer):
    """Depthwise Conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(DWConv, self).__init__()
        self.dw_conv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            bias=bias,
            act=act)
        self.pw_conv = BaseConv(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            groups=1,
            bias=bias,
            act=act)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class Focus(nn.Layer):
    """Focus width and height information into channel space, used in YOLOX."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(Focus, self).__init__()
        self.conv = BaseConv(
            in_channels * 4,
            out_channels,
            ksize=ksize,
            stride=stride,
            bias=bias,
            act=act)

    def forward(self, inputs):
        # inputs [bs, C, H, W] -> outputs [bs, 4C, W/2, H/2]
        top_left = inputs[:, :, 0::2, 0::2]
        top_right = inputs[:, :, 0::2, 1::2]
        bottom_left = inputs[:, :, 1::2, 0::2]
        bottom_right = inputs[:, :, 1::2, 1::2]
        outputs = paddle.concat(
            [top_left, bottom_left, top_right, bottom_right], 1)
        return self.conv(outputs)


class BottleNeck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(BottleNeck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = Conv(
            hidden_channels,
            out_channels,
            ksize=3,
            stride=1,
            bias=bias,
            act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_shortcut:
            y = y + x
        return y


class SPPLayer(nn.Layer):
    """Spatial Pyramid Pooling (SPP) layer used in YOLOv3-SPP and YOLOX"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 bias=False,
                 act="silu"):
        super(SPPLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpoolings = nn.LayerList([
            nn.MaxPool2D(
                kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.concat([x] + [mp(x) for mp in self.maxpoolings], axis=1)
        x = self.conv2(x)
        return x


class SPPFLayer(nn.Layer):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer used in YOLOv5 by Glenn Jocher,
        equivalent to SPP(k=(5, 9, 13))
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=5,
                 bias=False,
                 act='silu'):
        super(SPPFLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpooling = nn.MaxPool2D(
            kernel_size=ksize, stride=1, padding=ksize // 2)
        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpooling(x)
        y2 = self.maxpooling(y1)
        y3 = self.maxpooling(y2)
        concats = paddle.concat([x, y1, y2, y3], axis=1)
        out = self.conv2(concats)
        return out


class CSPLayer(nn.Layer):
    """CSP (Cross Stage Partial) layer with 3 convs, named C3 in YOLOv5"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(* [
            BottleNeck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])
        self.conv3 = BaseConv(
            hidden_channels * 2,
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = paddle.concat([x_1, x_2], axis=1)
        x = self.conv3(x)
        return x


@register
@serializable
class CSPDarkNet(nn.Layer):
    """
    CSPDarkNet backbone.
    Args:
        arch (str): Architecture of CSPDarkNet, from {P5, P6, X}, default as X,
            and 'X' means used in YOLOX, 'P5/P6' means used in YOLOv5.
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            CSPLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        return_idx (list): Index of stages whose feature maps are returned.
    """

    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels, num_blocks, add_shortcut, use_spp(use_sppf)
    # 'X' means setting used in YOLOX, 'P5/P6' means setting used in YOLOv5.
    arch_settings = {
        'X': [[64, 128, 3, True, False], [128, 256, 9, True, False],
              [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, True, True]],
    }

    def __init__(self,
                 arch='X',
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(CSPDarkNet, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv
        arch_setting = self.arch_settings[arch]
        base_channels = int(arch_setting[0][0] * width_mult)

        # Note: differences between the latest YOLOv5 and the original YOLOX
        # 1. self.stem, use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 2. use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 3. put SPPF before(YOLOv5) or SPP after(YOLOX) the last cspdark block's CSPLayer
        # 4. whether SPPF(SPP)'CSPLayer add shortcut, True in YOLOv5, False in YOLOX
        if arch in ['P5', 'P6']:
            # in the latest YOLOv5, use Conv stem, and SPPF (fast, only single spp kernal size)
            self.stem = Conv(
                3, base_channels, ksize=6, stride=2, bias=False, act=act)
            spp_kernal_sizes = 5
        elif arch in ['X']:
            # in the original YOLOX, use Focus stem, and SPP (three spp kernal sizes)
            self.stem = Focus(
                3, base_channels, ksize=3, stride=1, bias=False, act=act)
            spp_kernal_sizes = (5, 9, 13)
        else:
            raise AttributeError("Unsupported arch type: {}".format(arch))

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, shortcut,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            conv_layer = self.add_sublayer(
                'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                Conv(
                    in_channels, out_channels, 3, 2, bias=False, act=act))
            stage.append(conv_layer)
            layers_num += 1

            if use_spp and arch in ['X']:
                # in YOLOX use SPPLayer
                spp_layer = self.add_sublayer(
                    'layers{}.stage{}.spp_layer'.format(layers_num, i + 1),
                    SPPLayer(
                        out_channels,
                        out_channels,
                        kernel_sizes=spp_kernal_sizes,
                        bias=False,
                        act=act))
                stage.append(spp_layer)
                layers_num += 1

            csp_layer = self.add_sublayer(
                'layers{}.stage{}.csp_layer'.format(layers_num, i + 1),
                CSPLayer(
                    out_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    shortcut=shortcut,
                    depthwise=depthwise,
                    bias=False,
                    act=act))
            stage.append(csp_layer)
            layers_num += 1

            if use_spp and arch in ['P5', 'P6']:
                # in latest YOLOv5 use SPPFLayer instead of SPPLayer
                sppf_layer = self.add_sublayer(
                    'layers{}.stage{}.sppf_layer'.format(layers_num, i + 1),
                    SPPFLayer(
                        out_channels,
                        out_channels,
                        ksize=5,
                        bias=False,
                        act=act))
                stage.append(sppf_layer)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_dark_blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]
