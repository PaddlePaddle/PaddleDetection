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
from paddle.nn import AdaptiveAvgPool2D, Linear
from paddle.regularizer import L2Decay
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Uniform
from numbers import Integral
import math

from ppdet.core.workspace import register
from ..shape_spec import ShapeSpec

__all__ = ['HRNet']


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride=1,
                 norm_type='bn',
                 norm_groups=32,
                 use_dcn=False,
                 norm_decay=0.,
                 freeze_norm=False,
                 act=None,
                 name=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn']

        self.act = act
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=False)

        norm_lr = 0. if freeze_norm else 1.

        param_attr = ParamAttr(
            learning_rate=norm_lr, regularizer=L2Decay(norm_decay))
        bias_attr = ParamAttr(
            learning_rate=norm_lr, regularizer=L2Decay(norm_decay))
        global_stats = True if freeze_norm else None
        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2D(
                ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr,
                use_global_stats=global_stats)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)
        norm_params = self.norm.parameters()
        if freeze_norm:
            for param in norm_params:
                param.stop_gradient = True

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)

        if self.act == 'relu':
            out = F.relu(out)
        return out


class Layer1(nn.Layer):
    def __init__(self,
                 num_channels,
                 has_se=False,
                 norm_decay=0.,
                 freeze_norm=True,
                 name=None):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = []

        for i in range(4):
            bottleneck_block = self.add_sublayer(
                "block_{}_{}".format(name, i + 1),
                BottleneckBlock(
                    num_channels=num_channels if i == 0 else 256,
                    num_filters=64,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    name=name + '_' + str(i + 1)))
            self.bottleneck_block_list.append(bottleneck_block)

    def forward(self, input):
        conv = input
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_decay=0.,
                 freeze_norm=True,
                 name=None):
        super(TransitionLayer, self).__init__()

        num_in = len(in_channels)
        num_out = len(out_channels)
        out = []
        self.conv_bn_func_list = []
        for i in range(num_out):
            residual = None
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self.add_sublayer(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvNormLayer(
                            ch_in=in_channels[i],
                            ch_out=out_channels[i],
                            filter_size=3,
                            norm_decay=norm_decay,
                            freeze_norm=freeze_norm,
                            act='relu',
                            name=name + '_layer_' + str(i + 1)))
            else:
                residual = self.add_sublayer(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvNormLayer(
                        ch_in=in_channels[-1],
                        ch_out=out_channels[i],
                        filter_size=3,
                        stride=2,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        act='relu',
                        name=name + '_layer_' + str(i + 1)))
            self.conv_bn_func_list.append(residual)

    def forward(self, input):
        outs = []
        for idx, conv_bn_func in enumerate(self.conv_bn_func_list):
            if conv_bn_func is None:
                outs.append(input[idx])
            else:
                if idx < len(input):
                    outs.append(conv_bn_func(input[idx]))
                else:
                    outs.append(conv_bn_func(input[-1]))
        return outs


class Branches(nn.Layer):
    def __init__(self,
                 block_num,
                 in_channels,
                 out_channels,
                 has_se=False,
                 norm_decay=0.,
                 freeze_norm=True,
                 name=None):
        super(Branches, self).__init__()

        self.basic_block_list = []
        for i in range(len(out_channels)):
            self.basic_block_list.append([])
            for j in range(block_num):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func = self.add_sublayer(
                    "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                    BasicBlock(
                        num_channels=in_ch,
                        num_filters=out_channels[i],
                        has_se=has_se,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        name=name + '_branch_layer_' + str(i + 1) + '_' +
                        str(j + 1)))
                self.basic_block_list[i].append(basic_block_func)

    def forward(self, inputs):
        outs = []
        for idx, input in enumerate(inputs):
            conv = input
            basic_block_list = self.basic_block_list[idx]
            for basic_block_func in basic_block_list:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 norm_decay=0.,
                 freeze_norm=True,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvNormLayer(
            ch_in=num_channels,
            ch_out=num_filters,
            filter_size=1,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            act="relu",
            name=name + "_conv1")
        self.conv2 = ConvNormLayer(
            ch_in=num_filters,
            ch_out=num_filters,
            filter_size=3,
            stride=stride,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            act="relu",
            name=name + "_conv2")
        self.conv3 = ConvNormLayer(
            ch_in=num_filters,
            ch_out=num_filters * 4,
            filter_size=1,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            act=None,
            name=name + "_conv3")

        if self.downsample:
            self.conv_down = ConvNormLayer(
                ch_in=num_channels,
                ch_out=num_filters * 4,
                filter_size=1,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                act=None,
                name=name + "_downsample")

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name='fc' + name)

    def forward(self, input):
        residual = input
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(input)

        if self.has_se:
            conv3 = self.se(conv3)

        y = paddle.add(x=residual, y=conv3)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 norm_decay=0.,
                 freeze_norm=True,
                 name=None):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample
        self.conv1 = ConvNormLayer(
            ch_in=num_channels,
            ch_out=num_filters,
            filter_size=3,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            stride=stride,
            act="relu",
            name=name + "_conv1")
        self.conv2 = ConvNormLayer(
            ch_in=num_filters,
            ch_out=num_filters,
            filter_size=3,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            stride=1,
            act=None,
            name=name + "_conv2")

        if self.downsample:
            self.conv_down = ConvNormLayer(
                ch_in=num_channels,
                ch_out=num_filters * 4,
                filter_size=1,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                act=None,
                name=name + "_downsample")

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name='fc' + name)

    def forward(self, input):
        residual = input
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(input)

        if self.has_se:
            conv2 = self.se(conv2)

        y = paddle.add(x=residual, y=conv2)
        y = F.relu(y)
        return y


class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_filters,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

    def forward(self, input):
        pool = self.pool2d_gap(input)
        pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])
        out = input * excitation
        return out


class Stage(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_filters,
                 has_se=False,
                 norm_decay=0.,
                 freeze_norm=True,
                 multi_scale_output=True,
                 name=None):
        super(Stage, self).__init__()

        self._num_modules = num_modules
        self.stage_func_list = []
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_filters=num_filters,
                        has_se=has_se,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1)))
            else:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_filters=num_filters,
                        has_se=has_se,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        name=name + '_' + str(i + 1)))

            self.stage_func_list.append(stage_func)

    def forward(self, input):
        out = input
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 norm_decay=0.,
                 freeze_norm=True,
                 name=None):
        super(HighResolutionModule, self).__init__()
        self.branches_func = Branches(
            block_num=4,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name)

    def forward(self, input):
        out = self.branches_func(input)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 norm_decay=0.,
                 freeze_norm=True,
                 name=None):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels

        self.residual_func_list = []
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                residual_func = None
                if j > i:
                    residual_func = self.add_sublayer(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvNormLayer(
                            ch_in=in_channels[j],
                            ch_out=out_channels[i],
                            filter_size=1,
                            stride=1,
                            act=None,
                            norm_decay=norm_decay,
                            freeze_norm=freeze_norm,
                            name=name + '_layer_' + str(i + 1) + '_' +
                            str(j + 1)))
                    self.residual_func_list.append(residual_func)
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvNormLayer(
                                    ch_in=pre_num_filters,
                                    ch_out=out_channels[i],
                                    filter_size=3,
                                    stride=2,
                                    norm_decay=norm_decay,
                                    freeze_norm=freeze_norm,
                                    act=None,
                                    name=name + '_layer_' + str(i + 1) + '_' +
                                    str(j + 1) + '_' + str(k + 1)))
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvNormLayer(
                                    ch_in=pre_num_filters,
                                    ch_out=out_channels[j],
                                    filter_size=3,
                                    stride=2,
                                    norm_decay=norm_decay,
                                    freeze_norm=freeze_norm,
                                    act="relu",
                                    name=name + '_layer_' + str(i + 1) + '_' +
                                    str(j + 1) + '_' + str(k + 1)))
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)

    def forward(self, input):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = input[i]
            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](input[j])
                    residual_func_idx += 1
                    y = F.interpolate(y, scale_factor=2**(j - i))
                    residual = paddle.add(x=residual, y=y)
                elif j < i:
                    y = input[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = paddle.add(x=residual, y=y)
            residual = F.relu(residual)
            outs.append(residual)

        return outs


@register
class HRNet(nn.Layer):
    """
    HRNet, see https://arxiv.org/abs/1908.07919

    Args:
        width (int): the width of HRNet
        has_se (bool): whether to add SE block for each stage
        freeze_at (int): the stage to freeze
        freeze_norm (bool): whether to freeze norm in HRNet
        norm_decay (float): weight decay for normalization layer weights
        return_idx (List): the stage to return
        upsample (bool): whether to upsample and concat the backbone feats
    """

    def __init__(self,
                 width=18,
                 has_se=False,
                 freeze_at=0,
                 freeze_norm=True,
                 norm_decay=0.,
                 return_idx=[0, 1, 2, 3],
                 upsample=False):
        super(HRNet, self).__init__()

        self.width = width
        self.has_se = has_se
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]

        assert len(return_idx) > 0, "need one or more return index"
        self.freeze_at = freeze_at
        self.return_idx = return_idx
        self.upsample = upsample

        self.channels = {
            18: [[18, 36], [18, 36, 72], [18, 36, 72, 144]],
            30: [[30, 60], [30, 60, 120], [30, 60, 120, 240]],
            32: [[32, 64], [32, 64, 128], [32, 64, 128, 256]],
            40: [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            44: [[44, 88], [44, 88, 176], [44, 88, 176, 352]],
            48: [[48, 96], [48, 96, 192], [48, 96, 192, 384]],
            60: [[60, 120], [60, 120, 240], [60, 120, 240, 480]],
            64: [[64, 128], [64, 128, 256], [64, 128, 256, 512]]
        }

        channels_2, channels_3, channels_4 = self.channels[width]
        num_modules_2, num_modules_3, num_modules_4 = 1, 4, 3
        self._out_channels = [sum(channels_4)] if self.upsample else channels_4
        self._out_strides = [4] if self.upsample else [4, 8, 16, 32]

        self.conv_layer1_1 = ConvNormLayer(
            ch_in=3,
            ch_out=64,
            filter_size=3,
            stride=2,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            act='relu',
            name="layer1_1")

        self.conv_layer1_2 = ConvNormLayer(
            ch_in=64,
            ch_out=64,
            filter_size=3,
            stride=2,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            act='relu',
            name="layer1_2")

        self.la1 = Layer1(
            num_channels=64,
            has_se=has_se,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name="layer2")

        self.tr1 = TransitionLayer(
            in_channels=[256],
            out_channels=channels_2,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name="tr1")

        self.st2 = Stage(
            num_channels=channels_2,
            num_modules=num_modules_2,
            num_filters=channels_2,
            has_se=self.has_se,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name="st2")

        self.tr2 = TransitionLayer(
            in_channels=channels_2,
            out_channels=channels_3,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name="tr2")

        self.st3 = Stage(
            num_channels=channels_3,
            num_modules=num_modules_3,
            num_filters=channels_3,
            has_se=self.has_se,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name="st3")

        self.tr3 = TransitionLayer(
            in_channels=channels_3,
            out_channels=channels_4,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name="tr3")
        self.st4 = Stage(
            num_channels=channels_4,
            num_modules=num_modules_4,
            num_filters=channels_4,
            has_se=self.has_se,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            multi_scale_output=len(return_idx) > 1,
            name="st4")

    def forward(self, inputs):
        x = inputs['image']
        conv1 = self.conv_layer1_1(x)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)
        tr1 = self.tr1([la1])
        st2 = self.st2(tr1)
        tr2 = self.tr2(st2)

        st3 = self.st3(tr2)
        tr3 = self.tr3(st3)

        st4 = self.st4(tr3)

        if self.upsample:
            # Upsampling
            x0_h, x0_w = st4[0].shape[2:4]
            x1 = F.upsample(st4[1], size=(x0_h, x0_w), mode='bilinear')
            x2 = F.upsample(st4[2], size=(x0_h, x0_w), mode='bilinear')
            x3 = F.upsample(st4[3], size=(x0_h, x0_w), mode='bilinear')
            x = paddle.concat([st4[0], x1, x2, x3], 1)
            return x

        res = []
        for i, layer in enumerate(st4):
            if i == self.freeze_at:
                layer.stop_gradient = True
            if i in self.return_idx:
                res.append(layer)

        return res

    @property
    def out_shape(self):
        if self.upsample:
            self.return_idx = [0]
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]
