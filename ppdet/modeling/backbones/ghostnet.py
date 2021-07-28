# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import AdaptiveAvgPool2D, Linear
from paddle.nn.initializer import Uniform

from ppdet.core.workspace import register, serializable
from numbers import Integral
from ..shape_spec import ShapeSpec
from .mobilenet_v3 import make_divisible, ConvBNLayer

__all__ = ['GhostNet']


class ExtraBlockDW(nn.Layer):
    def __init__(self,
                 in_c,
                 ch_1,
                 ch_2,
                 stride,
                 lr_mult,
                 conv_decay=0.,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 name=None):
        super(ExtraBlockDW, self).__init__()
        self.pointwise_conv = ConvBNLayer(
            in_c=in_c,
            out_c=ch_1,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu6',
            lr_mult=lr_mult,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name + "_extra1")
        self.depthwise_conv = ConvBNLayer(
            in_c=ch_1,
            out_c=ch_2,
            filter_size=3,
            stride=stride,
            padding=1,  #
            num_groups=int(ch_1),
            act='relu6',
            lr_mult=lr_mult,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name + "_extra2_dw")
        self.normal_conv = ConvBNLayer(
            in_c=ch_2,
            out_c=ch_2,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu6',
            lr_mult=lr_mult,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name + "_extra2_sep")

    def forward(self, inputs):
        x = self.pointwise_conv(inputs)
        x = self.depthwise_conv(x)
        x = self.normal_conv(x)
        return x


class SEBlock(nn.Layer):
    def __init__(self, num_channels, lr_mult, reduction_ratio=4, name=None):
        super(SEBlock, self).__init__()
        self.pool2d_gap = AdaptiveAvgPool2D(1)
        self._num_channels = num_channels
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        med_ch = num_channels // reduction_ratio
        self.squeeze = Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(
                learning_rate=lr_mult, initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_channels,
            weight_attr=ParamAttr(
                learning_rate=lr_mult, initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(learning_rate=lr_mult))

    def forward(self, inputs):
        pool = self.pool2d_gap(inputs)
        pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = paddle.clip(x=excitation, min=0, max=1)
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])
        out = paddle.multiply(inputs, excitation)
        return out


class GhostModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 output_channels,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 relu=True,
                 lr_mult=1.,
                 conv_decay=0.,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 name=None):
        super(GhostModule, self).__init__()
        init_channels = int(math.ceil(output_channels / ratio))
        new_channels = int(init_channels * (ratio - 1))
        self.primary_conv = ConvBNLayer(
            in_c=in_channels,
            out_c=init_channels,
            filter_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            num_groups=1,
            act="relu" if relu else None,
            lr_mult=lr_mult,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name + "_primary_conv")
        self.cheap_operation = ConvBNLayer(
            in_c=init_channels,
            out_c=new_channels,
            filter_size=dw_size,
            stride=1,
            padding=int((dw_size - 1) // 2),
            num_groups=init_channels,
            act="relu" if relu else None,
            lr_mult=lr_mult,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name + "_cheap_operation")

    def forward(self, inputs):
        x = self.primary_conv(inputs)
        y = self.cheap_operation(x)
        out = paddle.concat([x, y], axis=1)
        return out


class GhostBottleneck(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 output_channels,
                 kernel_size,
                 stride,
                 use_se,
                 lr_mult,
                 conv_decay=0.,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 return_list=False,
                 name=None):
        super(GhostBottleneck, self).__init__()
        self._stride = stride
        self._use_se = use_se
        self._num_channels = in_channels
        self._output_channels = output_channels
        self.return_list = return_list

        self.ghost_module_1 = GhostModule(
            in_channels=in_channels,
            output_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            relu=True,
            lr_mult=lr_mult,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name + "_ghost_module_1")
        if stride == 2:
            self.depthwise_conv = ConvBNLayer(
                in_c=hidden_dim,
                out_c=hidden_dim,
                filter_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
                num_groups=hidden_dim,
                act=None,
                lr_mult=lr_mult,
                conv_decay=conv_decay,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                name=name +
                "_depthwise_depthwise"  # looks strange due to an old typo, will be fixed later.
            )
        if use_se:
            self.se_block = SEBlock(hidden_dim, lr_mult, name=name + "_se")
        self.ghost_module_2 = GhostModule(
            in_channels=hidden_dim,
            output_channels=output_channels,
            kernel_size=1,
            relu=False,
            lr_mult=lr_mult,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name=name + "_ghost_module_2")
        if stride != 1 or in_channels != output_channels:
            self.shortcut_depthwise = ConvBNLayer(
                in_c=in_channels,
                out_c=in_channels,
                filter_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
                num_groups=in_channels,
                act=None,
                lr_mult=lr_mult,
                conv_decay=conv_decay,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                name=name +
                "_shortcut_depthwise_depthwise"  # looks strange due to an old typo, will be fixed later.
            )
            self.shortcut_conv = ConvBNLayer(
                in_c=in_channels,
                out_c=output_channels,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                act=None,
                lr_mult=lr_mult,
                conv_decay=conv_decay,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                name=name + "_shortcut_conv")

    def forward(self, inputs):
        y = self.ghost_module_1(inputs)
        x = y
        if self._stride == 2:
            x = self.depthwise_conv(x)
        if self._use_se:
            x = self.se_block(x)
        x = self.ghost_module_2(x)

        if self._stride == 1 and self._num_channels == self._output_channels:
            shortcut = inputs
        else:
            shortcut = self.shortcut_depthwise(inputs)
            shortcut = self.shortcut_conv(shortcut)
        x = paddle.add(x=x, y=shortcut)

        if self.return_list:
            return [y, x]
        else:
            return x


@register
@serializable
class GhostNet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(
            self,
            scale=1.3,
            feature_maps=[6, 12, 15],
            with_extra_blocks=False,
            extra_block_filters=[[256, 512], [128, 256], [128, 256], [64, 128]],
            lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            conv_decay=0.,
            norm_type='bn',
            norm_decay=0.0,
            freeze_norm=False):
        super(GhostNet, self).__init__()
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        if norm_type == 'sync_bn' and freeze_norm:
            raise ValueError(
                "The norm_type should not be sync_bn when freeze_norm is True")
        self.feature_maps = feature_maps
        self.with_extra_blocks = with_extra_blocks
        self.extra_block_filters = extra_block_filters

        inplanes = 16
        self.cfgs = [
            # k, t, c, SE, s
            [3, 16, 16, 0, 1],
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            [5, 72, 40, 1, 2],
            [5, 120, 40, 1, 1],
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],  # SSDLite output
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]
        self.scale = scale
        conv1_out_ch = int(make_divisible(inplanes * self.scale, 4))
        self.conv1 = ConvBNLayer(
            in_c=3,
            out_c=conv1_out_ch,
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            act="relu",
            lr_mult=1.,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            name="conv1")

        # build inverted residual blocks
        self._out_channels = []
        self.ghost_bottleneck_list = []
        idx = 0
        inplanes = conv1_out_ch
        for k, exp_size, c, use_se, s in self.cfgs:
            lr_idx = min(idx // 3, len(lr_mult_list) - 1)
            lr_mult = lr_mult_list[lr_idx]

            # for SSD/SSDLite, first head input is after ResidualUnit expand_conv
            return_list = self.with_extra_blocks and idx + 2 in self.feature_maps

            ghost_bottleneck = self.add_sublayer(
                "_ghostbottleneck_" + str(idx),
                sublayer=GhostBottleneck(
                    in_channels=inplanes,
                    hidden_dim=int(make_divisible(exp_size * self.scale, 4)),
                    output_channels=int(make_divisible(c * self.scale, 4)),
                    kernel_size=k,
                    stride=s,
                    use_se=use_se,
                    lr_mult=lr_mult,
                    conv_decay=conv_decay,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    return_list=return_list,
                    name="_ghostbottleneck_" + str(idx)))
            self.ghost_bottleneck_list.append(ghost_bottleneck)
            inplanes = int(make_divisible(c * self.scale, 4))
            idx += 1
            self._update_out_channels(
                int(make_divisible(exp_size * self.scale, 4))
                if return_list else inplanes, idx + 1, feature_maps)

        if self.with_extra_blocks:
            self.extra_block_list = []
            extra_out_c = int(make_divisible(self.scale * self.cfgs[-1][1], 4))
            lr_idx = min(idx // 3, len(lr_mult_list) - 1)
            lr_mult = lr_mult_list[lr_idx]

            conv_extra = self.add_sublayer(
                "conv" + str(idx + 2),
                sublayer=ConvBNLayer(
                    in_c=inplanes,
                    out_c=extra_out_c,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    num_groups=1,
                    act="relu6",
                    lr_mult=lr_mult,
                    conv_decay=conv_decay,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    name="conv" + str(idx + 2)))
            self.extra_block_list.append(conv_extra)
            idx += 1
            self._update_out_channels(extra_out_c, idx + 1, feature_maps)

            for j, block_filter in enumerate(self.extra_block_filters):
                in_c = extra_out_c if j == 0 else self.extra_block_filters[j -
                                                                           1][1]
                conv_extra = self.add_sublayer(
                    "conv" + str(idx + 2),
                    sublayer=ExtraBlockDW(
                        in_c,
                        block_filter[0],
                        block_filter[1],
                        stride=2,
                        lr_mult=lr_mult,
                        conv_decay=conv_decay,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        name='conv' + str(idx + 2)))
                self.extra_block_list.append(conv_extra)
                idx += 1
                self._update_out_channels(block_filter[1], idx + 1,
                                          feature_maps)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        x = self.conv1(inputs['image'])
        outs = []
        for idx, ghost_bottleneck in enumerate(self.ghost_bottleneck_list):
            x = ghost_bottleneck(x)
            if idx + 2 in self.feature_maps:
                if isinstance(x, list):
                    outs.append(x[0])
                    x = x[1]
                else:
                    outs.append(x)

        if not self.with_extra_blocks:
            return outs

        for i, block in enumerate(self.extra_block_list):
            idx = i + len(self.ghost_bottleneck_list)
            x = block(x)
            if idx + 2 in self.feature_maps:
                outs.append(x)
        return outs

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
