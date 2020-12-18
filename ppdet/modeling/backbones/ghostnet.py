# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from collections import OrderedDict

from ppdet.core.workspace import register

__all__ = ["GhostNet"]


@register
class GhostNet(object):
    """
    scale (float): scaling factor for convolution groups proportion of GhostNet.
    feature_maps (list): index of stages whose feature maps are returned.
    conv_decay (float): weight decay for convolution layer weights.
    extra_block_filters (list): number of filter for each extra block.
    lr_mult_list (list): learning rate ratio of different blocks, lower learning rate ratio
                             is need for pretrained model got using distillation(default as 
                             [1.0, 1.0, 1.0, 1.0, 1.0]).
    """

    def __init__(
            self,
            scale,
            feature_maps=[5, 6, 7, 8, 9, 10],
            conv_decay=0.00001,
            extra_block_filters=[[256, 512], [128, 256], [128, 256], [64, 128]],
            lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            freeze_norm=False):
        self.scale = scale
        self.feature_maps = feature_maps
        self.extra_block_filters = extra_block_filters
        self.end_points = []
        self.block_stride = 0
        self.conv_decay = conv_decay
        self.lr_mult_list = lr_mult_list
        self.freeze_norm = freeze_norm
        self.curr_stage = 0

        self.cfgs = [
            # k, t, c, se, s
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
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]

    def _conv_bn_layer(self,
                       input,
                       num_filters,
                       filter_size,
                       stride=1,
                       groups=1,
                       act=None,
                       name=None):
        lr_idx = self.curr_stage // 3
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]
        norm_lr = 0. if self.freeze_norm else lr_mult

        x = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(
                regularizer=L2Decay(self.conv_decay),
                learning_rate=lr_mult,
                initializer=fluid.initializer.MSRA(),
                name=name + "_weights"),
            bias_attr=False)
        bn_name = name + "_bn"
        x = fluid.layers.batch_norm(
            input=x,
            act=act,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                learning_rate=norm_lr,
                regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                learning_rate=norm_lr,
                regularizer=L2Decay(0.0)),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=name + "_variance")
        return x

    def se_block(self, input, num_channels, reduction_ratio=4, name=None):
        lr_idx = self.curr_stage // 3
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=ParamAttr(
                learning_rate=lr_mult,
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_1_weights'),
            bias_attr=ParamAttr(
                name=name + '_1_offset', learning_rate=lr_mult))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act=None,
            param_attr=ParamAttr(
                learning_rate=lr_mult,
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_2_weights'),
            bias_attr=ParamAttr(
                name=name + '_2_offset', learning_rate=lr_mult))
        excitation = fluid.layers.clip(x=excitation, min=0, max=1)
        se_scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return se_scale

    def depthwise_conv(self,
                       input,
                       output,
                       kernel_size,
                       stride=1,
                       relu=False,
                       name=None):
        return self._conv_bn_layer(
            input=input,
            num_filters=output,
            filter_size=kernel_size,
            stride=stride,
            groups=input.shape[1],
            act="relu" if relu else None,
            name=name + "_depthwise")

    def ghost_module(self,
                     input,
                     output,
                     kernel_size=1,
                     ratio=2,
                     dw_size=3,
                     stride=1,
                     relu=True,
                     name=None):
        self.output = output
        init_channels = int(math.ceil(output / ratio))
        new_channels = int(init_channels * (ratio - 1))
        primary_conv = self._conv_bn_layer(
            input=input,
            num_filters=init_channels,
            filter_size=kernel_size,
            stride=stride,
            groups=1,
            act="relu" if relu else None,
            name=name + "_primary_conv")
        cheap_operation = self._conv_bn_layer(
            input=primary_conv,
            num_filters=new_channels,
            filter_size=dw_size,
            stride=1,
            groups=init_channels,
            act="relu" if relu else None,
            name=name + "_cheap_operation")
        out = fluid.layers.concat([primary_conv, cheap_operation], axis=1)
        return out

    def ghost_bottleneck(self,
                         input,
                         hidden_dim,
                         output,
                         kernel_size,
                         stride,
                         use_se,
                         name=None):
        inp_channels = input.shape[1]
        x = self.ghost_module(
            input=input,
            output=hidden_dim,
            kernel_size=1,
            stride=1,
            relu=True,
            name=name + "_ghost_module_1")

        if self.block_stride == 4 and stride == 2:
            self.block_stride += 1
            if self.block_stride in self.feature_maps:
                self.end_points.append(x)

        if stride == 2:
            x = self.depthwise_conv(
                input=x,
                output=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                relu=False,
                name=name + "_depthwise")
        if use_se:
            x = self.se_block(
                input=x, num_channels=hidden_dim, name=name + "_se")
        x = self.ghost_module(
            input=x,
            output=output,
            kernel_size=1,
            relu=False,
            name=name + "_ghost_module_2")
        if stride == 1 and inp_channels == output:
            shortcut = input
        else:
            shortcut = self.depthwise_conv(
                input=input,
                output=inp_channels,
                kernel_size=kernel_size,
                stride=stride,
                relu=False,
                name=name + "_shortcut_depthwise")
            shortcut = self._conv_bn_layer(
                input=shortcut,
                num_filters=output,
                filter_size=1,
                stride=1,
                groups=1,
                act=None,
                name=name + "_shortcut_conv")
        return fluid.layers.elementwise_add(x=x, y=shortcut, axis=-1)

    def _extra_block_dw(self,
                        input,
                        num_filters1,
                        num_filters2,
                        stride,
                        name=None):
        pointwise_conv = self._conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=int(num_filters1),
            stride=1,
            act='relu6',
            name=name + "_extra1")
        depthwise_conv = self._conv_bn_layer(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2),
            stride=stride,
            groups=int(num_filters1),
            act='relu6',
            name=name + "_extra2_dw")
        normal_conv = self._conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2),
            stride=1,
            act='relu6',
            name=name + "_extra2_sep")
        return normal_conv

    def _make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def __call__(self, input):
        # build first layer
        output_channel = int(self._make_divisible(16 * self.scale, 4))
        x = self._conv_bn_layer(
            input=input,
            num_filters=output_channel,
            filter_size=3,
            stride=2,
            groups=1,
            act="relu",
            name="conv1")
        # build inverted residual blocks
        idx = 0
        for k, exp_size, c, use_se, s in self.cfgs:
            if s == 2:
                self.block_stride += 1
                if self.block_stride in self.feature_maps:
                    self.end_points.append(x)
            output_channel = int(self._make_divisible(c * self.scale, 4))
            hidden_channel = int(self._make_divisible(exp_size * self.scale, 4))
            x = self.ghost_bottleneck(
                input=x,
                hidden_dim=hidden_channel,
                output=output_channel,
                kernel_size=k,
                stride=s,
                use_se=use_se,
                name="_ghostbottleneck_" + str(idx))
            idx += 1
            self.curr_stage += 1
        self.block_stride += 1
        if self.block_stride in self.feature_maps:
            self.end_points.append(conv)

        # extra block
        # check whether conv_extra is needed
        if self.block_stride < max(self.feature_maps):
            conv_extra = self._conv_bn_layer(
                x,
                num_filters=self._make_divisible(self.scale * self.cfgs[-1][1]),
                filter_size=1,
                stride=1,
                groups=1,
                act='relu6',
                name='conv' + str(idx + 2))
            self.block_stride += 1
            if self.block_stride in self.feature_maps:
                self.end_points.append(conv_extra)
            idx += 1
        for block_filter in self.extra_block_filters:
            conv_extra = self._extra_block_dw(conv_extra, block_filter[0],
                                              block_filter[1], 2,
                                              'conv' + str(idx + 2))
            self.block_stride += 1
            if self.block_stride in self.feature_maps:
                self.end_points.append(conv_extra)
            idx += 1

        return OrderedDict([('ghost_{}'.format(idx), feat)
                            for idx, feat in enumerate(self.end_points)])
        return res
