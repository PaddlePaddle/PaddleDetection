# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register, serializable
from numbers import Integral
from paddle.fluid.initializer import MSRA
import math

__all__ = ['HRNet']


@register
@serializable
class HRNet(object):
    """
    HRNet, see https://arxiv.org/abs/1908.07919
    Args:
        width (int): network width, should be 18, 30, 32, 40, 44, 48, 60 or 64
        has_se (bool): whether contain squeeze_excitation(SE) block or not
        freeze_at (int): freeze the backbone at which stage
        norm_type (str): normalization type, 'bn'/'sync_bn'
        freeze_norm (bool): freeze normalization layers
        norm_decay (float): weight decay for normalization layer weights
        feature_maps (list): index of stages whose feature maps are returned
    """

    def __init__(self,
                 width=40,
                 has_se=False,
                 freeze_at=2,
                 norm_type='bn',
                 freeze_norm=True,
                 norm_decay=0.,
                 feature_maps=[2, 3, 4, 5]):
        super(HRNet, self).__init__()

        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]

        assert 0 <= freeze_at <= 4, "freeze_at should be 0, 1, 2, 3 or 4"
        assert len(feature_maps) > 0, "need one or more feature maps"
        assert norm_type in ['bn', 'sync_bn']

        self.width = width
        self.has_se = has_se
        self.channels = {
            18: [[18, 36], [18, 36, 72], [18, 36, 72, 144]],
            30: [[30, 60], [30, 60, 120], [30, 60, 120, 240]],
            32: [[32, 64], [32, 64, 128], [32, 64, 128, 256]],
            40: [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            44: [[44, 88], [44, 88, 176], [44, 88, 176, 352]],
            48: [[48, 96], [48, 96, 192], [48, 96, 192, 384]],
            60: [[60, 120], [60, 120, 240], [60, 120, 240, 480]],
            64: [[64, 128], [64, 128, 256], [64, 128, 256, 512]],
        }

        self.freeze_at = freeze_at
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self._model_type = 'HRNet'
        self.feature_maps = feature_maps
        self.end_points = []
        return

    def net(self, input, class_dim=1000):
        width = self.width
        channels_2, channels_3, channels_4 = self.channels[width]
        num_modules_2, num_modules_3, num_modules_4 = 1, 4, 3

        x = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=64,
            stride=2,
            if_act=True,
            name='layer1_1')
        x = self.conv_bn_layer(
            input=x,
            filter_size=3,
            num_filters=64,
            stride=2,
            if_act=True,
            name='layer1_2')

        la1 = self.layer1(x, name='layer2')
        tr1 = self.transition_layer([la1], [256], channels_2, name='tr1')
        st2 = self.stage(tr1, num_modules_2, channels_2, name='st2')
        tr2 = self.transition_layer(st2, channels_2, channels_3, name='tr2')
        st3 = self.stage(tr2, num_modules_3, channels_3, name='st3')
        tr3 = self.transition_layer(st3, channels_3, channels_4, name='tr3')
        st4 = self.stage(tr3, num_modules_4, channels_4, name='st4')

        self.end_points = st4
        return st4[-1]

    def layer1(self, input, name=None):
        conv = input
        for i in range(4):
            conv = self.bottleneck_block(
                conv,
                num_filters=64,
                downsample=True if i == 0 else False,
                name=name + '_' + str(i + 1))
        return conv

    def transition_layer(self, x, in_channels, out_channels, name=None):
        num_in = len(in_channels)
        num_out = len(out_channels)
        out = []
        for i in range(num_out):
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self.conv_bn_layer(
                        x[i],
                        filter_size=3,
                        num_filters=out_channels[i],
                        name=name + '_layer_' + str(i + 1))
                    out.append(residual)
                else:
                    out.append(x[i])
            else:
                residual = self.conv_bn_layer(
                    x[-1],
                    filter_size=3,
                    num_filters=out_channels[i],
                    stride=2,
                    name=name + '_layer_' + str(i + 1))
                out.append(residual)
        return out

    def branches(self, x, block_num, channels, name=None):
        out = []
        for i in range(len(channels)):
            residual = x[i]
            for j in range(block_num):
                residual = self.basic_block(
                    residual,
                    channels[i],
                    name=name + '_branch_layer_' + str(i + 1) + '_' +
                    str(j + 1))
            out.append(residual)
        return out

    def fuse_layers(self, x, channels, multi_scale_output=True, name=None):
        out = []
        for i in range(len(channels) if multi_scale_output else 1):
            residual = x[i]
            for j in range(len(channels)):
                if j > i:
                    y = self.conv_bn_layer(
                        x[j],
                        filter_size=1,
                        num_filters=channels[i],
                        if_act=False,
                        name=name + '_layer_' + str(i + 1) + '_' + str(j + 1))
                    y = fluid.layers.resize_nearest(input=y, scale=2**(j - i))
                    residual = fluid.layers.elementwise_add(
                        x=residual, y=y, act=None)
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            y = self.conv_bn_layer(
                                y,
                                filter_size=3,
                                num_filters=channels[i],
                                stride=2,
                                if_act=False,
                                name=name + '_layer_' + str(i + 1) + '_' +
                                str(j + 1) + '_' + str(k + 1))
                        else:
                            y = self.conv_bn_layer(
                                y,
                                filter_size=3,
                                num_filters=channels[j],
                                stride=2,
                                name=name + '_layer_' + str(i + 1) + '_' +
                                str(j + 1) + '_' + str(k + 1))
                    residual = fluid.layers.elementwise_add(
                        x=residual, y=y, act=None)

            residual = fluid.layers.relu(residual)
            out.append(residual)
        return out

    def high_resolution_module(self,
                               x,
                               channels,
                               multi_scale_output=True,
                               name=None):
        residual = self.branches(x, 4, channels, name=name)
        out = self.fuse_layers(
            residual,
            channels,
            multi_scale_output=multi_scale_output,
            name=name)
        return out

    def stage(self,
              x,
              num_modules,
              channels,
              multi_scale_output=True,
              name=None):
        out = x
        for i in range(num_modules):
            if i == num_modules - 1 and multi_scale_output == False:
                out = self.high_resolution_module(
                    out,
                    channels,
                    multi_scale_output=False,
                    name=name + '_' + str(i + 1))
            else:
                out = self.high_resolution_module(
                    out, channels, name=name + '_' + str(i + 1))

        return out

    def last_cls_out(self, x, name=None):
        out = []
        num_filters_list = [128, 256, 512, 1024]
        for i in range(len(x)):
            out.append(
                self.conv_bn_layer(
                    input=x[i],
                    filter_size=1,
                    num_filters=num_filters_list[i],
                    name=name + 'conv_' + str(i + 1)))
        return out

    def basic_block(self,
                    input,
                    num_filters,
                    stride=1,
                    downsample=False,
                    name=None):
        residual = input
        conv = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=num_filters,
            stride=stride,
            name=name + '_conv1')
        conv = self.conv_bn_layer(
            input=conv,
            filter_size=3,
            num_filters=num_filters,
            if_act=False,
            name=name + '_conv2')
        if downsample:
            residual = self.conv_bn_layer(
                input=input,
                filter_size=1,
                num_filters=num_filters,
                if_act=False,
                name=name + '_downsample')
        if self.has_se:
            conv = self.squeeze_excitation(
                input=conv,
                num_channels=num_filters,
                reduction_ratio=16,
                name='fc' + name)
        return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')

    def bottleneck_block(self,
                         input,
                         num_filters,
                         stride=1,
                         downsample=False,
                         name=None):
        residual = input
        conv = self.conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_filters,
            name=name + '_conv1')
        conv = self.conv_bn_layer(
            input=conv,
            filter_size=3,
            num_filters=num_filters,
            stride=stride,
            name=name + '_conv2')
        conv = self.conv_bn_layer(
            input=conv,
            filter_size=1,
            num_filters=num_filters * 4,
            if_act=False,
            name=name + '_conv3')
        if downsample:
            residual = self.conv_bn_layer(
                input=input,
                filter_size=1,
                num_filters=num_filters * 4,
                if_act=False,
                name=name + '_downsample')
        if self.has_se:
            conv = self.squeeze_excitation(
                input=conv,
                num_channels=num_filters * 4,
                reduction_ratio=16,
                name='fc' + name)
        return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels / reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride=1,
                      padding=1,
                      num_groups=1,
                      if_act=True,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            act=None,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = self._bn(input=conv, bn_name=bn_name)
        if if_act:
            bn = fluid.layers.relu(bn)
        return bn

    def _bn(self, input, act=None, bn_name=None):
        norm_lr = 0. if self.freeze_norm else 1.
        norm_decay = self.norm_decay
        pattr = ParamAttr(
            name=bn_name + '_scale',
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        battr = ParamAttr(
            name=bn_name + '_offset',
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        global_stats = True if self.freeze_norm else False
        out = fluid.layers.batch_norm(
            input=input,
            act=act,
            name=bn_name + '.output.1',
            param_attr=pattr,
            bias_attr=battr,
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            use_global_stats=global_stats)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
        if self.freeze_norm:
            scale.stop_gradient = True
            bias.stop_gradient = True
        return out

    def __call__(self, input):
        assert isinstance(input, Variable)
        assert not (set(self.feature_maps) - set([2, 3, 4, 5])), \
            "feature maps {} not in [2, 3, 4, 5]".format(self.feature_maps)

        res_endpoints = []

        res = input
        feature_maps = self.feature_maps
        self.net(input)

        for i in feature_maps:
            res = self.end_points[i - 2]
            if i in self.feature_maps:
                res_endpoints.append(res)
            if self.freeze_at >= i:
                res.stop_gradient = True

        return OrderedDict([('res{}_sum'.format(self.feature_maps[idx]), feat)
                            for idx, feat in enumerate(res_endpoints)])
