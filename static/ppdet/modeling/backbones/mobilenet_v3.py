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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

import numpy as np
from ppdet.core.workspace import register
from numbers import Integral

__all__ = ['MobileNetV3', 'MobileNetV3RCNN']


@register
class MobileNetV3(object):
    """
    MobileNet v3, see https://arxiv.org/abs/1905.02244
    Args:
	scale (float): scaling factor for convolution groups proportion of mobilenet_v3.
        model_name (str): There are two modes, small and large.
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported.
        norm_decay (float): weight decay for normalization layer weights.
        conv_decay (float): weight decay for convolution layer weights.
        feature_maps (list): index of stages whose feature maps are returned.
        extra_block_filters (list): number of filter for each extra block.
        lr_mult_list (list): learning rate ratio of different blocks, lower learning rate ratio
                             is need for pretrained model got using distillation(default as
                             [1.0, 1.0, 1.0, 1.0, 1.0]).
        freeze_norm (bool): freeze normalization layers.
        multiplier (float): The multiplier by which to reduce the convolution expansion and
                            number of channels.
    """
    __shared__ = ['norm_type']

    def __init__(
            self,
            scale=1.0,
            model_name='small',
            feature_maps=[5, 6, 7, 8, 9, 10],
            conv_decay=0.0,
            norm_type='bn',
            norm_decay=0.0,
            extra_block_filters=[[256, 512], [128, 256], [128, 256], [64, 128]],
            lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            freeze_norm=False,
            multiplier=1.0):
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]

        if norm_type == 'sync_bn' and freeze_norm:
            raise ValueError(
                "The norm_type should not be sync_bn when freeze_norm is True")
        self.scale = scale
        self.model_name = model_name
        self.feature_maps = feature_maps
        self.extra_block_filters = extra_block_filters
        self.conv_decay = conv_decay
        self.norm_decay = norm_decay
        self.inplanes = 16
        self.end_points = []
        self.block_stride = 0

        self.lr_mult_list = lr_mult_list
        self.freeze_norm = freeze_norm
        self.norm_type = norm_type
        self.curr_stage = 0

        if model_name == "large":
            self.cfg = [
                # kernel_size, expand, channel, se_block, act_mode, stride
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # kernel_size, expand, channel, se_block, act_mode, stride
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError

        if multiplier != 1.0:
            self.cfg[-3][2] = int(self.cfg[-3][2] * multiplier)
            self.cfg[-2][1] = int(self.cfg[-2][1] * multiplier)
            self.cfg[-2][2] = int(self.cfg[-2][2] * multiplier)
            self.cfg[-1][1] = int(self.cfg[-1][1] * multiplier)
            self.cfg[-1][2] = int(self.cfg[-1][2] * multiplier)

    def _conv_bn_layer(self,
                       input,
                       filter_size,
                       num_filters,
                       stride,
                       padding,
                       num_groups=1,
                       if_act=True,
                       act=None,
                       name=None,
                       use_cudnn=True):
        lr_idx = self.curr_stage // 3
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                name=name + '_weights',
                learning_rate=lr_mult,
                regularizer=L2Decay(self.conv_decay)),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = self._bn(conv, bn_name=bn_name)

        if if_act:
            if act == 'relu':
                bn = fluid.layers.relu(bn)
            elif act == 'hard_swish':
                bn = self._hard_swish(bn)
            elif act == 'relu6':
                bn = fluid.layers.relu6(bn)
        return bn

    def _bn(self, input, act=None, bn_name=None):
        lr_idx = self.curr_stage // 3
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]
        norm_lr = 0. if self.freeze_norm else lr_mult
        norm_decay = self.norm_decay
        pattr = ParamAttr(
            name=bn_name + '_scale',
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        battr = ParamAttr(
            name=bn_name + '_offset',
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        conv = input

        if self.norm_type in ['bn', 'sync_bn']:
            global_stats = True if self.freeze_norm else False
            out = fluid.layers.batch_norm(
                input=conv,
                act=act,
                name=bn_name + '.output.1',
                param_attr=pattr,
                bias_attr=battr,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance',
                use_global_stats=global_stats)
            scale = fluid.framework._get_var(pattr.name)
            bias = fluid.framework._get_var(battr.name)
        elif self.norm_type == 'affine_channel':
            scale = fluid.layers.create_parameter(
                shape=[conv.shape[1]],
                dtype=conv.dtype,
                attr=pattr,
                default_initializer=fluid.initializer.Constant(1.))
            bias = fluid.layers.create_parameter(
                shape=[conv.shape[1]],
                dtype=conv.dtype,
                attr=battr,
                default_initializer=fluid.initializer.Constant(0.))
            out = fluid.layers.affine_channel(
                x=conv, scale=scale, bias=bias, act=act)

        if self.freeze_norm:
            scale.stop_gradient = True
            bias.stop_gradient = True

        return out

    def _hard_swish(self, x):
        return fluid.layers.elementwise_mul(x, fluid.layers.relu6(x + 3) / 6.)

    def _se_block(self, input, num_out_filter, ratio=4, name=None):
        lr_idx = self.curr_stage // 3
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]

        num_mid_filter = int(num_out_filter // ratio)
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act='relu',
            param_attr=ParamAttr(
                name=name + '_1_weights',
                learning_rate=lr_mult,
                regularizer=L2Decay(self.conv_decay)),
            bias_attr=ParamAttr(
                name=name + '_1_offset',
                learning_rate=lr_mult,
                regularizer=L2Decay(self.conv_decay)))
        conv2 = fluid.layers.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=ParamAttr(
                name=name + '_2_weights',
                learning_rate=lr_mult,
                regularizer=L2Decay(self.conv_decay)),
            bias_attr=ParamAttr(
                name=name + '_2_offset',
                learning_rate=lr_mult,
                regularizer=L2Decay(self.conv_decay)))

        scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        return scale

    def _residual_unit(self,
                       input,
                       num_in_filter,
                       num_mid_filter,
                       num_out_filter,
                       stride,
                       filter_size,
                       act=None,
                       use_se=False,
                       name=None):
        input_data = input
        conv0 = self._conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_mid_filter,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + '_expand')

        if self.block_stride == 4 and stride == 2:
            self.block_stride += 1
            if self.block_stride in self.feature_maps:
                self.end_points.append(conv0)

        with fluid.name_scope('res_conv1'):
            conv1 = self._conv_bn_layer(
                input=conv0,
                filter_size=filter_size,
                num_filters=num_mid_filter,
                stride=stride,
                padding=int((filter_size - 1) // 2),
                if_act=True,
                act=act,
                num_groups=num_mid_filter,
                use_cudnn=False,
                name=name + '_depthwise')

        if use_se:
            with fluid.name_scope('se_block'):
                conv1 = self._se_block(
                    input=conv1,
                    num_out_filter=num_mid_filter,
                    name=name + '_se')

        conv2 = self._conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear')
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return fluid.layers.elementwise_add(x=input_data, y=conv2, act=None)

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
            padding="SAME",
            act='relu6',
            name=name + "_extra1")
        depthwise_conv = self._conv_bn_layer(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2),
            stride=stride,
            padding="SAME",
            num_groups=int(num_filters1),
            act='relu6',
            use_cudnn=False,
            name=name + "_extra2_dw")
        normal_conv = self._conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2),
            stride=1,
            padding="SAME",
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
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        blocks = []

        #conv1
        conv = self._conv_bn_layer(
            input,
            filter_size=3,
            num_filters=self._make_divisible(inplanes * scale),
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        i = 0
        inplanes = self._make_divisible(inplanes * scale)
        for layer_cfg in cfg:
            if layer_cfg[5] == 2:
                self.block_stride += 1
                if self.block_stride in self.feature_maps:
                    self.end_points.append(conv)

            conv = self._residual_unit(
                input=conv,
                num_in_filter=inplanes,
                num_mid_filter=self._make_divisible(scale * layer_cfg[1]),
                num_out_filter=self._make_divisible(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2))
            inplanes = self._make_divisible(scale * layer_cfg[2])
            i += 1
            self.curr_stage += 1
        self.block_stride += 1
        if self.block_stride in self.feature_maps:
            self.end_points.append(conv)

        # extra block
        # check whether conv_extra is needed
        if self.block_stride < max(self.feature_maps):
            conv_extra = self._conv_bn_layer(
                conv,
                filter_size=1,
                num_filters=self._make_divisible(scale * cfg[-1][1]),
                stride=1,
                padding="SAME",
                num_groups=1,
                if_act=True,
                act='hard_swish',
                name='conv' + str(i + 2))
            self.block_stride += 1
            if self.block_stride in self.feature_maps:
                self.end_points.append(conv_extra)
            i += 1
        for block_filter in self.extra_block_filters:
            conv_extra = self._extra_block_dw(conv_extra, block_filter[0],
                                              block_filter[1], 2,
                                              'conv' + str(i + 2))
            self.block_stride += 1
            if self.block_stride in self.feature_maps:
                self.end_points.append(conv_extra)
            i += 1

        return OrderedDict([('mbv3_{}'.format(idx), feat)
                            for idx, feat in enumerate(self.end_points)])


@register
class MobileNetV3RCNN(MobileNetV3):
    def __init__(self,
                 scale=1.0,
                 model_name='large',
                 conv_decay=0.0,
                 norm_type='bn',
                 norm_decay=0.0,
                 freeze_norm=True,
                 feature_maps=[2, 3, 4, 5],
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(MobileNetV3RCNN, self).__init__(
            scale=scale,
            model_name=model_name,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            lr_mult_list=lr_mult_list,
            feature_maps=feature_maps,
            freeze_norm=freeze_norm)
        self.curr_stage = 0
        self.block_stride = 1

    def _residual_unit(self,
                       input,
                       num_in_filter,
                       num_mid_filter,
                       num_out_filter,
                       stride,
                       filter_size,
                       act=None,
                       use_se=False,
                       name=None):
        input_data = input
        conv0 = self._conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_mid_filter,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + '_expand')

        feature_level = int(np.log2(self.block_stride))
        if feature_level in self.feature_maps and stride == 2:
            self.end_points.append(conv0)

        conv1 = self._conv_bn_layer(
            input=conv0,
            filter_size=filter_size,
            num_filters=num_mid_filter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act,
            num_groups=num_mid_filter,
            use_cudnn=False,
            name=name + '_depthwise')

        if use_se:
            conv1 = self._se_block(
                input=conv1, num_out_filter=num_mid_filter, name=name + '_se')

        conv2 = self._conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear')
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return fluid.layers.elementwise_add(x=input_data, y=conv2, act=None)

    def __call__(self, input):
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        #conv1
        conv = self._conv_bn_layer(
            input,
            filter_size=3,
            num_filters=self._make_divisible(inplanes * scale),
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        i = 0
        inplanes = self._make_divisible(inplanes * scale)
        for layer_cfg in cfg:
            self.block_stride *= layer_cfg[5]
            conv = self._residual_unit(
                input=conv,
                num_in_filter=inplanes,
                num_mid_filter=self._make_divisible(scale * layer_cfg[1]),
                num_out_filter=self._make_divisible(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2))
            inplanes = self._make_divisible(scale * layer_cfg[2])
            i += 1
            self.curr_stage += 1

        if np.max(self.feature_maps) >= 5:
            conv = self._conv_bn_layer(
                input=conv,
                filter_size=1,
                num_filters=self._make_divisible(scale * cfg[-1][1]),
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act='hard_swish',
                name='conv_last')
            self.end_points.append(conv)
            i += 1

        res = OrderedDict([('mv3_{}'.format(idx), self.end_points[idx])
                           for idx, feat_idx in enumerate(self.feature_maps)])
        return res
