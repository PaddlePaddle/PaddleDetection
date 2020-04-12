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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

import math
import numpy as np
from collections import OrderedDict

from ppdet.core.workspace import register
from .mobilenet_v3 import MobileNetV3

__all__ = ['MobileNetV3RCNN']


@register
class MobileNetV3RCNN(MobileNetV3):
    def __init__(
            self,
            scale=1.0,
            model_name='large',
            with_extra_blocks=False,
            conv_decay=0.0,
            norm_type='bn',
            norm_decay=0.0,
            freeze_norm=True,
            feature_maps=[2, 3, 4, 5],
            lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0], ):
        super(MobileNetV3RCNN, self).__init__(
            scale=scale,
            model_name=model_name,
            with_extra_blocks=with_extra_blocks,
            conv_decay=conv_decay,
            norm_type=norm_type,
            norm_decay=norm_decay,
            lr_mult_list=lr_mult_list, )
        self.feature_maps = feature_maps
        self.curr_stage = 0

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
        cls_ch_squeeze = self.cls_ch_squeeze
        cls_ch_expand = self.cls_ch_expand
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
                num_filters=self._make_divisible(scale * cls_ch_squeeze),
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act='hard_swish',
                name='conv_last')
            self.end_points.append(conv)
            i += 1

        res = OrderedDict(
            [('res{}_sum'.format(self.feature_maps[idx]), self.end_points[idx])
             for idx, feat_idx in enumerate(self.feature_maps)])
        return res
