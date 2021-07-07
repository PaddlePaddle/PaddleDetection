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
from __future__ import unicode_literals

import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import ConstantInitializer


def spatial_pool(x, pooling_type, name):
    _, channel, height, width = x.shape
    if pooling_type == 'att':
        input_x = x
        # [N, 1, C, H * W]
        input_x = fluid.layers.reshape(input_x, shape=(0, 1, channel, -1))
        context_mask = fluid.layers.conv2d(
            input=x,
            num_filters=1,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=ParamAttr(name=name + "_bias"))
        # [N, 1, H * W]
        context_mask = fluid.layers.reshape(context_mask, shape=(0, 0, -1))
        # [N, 1, H * W]
        context_mask = fluid.layers.softmax(context_mask, axis=2)
        # [N, 1, H * W, 1]
        context_mask = fluid.layers.reshape(context_mask, shape=(0, 0, -1, 1))
        # [N, 1, C, 1]
        context = fluid.layers.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = fluid.layers.reshape(context, shape=(0, channel, 1, 1))
    else:
        # [N, C, 1, 1]
        context = fluid.layers.pool2d(
            input=x, pool_type='avg', global_pooling=True)
    return context


def channel_conv(input, inner_ch, out_ch, name):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=inner_ch,
        filter_size=1,
        stride=1,
        padding=0,
        param_attr=ParamAttr(name=name + "_conv1_weights"),
        bias_attr=ParamAttr(name=name + "_conv1_bias"),
        name=name + "_conv1", )
    conv = fluid.layers.layer_norm(
        conv,
        begin_norm_axis=1,
        param_attr=ParamAttr(name=name + "_ln_weights"),
        bias_attr=ParamAttr(name=name + "_ln_bias"),
        act="relu",
        name=name + "_ln")

    conv = fluid.layers.conv2d(
        input=conv,
        num_filters=out_ch,
        filter_size=1,
        stride=1,
        padding=0,
        param_attr=ParamAttr(
            name=name + "_conv2_weights",
            initializer=ConstantInitializer(value=0.0), ),
        bias_attr=ParamAttr(
            name=name + "_conv2_bias",
            initializer=ConstantInitializer(value=0.0), ),
        name=name + "_conv2")
    return conv


def add_gc_block(x,
                 ratio=1.0 / 16,
                 pooling_type='att',
                 fusion_types=['channel_add'],
                 name=None):
    '''
    GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond, see https://arxiv.org/abs/1904.11492
    Args:
        ratio (float): channel reduction ratio
        pooling_type (str): pooling type, support att and avg
        fusion_types (list): fusion types, support channel_add and channel_mul
        name (str): prefix name of gc block
    '''
    assert pooling_type in ['avg', 'att']
    assert isinstance(fusion_types, (list, tuple))
    valid_fusion_types = ['channel_add', 'channel_mul']
    assert all([f in valid_fusion_types for f in fusion_types])
    assert len(fusion_types) > 0, 'at least one fusion should be used'

    inner_ch = int(ratio * x.shape[1])
    out_ch = x.shape[1]
    context = spatial_pool(x, pooling_type, name + "_spatial_pool")
    out = x
    if 'channel_mul' in fusion_types:
        inner_out = channel_conv(context, inner_ch, out_ch, name + "_mul")
        channel_mul_term = fluid.layers.sigmoid(inner_out)
        out = out * channel_mul_term

    if 'channel_add' in fusion_types:
        channel_add_term = channel_conv(context, inner_ch, out_ch,
                                        name + "_add")
        out = out + channel_add_term

    return out
