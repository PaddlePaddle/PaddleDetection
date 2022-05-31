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

import collections
import math
import re

from paddle import fluid
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register

__all__ = ['EfficientNet']

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'width_coefficient',
    'depth_coefficient', 'depth_divisor'
])

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'stride', 'se_ratio'
])

GlobalParams.__new__.__defaults__ = (None, ) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)


def _decode_block_string(block_string):
    assert isinstance(block_string, str)

    ops = block_string.split('_')
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    assert (('s' in options and len(options['s']) == 1) or
            (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        se_ratio=float(options['se']) if 'se' in options else None,
        stride=int(options['s'][0]))


def get_model_params(scale):
    block_strings = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    block_args = []
    for block_string in block_strings:
        block_args.append(_decode_block_string(block_string))

    params_dict = {
        # width, depth
        'b0': (1.0, 1.0),
        'b1': (1.0, 1.1),
        'b2': (1.1, 1.2),
        'b3': (1.2, 1.4),
        'b4': (1.4, 1.8),
        'b5': (1.6, 2.2),
        'b6': (1.8, 2.6),
        'b7': (2.0, 3.1),
    }

    w, d = params_dict[scale]

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        width_coefficient=w,
        depth_coefficient=d,
        depth_divisor=8)

    return block_args, global_params


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    filters *= multiplier
    min_depth = divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def conv2d(inputs,
           num_filters,
           filter_size,
           stride=1,
           padding='SAME',
           groups=1,
           use_bias=False,
           name='conv2d'):
    param_attr = fluid.ParamAttr(name=name + '_weights')
    bias_attr = False
    if use_bias:
        bias_attr = fluid.ParamAttr(
            name=name + '_offset', regularizer=L2Decay(0.))
    feats = fluid.layers.conv2d(
        inputs,
        num_filters,
        filter_size,
        groups=groups,
        name=name,
        stride=stride,
        padding=padding,
        param_attr=param_attr,
        bias_attr=bias_attr)
    return feats


def batch_norm(inputs, momentum, eps, name=None):
    param_attr = fluid.ParamAttr(name=name + '_scale', regularizer=L2Decay(0.))
    bias_attr = fluid.ParamAttr(name=name + '_offset', regularizer=L2Decay(0.))
    return fluid.layers.batch_norm(
        input=inputs,
        momentum=momentum,
        epsilon=eps,
        name=name,
        moving_mean_name=name + '_mean',
        moving_variance_name=name + '_variance',
        param_attr=param_attr,
        bias_attr=bias_attr)


def mb_conv_block(inputs,
                  input_filters,
                  output_filters,
                  expand_ratio,
                  kernel_size,
                  stride,
                  momentum,
                  eps,
                  se_ratio=None,
                  name=None):
    feats = inputs
    num_filters = input_filters * expand_ratio

    if expand_ratio != 1:
        feats = conv2d(feats, num_filters, 1, name=name + '_expand_conv')
        feats = batch_norm(feats, momentum, eps, name=name + '_bn0')
        feats = fluid.layers.swish(feats)

    feats = conv2d(
        feats,
        num_filters,
        kernel_size,
        stride,
        groups=num_filters,
        name=name + '_depthwise_conv')
    feats = batch_norm(feats, momentum, eps, name=name + '_bn1')
    feats = fluid.layers.swish(feats)

    if se_ratio is not None:
        filter_squeezed = max(1, int(input_filters * se_ratio))
        squeezed = fluid.layers.pool2d(
            feats, pool_type='avg', global_pooling=True)
        squeezed = conv2d(
            squeezed,
            filter_squeezed,
            1,
            use_bias=True,
            name=name + '_se_reduce')
        squeezed = fluid.layers.swish(squeezed)
        squeezed = conv2d(
            squeezed, num_filters, 1, use_bias=True, name=name + '_se_expand')
        feats = feats * fluid.layers.sigmoid(squeezed)

    feats = conv2d(feats, output_filters, 1, name=name + '_project_conv')
    feats = batch_norm(feats, momentum, eps, name=name + '_bn2')

    if stride == 1 and input_filters == output_filters:
        feats = fluid.layers.elementwise_add(feats, inputs)

    return feats


@register
class EfficientNet(object):
    """
    EfficientNet, see https://arxiv.org/abs/1905.11946

    Args:
        scale (str): compounding scale factor, 'b0' - 'b7'.
        use_se (bool): use squeeze and excite module.
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported
    """
    __shared__ = ['norm_type']

    def __init__(self, scale='b0', use_se=True, norm_type='bn'):
        assert scale in ['b' + str(i) for i in range(8)], \
            "valid scales are b0 - b7"
        assert norm_type in ['bn', 'sync_bn'], \
            "only 'bn' and 'sync_bn' are supported"

        super(EfficientNet, self).__init__()
        self.norm_type = norm_type
        self.scale = scale
        self.use_se = use_se

    def __call__(self, inputs):
        blocks_args, global_params = get_model_params(self.scale)
        momentum = global_params.batch_norm_momentum
        eps = global_params.batch_norm_epsilon

        num_filters = round_filters(32, global_params)
        feats = conv2d(
            inputs,
            num_filters=num_filters,
            filter_size=3,
            stride=2,
            name='_conv_stem')
        feats = batch_norm(feats, momentum=momentum, eps=eps, name='_bn0')
        feats = fluid.layers.swish(feats)

        layer_count = 0
        feature_maps = []

        for b, block_arg in enumerate(blocks_args):
            for r in range(block_arg.num_repeat):
                input_filters = round_filters(block_arg.input_filters,
                                              global_params)
                output_filters = round_filters(block_arg.output_filters,
                                               global_params)
                kernel_size = block_arg.kernel_size
                stride = block_arg.stride
                se_ratio = None
                if self.use_se:
                    se_ratio = block_arg.se_ratio

                if r > 0:
                    input_filters = output_filters
                    stride = 1

                feats = mb_conv_block(
                    feats,
                    input_filters,
                    output_filters,
                    block_arg.expand_ratio,
                    kernel_size,
                    stride,
                    momentum,
                    eps,
                    se_ratio=se_ratio,
                    name='_blocks.{}.'.format(layer_count))

                layer_count += 1

            feature_maps.append(feats)

        return list(feature_maps[i] for i in [2, 4, 6])
