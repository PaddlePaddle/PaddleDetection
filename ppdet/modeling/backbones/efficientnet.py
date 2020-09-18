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
    'depth_coefficient', 'depth_divisor', 'min_depth', 'drop_connect_rate',
    'relu_fn', 'batch_norm', 'use_se', 'local_pooling', 'condconv_num_experts',
    'clip_projection_output', 'blocks_args', 'fix_head_stem'
])

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
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
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        stride=int(options['s'][0]),
        conv_type=int(options['c']) if 'c' in options else 0,
        fused_conv=int(options['f']) if 'f' in options else 0,
        super_pixel=int(options['p']) if 'p' in options else 0,
        condconv=('cc' in block_string))


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
        'l2': (4.3, 5.3),
    }

    w, d = params_dict[scale]

    global_params = GlobalParams(
        blocks_args=block_strings,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=0 if scale == 'b0' else 0.2,
        width_coefficient=w,
        depth_coefficient=d,
        depth_divisor=8,
        min_depth=None,
        fix_head_stem=False,
        use_se=True,
        clip_projection_output=False)

    return block_args, global_params


def round_filters(filters, global_params, skip=False):
    multiplier = global_params.width_coefficient
    if skip or not multiplier:
        return filters
    divisor = global_params.depth_divisor
    filters *= multiplier
    min_depth = global_params.min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params, skip=False):
    multiplier = global_params.depth_coefficient
    if skip or not multiplier:
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


def _drop_connect(inputs, prob, mode):
    if mode != 'train':
        return inputs
    keep_prob = 1.0 - prob
    inputs_shape = fluid.layers.shape(inputs)
    random_tensor = keep_prob + fluid.layers.uniform_random(shape=[inputs_shape[0], 1, 1, 1], min=0., max=1.)
    binary_tensor = fluid.layers.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def mb_conv_block(inputs,
                  input_filters,
                  output_filters,
                  expand_ratio,
                  kernel_size,
                  stride,
                  id_skip,
                  drop_connect_rate,
                  momentum,
                  eps,
                  mode,
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

    if id_skip and stride == 1 and input_filters == output_filters:
        if drop_connect_rate:
            feats = _drop_connect(feats, drop_connect_rate, mode)
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

    def __call__(self, inputs, mode):
        assert mode in ['train', 'test'], \
            "only 'train' and 'test' mode are supported"
        blocks_args, global_params = get_model_params(self.scale)
        momentum = global_params.batch_norm_momentum
        eps = global_params.batch_norm_epsilon

        num_filters = round_filters(blocks_args[0].input_filters, global_params, global_params.fix_head_stem)
        feats = conv2d(
            inputs,
            num_filters=num_filters,
            filter_size=3,
            stride=2,
            name='_conv_stem')
        feats = batch_norm(feats, momentum=momentum, eps=eps, name='_bn0')
        feats = fluid.layers.swish(feats)

        layer_count = 0
        num_blocks = sum([block_arg.num_repeat for block_arg in blocks_args])
        feature_maps = []

        for block_arg in blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_arg = block_arg._replace(
                input_filters=round_filters(block_arg.input_filters,
                                            global_params),
                output_filters=round_filters(block_arg.output_filters,
                                             global_params),
                num_repeat=round_repeats(block_arg.num_repeat,
                                         global_params))

            # The first block needs to take care of stride,
            # and filter size increase.
            drop_connect_rate = global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(layer_count) / num_blocks
            feats = mb_conv_block(
                feats,
                block_arg.input_filters,
                block_arg.output_filters,
                block_arg.expand_ratio,
                block_arg.kernel_size,
                block_arg.stride,
                block_arg.id_skip,
                drop_connect_rate,
                momentum,
                eps,
                mode,
                se_ratio=block_arg.se_ratio,
                name='_blocks.{}.'.format(layer_count))
            layer_count += 1

            # Other block
            if block_arg.num_repeat > 1:
                block_arg = block_arg._replace(input_filters=block_arg.output_filters, stride=1)

            for _ in range(block_arg.num_repeat - 1):
                drop_connect_rate = global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(layer_count) / num_blocks
                feats = mb_conv_block(
                    feats,
                    block_arg.input_filters,
                    block_arg.output_filters,
                    block_arg.expand_ratio,
                    block_arg.kernel_size,
                    block_arg.stride,
                    block_arg.id_skip,
                    drop_connect_rate,
                    momentum,
                    eps,
                    mode,
                    se_ratio=block_arg.se_ratio,
                    name='_blocks.{}.'.format(layer_count))

                layer_count += 1

            feature_maps.append(feats)

        return list(feature_maps[i] for i in [2, 4, 6])
