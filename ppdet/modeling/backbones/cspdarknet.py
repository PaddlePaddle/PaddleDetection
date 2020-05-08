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

import six

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register

__all__ = ['CSPDarkNet']


@register
class CSPDarkNet(object):
    """
    CSPDarkNet, see https://arxiv.org/abs/1911.11929 
    Args:
        depth (int): network depth, currently only cspdarknet 53 is supported
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported
        norm_decay (float): weight decay for normalization layer weights
    """
    __shared__ = ['norm_type', 'weight_prefix_name']

    def __init__(self,
                 depth=53,
                 norm_type='bn',
                 norm_decay=0.,
                 weight_prefix_name=''):
        assert depth in [53], "unsupported depth value"
        self.depth = depth
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.depth_cfg = {53: ([1, 2, 8, 8, 4], self.basicblock)}
        self.prefix_name = weight_prefix_name

    def _softplus(self, input):
        expf = fluid.layers.exp(fluid.layers.clip(input, -200, 50))
        return fluid.layers.log(1 + expf)

    def _mish(self, input):
        return input * fluid.layers.tanh(self._softplus(input))

    def _conv_norm(self,
                   input,
                   ch_out,
                   filter_size,
                   stride,
                   padding,
                   act='mish',
                   name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(float(self.norm_decay)),
            name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(float(self.norm_decay)),
            name=bn_name + '.offset')

        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'mish':
            out = self._mish(out)

        return out

    def _downsample(self,
                    input,
                    ch_out,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    name=None):
        return self._conv_norm(
            input,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            name=name)

    def conv_layer(self,
                   input,
                   ch_out,
                   filter_size=1,
                   stride=1,
                   padding=0,
                   name=None):
        return self._conv_norm(
            input,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            name=name)

    def basicblock(self, input, ch_out, scale_first=False, name=None):
        conv1 = self._conv_norm(
            input,
            ch_out=ch_out // 2 if scale_first else ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            name=name + ".0")
        conv2 = self._conv_norm(
            conv1,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1,
            name=name + ".1")
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out

    def layer_warp(self,
                   block_func,
                   input,
                   ch_out,
                   count,
                   keep_ch=False,
                   scale_first=False,
                   name=None):
        if scale_first:
            ch_out = ch_out * 2
        right = self.conv_layer(
            input, ch_out, name='{}.route_in.right'.format(name))
        neck = self.conv_layer(input, ch_out, name='{}.neck'.format(name))
        out = block_func(
            neck,
            ch_out=ch_out,
            scale_first=scale_first,
            name='{}.0'.format(name))
        for j in six.moves.xrange(1, count):
            out = block_func(out, ch_out=ch_out, name='{}.{}'.format(name, j))
        left = self.conv_layer(
            out, ch_out, name='{}.route_in.left'.format(name))
        route = fluid.layers.concat([left, right], axis=1)
        out = self.conv_layer(
            route,
            ch_out=ch_out if keep_ch else ch_out * 2,
            name='{}.conv_layer'.format(name))
        return out

    def __call__(self, input):
        """
        Get the backbone of CSPDarkNet, that is output for the 5 stages.

        Args:
            input (Variable): input variable.

        Returns:
            The last variables of each stage.
        """
        stages, block_func = self.depth_cfg[self.depth]
        stages = stages[0:5]
        conv = self._conv_norm(
            input=input,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            act='mish',
            name=self.prefix_name + "conv")
        blocks = []
        for i, stage in enumerate(stages):
            input = conv if i == 0 else block
            downsample_ = self._downsample(
                input=input,
                ch_out=input.shape[1] * 2,
                name=self.prefix_name + "stage.{}.downsample".format(i))
            block = self.layer_warp(
                block_func=block_func,
                input=downsample_,
                ch_out=32 * 2**i,
                count=stage,
                keep_ch=(i == 0),
                scale_first=i == 0,
                name=self.prefix_name + "stage.{}".format(i))
            blocks.append(block)
        return blocks
