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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr

from ppdet.core.workspace import register

__all__ = ['VGG']


@register
class VGG(object):
    """
    VGG, see https://arxiv.org/abs/1409.1556

    Args:
        depth (int): the VGG net depth (16 or 19)
        normalizations (list): params list of init scale in l2 norm, skip init
            scale if param is -1.
        with_extra_blocks (bool): whether or not extra blocks should be added
        extra_block_filters (list): in each extra block, params:
            [in_channel, out_channel, padding_size, stride_size, filter_size]
    """

    def __init__(self,
                 depth=16,
                 with_extra_blocks=False,
                 normalizations=[20., -1, -1, -1, -1, -1],
                 extra_block_filters=[[256, 512, 1, 2, 3], [128, 256, 1, 2, 3],
                                      [128, 256, 0, 1, 3],
                                      [128, 256, 0, 1, 3]]):
        assert depth in [16, 19], \
            "depth {} not in [16, 19]"

        self.depth = depth
        self.depth_cfg = {16: [2, 2, 3, 3, 3], 19: [2, 2, 4, 4, 4]}
        self.with_extra_blocks = with_extra_blocks
        self.normalizations = normalizations
        self.extra_block_filters = extra_block_filters

    def __call__(self, input):
        layers = []
        layers += self._vgg_block(input)

        if not self.with_extra_blocks:
            return layers[-1]

        layers += self._add_extras_block(layers[-1])
        norm_cfg = self.normalizations
        for k, v in enumerate(layers):
            if not norm_cfg[k] == -1:
                layers[k] = self._l2_norm_scale(v, init_scale=norm_cfg[k])

        return layers

    def _vgg_block(self, input):
        nums = self.depth_cfg[self.depth]
        vgg_base = [64, 128, 256, 512, 512]
        conv = input
        layers = []
        for k, v in enumerate(vgg_base):
            conv = self._conv_block(
                conv, v, nums[k], name="conv{}_".format(k + 1))
            layers.append(conv)
            if k == 4:
                conv = self._pooling_block(conv, 3, 1, pool_padding=1)
            else:
                conv = self._pooling_block(conv, 2, 2)

        fc6 = self._conv_layer(conv, 1024, 3, 1, 6, dilation=6, name="fc6")
        fc7 = self._conv_layer(fc6, 1024, 1, 1, 0, name="fc7")

        return [layers[3], fc7]

    def _add_extras_block(self, input):
        cfg = self.extra_block_filters
        conv = input
        layers = []
        for k, v in enumerate(cfg):
            assert len(v) == 5, "extra_block_filters size not fix"
            conv = self._extra_block(
                conv,
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                name="conv{}_".format(6 + k))
            layers.append(conv)

        return layers

    def _conv_block(self, input, num_filter, groups, name=None):
        conv = input
        for i in range(groups):
            conv = self._conv_layer(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                name=name + str(i + 1))
        return conv

    def _extra_block(self,
                     input,
                     num_filters1,
                     num_filters2,
                     padding_size,
                     stride_size,
                     filter_size,
                     name=None):
        # 1x1 conv
        conv_1 = self._conv_layer(
            input=input,
            num_filters=int(num_filters1),
            filter_size=1,
            stride=1,
            act='relu',
            padding=0,
            name=name + "1")

        # 3x3 conv
        conv_2 = self._conv_layer(
            input=conv_1,
            num_filters=int(num_filters2),
            filter_size=filter_size,
            stride=stride_size,
            act='relu',
            padding=padding_size,
            name=name + "2")
        return conv_2

    def _conv_layer(self,
                    input,
                    num_filters,
                    filter_size,
                    stride,
                    padding,
                    dilation=1,
                    act='relu',
                    use_cudnn=True,
                    name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            act=act,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=ParamAttr(name=name + "_biases"),
            name=name + '.conv2d.output.1')
        return conv

    def _pooling_block(self,
                       conv,
                       pool_size,
                       pool_stride,
                       pool_padding=0,
                       ceil_mode=True):
        pool = fluid.layers.pool2d(
            input=conv,
            pool_size=pool_size,
            pool_type='max',
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            ceil_mode=ceil_mode)
        return pool

    def _l2_norm_scale(self, input, init_scale=1.0, channel_shared=False):
        from paddle.fluid.layer_helper import LayerHelper
        from paddle.fluid.initializer import Constant
        helper = LayerHelper("Scale")
        l2_norm = fluid.layers.l2_normalize(
            input, axis=1)  # l2 norm along channel
        shape = [1] if channel_shared else [input.shape[1]]
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=shape,
            dtype=input.dtype,
            default_initializer=Constant(init_scale))
        out = fluid.layers.elementwise_mul(
            x=l2_norm,
            y=scale,
            axis=-1 if channel_shared else 1,
            name="conv4_3_norm_scale")
        return out
