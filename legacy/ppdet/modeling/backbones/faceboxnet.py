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

__all__ = ['FaceBoxNet']


@register
class FaceBoxNet(object):
    """
    FaceBoxes, see https://https://arxiv.org/abs/1708.05234

    Args:
        with_extra_blocks (bool): whether or not extra blocks should be added
        lite_edition (bool): whether or not is FaceBoxes-lite
    """

    def __init__(self, with_extra_blocks=True, lite_edition=False):
        super(FaceBoxNet, self).__init__()

        self.with_extra_blocks = with_extra_blocks
        self.lite_edition = lite_edition

    def __call__(self, input):
        if self.lite_edition:
            return self._simplified_edition(input)
        else:
            return self._original_edition(input)

    def _simplified_edition(self, input):
        conv_1_1 = self._conv_norm_crelu(
            input=input,
            num_filters=8,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="conv_1_1")

        conv_1_2 = self._conv_norm_crelu(
            input=conv_1_1,
            num_filters=24,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="conv_1_2")

        pool1 = fluid.layers.pool2d(
            input=conv_1_2,
            pool_size=3,
            pool_padding=1,
            pool_type='avg',
            name="pool_1")

        conv_2_1 = self._conv_norm(
            input=pool1,
            num_filters=48,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="conv_2_1")

        conv_2_2 = self._conv_norm(
            input=conv_2_1,
            num_filters=64,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_2_2")

        conv_inception = conv_2_2

        for i in range(3):
            conv_inception = self._inceptionA(conv_inception, i)

        layers = []
        layers.append(conv_inception)

        conv_3_1 = self._conv_norm(
            input=conv_inception,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_3_1")

        conv_3_2 = self._conv_norm(
            input=conv_3_1,
            num_filters=256,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="conv_3_2")

        layers.append(conv_3_2)

        if not self.with_extra_blocks:
            return layers[-1]
        return layers[-2], layers[-1]

    def _original_edition(self, input):
        conv_1 = self._conv_norm_crelu(
            input=input,
            num_filters=24,
            filter_size=7,
            stride=4,
            padding=3,
            act='relu',
            name="conv_1")

        pool_1 = fluid.layers.pool2d(
            input=conv_1,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max',
            name="pool_1")

        conv_2 = self._conv_norm_crelu(
            input=pool_1,
            num_filters=64,
            filter_size=5,
            stride=2,
            padding=2,
            act='relu',
            name="conv_2")

        pool_2 = fluid.layers.pool2d(
            input=conv_1,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max',
            name="pool_2")

        conv_inception = pool_2

        for i in range(3):
            conv_inception = self._inceptionA(conv_inception, i)

        layers = []
        layers.append(conv_inception)

        conv_3_1 = self._conv_norm(
            input=conv_inception,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_3_1")

        conv_3_2 = self._conv_norm(
            input=conv_3_1,
            num_filters=256,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="conv_3_2")

        layers.append(conv_3_2)

        conv_4_1 = self._conv_norm(
            input=conv_3_2,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu',
            name="conv_4_1")

        conv_4_2 = self._conv_norm(
            input=conv_4_1,
            num_filters=256,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="conv_4_2")

        layers.append(conv_4_2)

        if not self.with_extra_blocks:
            return layers[-1]

        return layers[-3], layers[-2], layers[-1]

    def _conv_norm(self,
                   input,
                   filter_size,
                   num_filters,
                   stride,
                   padding,
                   num_groups=1,
                   act='relu',
                   use_cudnn=True,
                   name=None):
        parameter_attr = ParamAttr(
            learning_rate=0.1,
            initializer=fluid.initializer.MSRA(),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def _conv_norm_crelu(self,
                         input,
                         filter_size,
                         num_filters,
                         stride,
                         padding,
                         num_groups=1,
                         act='relu',
                         use_cudnn=True,
                         name=None):
        parameter_attr = ParamAttr(
            learning_rate=0.1,
            initializer=fluid.initializer.MSRA(),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)

        conv_a = fluid.layers.batch_norm(input=conv, act=act)
        conv_b = fluid.layers.scale(conv_a, -1)

        concat = fluid.layers.concat([conv_a, conv_b], axis=1)

        return concat

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

    def _inceptionA(self, data, idx):
        idx = str(idx)

        pool1 = fluid.layers.pool2d(
            input=data,
            pool_size=3,
            pool_padding=1,
            pool_type='avg',
            name='inceptionA_' + idx + '_pool1')
        conv1 = self._conv_norm(
            input=pool1,
            filter_size=1,
            num_filters=32,
            stride=1,
            padding=0,
            act='relu',
            name='inceptionA_' + idx + '_conv1')

        conv2 = self._conv_norm(
            input=data,
            filter_size=1,
            num_filters=32,
            stride=1,
            padding=0,
            act='relu',
            name='inceptionA_' + idx + '_conv2')

        conv3 = self._conv_norm(
            input=data,
            filter_size=1,
            num_filters=24,
            stride=1,
            padding=0,
            act='relu',
            name='inceptionA_' + idx + '_conv3_1')
        conv3 = self._conv_norm(
            input=conv3,
            filter_size=3,
            num_filters=32,
            stride=1,
            padding=1,
            act='relu',
            name='inceptionA_' + idx + '_conv3_2')

        conv4 = self._conv_norm(
            input=data,
            filter_size=1,
            num_filters=24,
            stride=1,
            padding=0,
            act='relu',
            name='inceptionA_' + idx + '_conv4_1')
        conv4 = self._conv_norm(
            input=conv4,
            filter_size=3,
            num_filters=32,
            stride=1,
            padding=1,
            act='relu',
            name='inceptionA_' + idx + '_conv4_2')
        conv4 = self._conv_norm(
            input=conv4,
            filter_size=3,
            num_filters=32,
            stride=1,
            padding=1,
            act='relu',
            name='inceptionA_' + idx + '_conv4_3')

        concat = fluid.layers.concat([conv1, conv2, conv3, conv4], axis=1)

        return concat
