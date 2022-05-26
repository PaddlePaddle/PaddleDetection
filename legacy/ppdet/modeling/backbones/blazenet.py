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

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['BlazeNet']


@register
class BlazeNet(object):
    """
    BlazeFace, see https://arxiv.org/abs/1907.05047

    Args:
        blaze_filters (list): number of filter for each blaze block
        double_blaze_filters (list): number of filter for each double_blaze block
        with_extra_blocks (bool): whether or not extra blocks should be added
        lite_edition (bool): whether or not is blazeface-lite
        use_5x5kernel (bool): whether or not filter size is 5x5 in depth-wise conv
    """

    def __init__(
            self,
            blaze_filters=[[24, 24], [24, 24], [24, 48, 2], [48, 48], [48, 48]],
            double_blaze_filters=[[48, 24, 96, 2], [96, 24, 96], [96, 24, 96],
                                  [96, 24, 96, 2], [96, 24, 96], [96, 24, 96]],
            with_extra_blocks=True,
            lite_edition=False,
            use_5x5kernel=True):
        super(BlazeNet, self).__init__()

        self.blaze_filters = blaze_filters
        self.double_blaze_filters = double_blaze_filters
        self.with_extra_blocks = with_extra_blocks
        self.lite_edition = lite_edition
        self.use_5x5kernel = use_5x5kernel

    def __call__(self, input):
        if not self.lite_edition:
            conv1_num_filters = self.blaze_filters[0][0]
            conv = self._conv_norm(
                input=input,
                num_filters=conv1_num_filters,
                filter_size=3,
                stride=2,
                padding=1,
                act='relu',
                name="conv1")

            for k, v in enumerate(self.blaze_filters):
                assert len(v) in [2, 3], \
                    "blaze_filters {} not in [2, 3]"
                if len(v) == 2:
                    conv = self.BlazeBlock(
                        conv,
                        v[0],
                        v[1],
                        use_5x5kernel=self.use_5x5kernel,
                        name='blaze_{}'.format(k))
                elif len(v) == 3:
                    conv = self.BlazeBlock(
                        conv,
                        v[0],
                        v[1],
                        stride=v[2],
                        use_5x5kernel=self.use_5x5kernel,
                        name='blaze_{}'.format(k))

            layers = []
            for k, v in enumerate(self.double_blaze_filters):
                assert len(v) in [3, 4], \
                    "blaze_filters {} not in [3, 4]"
                if len(v) == 3:
                    conv = self.BlazeBlock(
                        conv,
                        v[0],
                        v[1],
                        double_channels=v[2],
                        use_5x5kernel=self.use_5x5kernel,
                        name='double_blaze_{}'.format(k))
                elif len(v) == 4:
                    layers.append(conv)
                    conv = self.BlazeBlock(
                        conv,
                        v[0],
                        v[1],
                        double_channels=v[2],
                        stride=v[3],
                        use_5x5kernel=self.use_5x5kernel,
                        name='double_blaze_{}'.format(k))
            layers.append(conv)

            if not self.with_extra_blocks:
                return layers[-1]
            return layers[-2], layers[-1]
        else:
            conv1 = self._conv_norm(
                input=input,
                num_filters=24,
                filter_size=5,
                stride=2,
                padding=2,
                act='relu',
                name="conv1")
            conv2 = self.Blaze_lite(conv1, 24, 24, 1, 'conv2')
            conv3 = self.Blaze_lite(conv2, 24, 28, 1, 'conv3')
            conv4 = self.Blaze_lite(conv3, 28, 32, 2, 'conv4')
            conv5 = self.Blaze_lite(conv4, 32, 36, 1, 'conv5')
            conv6 = self.Blaze_lite(conv5, 36, 42, 1, 'conv6')
            conv7 = self.Blaze_lite(conv6, 42, 48, 2, 'conv7')
            in_ch = 48
            for i in range(5):
                conv7 = self.Blaze_lite(conv7, in_ch, in_ch + 8, 1,
                                        'conv{}'.format(8 + i))
                in_ch += 8
            assert in_ch == 88
            conv13 = self.Blaze_lite(conv7, 88, 96, 2, 'conv13')
            for i in range(4):
                conv13 = self.Blaze_lite(conv13, 96, 96, 1,
                                         'conv{}'.format(14 + i))

            return conv7, conv13

    def BlazeBlock(self,
                   input,
                   in_channels,
                   out_channels,
                   double_channels=None,
                   stride=1,
                   use_5x5kernel=True,
                   name=None):
        assert stride in [1, 2]
        use_pool = not stride == 1
        use_double_block = double_channels is not None
        act = 'relu' if use_double_block else None
        mixed_precision_enabled = mixed_precision_global_state() is not None

        if use_5x5kernel:
            conv_dw = self._conv_norm(
                input=input,
                filter_size=5,
                num_filters=in_channels,
                stride=stride,
                padding=2,
                num_groups=in_channels,
                use_cudnn=mixed_precision_enabled,
                name=name + "1_dw")
        else:
            conv_dw_1 = self._conv_norm(
                input=input,
                filter_size=3,
                num_filters=in_channels,
                stride=1,
                padding=1,
                num_groups=in_channels,
                use_cudnn=mixed_precision_enabled,
                name=name + "1_dw_1")
            conv_dw = self._conv_norm(
                input=conv_dw_1,
                filter_size=3,
                num_filters=in_channels,
                stride=stride,
                padding=1,
                num_groups=in_channels,
                use_cudnn=mixed_precision_enabled,
                name=name + "1_dw_2")

        conv_pw = self._conv_norm(
            input=conv_dw,
            filter_size=1,
            num_filters=out_channels,
            stride=1,
            padding=0,
            act=act,
            name=name + "1_sep")

        if use_double_block:
            if use_5x5kernel:
                conv_dw = self._conv_norm(
                    input=conv_pw,
                    filter_size=5,
                    num_filters=out_channels,
                    stride=1,
                    padding=2,
                    use_cudnn=mixed_precision_enabled,
                    name=name + "2_dw")
            else:
                conv_dw_1 = self._conv_norm(
                    input=conv_pw,
                    filter_size=3,
                    num_filters=out_channels,
                    stride=1,
                    padding=1,
                    num_groups=out_channels,
                    use_cudnn=mixed_precision_enabled,
                    name=name + "2_dw_1")
                conv_dw = self._conv_norm(
                    input=conv_dw_1,
                    filter_size=3,
                    num_filters=out_channels,
                    stride=1,
                    padding=1,
                    num_groups=out_channels,
                    use_cudnn=mixed_precision_enabled,
                    name=name + "2_dw_2")

            conv_pw = self._conv_norm(
                input=conv_dw,
                filter_size=1,
                num_filters=double_channels,
                stride=1,
                padding=0,
                name=name + "2_sep")

        # shortcut
        if use_pool:
            shortcut_channel = double_channels or out_channels
            shortcut_pool = self._pooling_block(input, stride, stride)
            channel_pad = self._conv_norm(
                input=shortcut_pool,
                filter_size=1,
                num_filters=shortcut_channel,
                stride=1,
                padding=0,
                name="shortcut" + name)
            return fluid.layers.elementwise_add(
                x=channel_pad, y=conv_pw, act='relu')
        return fluid.layers.elementwise_add(x=input, y=conv_pw, act='relu')

    def Blaze_lite(self, input, in_channels, out_channels, stride=1, name=None):
        assert stride in [1, 2]
        use_pool = not stride == 1
        ues_pad = not in_channels == out_channels
        conv_dw = self._conv_norm(
            input=input,
            filter_size=3,
            num_filters=in_channels,
            stride=stride,
            padding=1,
            num_groups=in_channels,
            name=name + "_dw")

        conv_pw = self._conv_norm(
            input=conv_dw,
            filter_size=1,
            num_filters=out_channels,
            stride=1,
            padding=0,
            name=name + "_sep")

        if use_pool:
            shortcut_pool = self._pooling_block(input, stride, stride)
        if ues_pad:
            conv_pad = shortcut_pool if use_pool else input
            channel_pad = self._conv_norm(
                input=conv_pad,
                filter_size=1,
                num_filters=out_channels,
                stride=1,
                padding=0,
                name="shortcut" + name)
            return fluid.layers.elementwise_add(
                x=channel_pad, y=conv_pw, act='relu')
        return fluid.layers.elementwise_add(x=input, y=conv_pw, act='relu')

    def _conv_norm(
            self,
            input,
            filter_size,
            num_filters,
            stride,
            padding,
            num_groups=1,
            act='relu',  # None
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
