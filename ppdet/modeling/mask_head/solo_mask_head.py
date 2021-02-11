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

import paddle
from paddle import fluid

from ppdet.core.workspace import register
from ppdet.modeling.ops import ConvNorm, DeformConvNorm

__all__ = ['SOLOv2MaskHead']


@register
class SOLOv2MaskHead(object):
    """
    MaskHead of SOLOv2

    Args:
        in_channels (int): The channel number of input variable.
        out_channels (int): The channel number of output variable.
        start_level (int): The position where the input starts.
        end_level (int): The position where the input ends.
        use_dcn_in_tower: Whether to use dcn in tower or not.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=128,
                 start_level=0,
                 end_level=3,
                 use_dcn_in_tower=False):
        super(SOLOv2MaskHead, self).__init__()
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        self.in_channels = in_channels
        self.use_dcn_in_tower = use_dcn_in_tower
        self.conv_type = [ConvNorm, DeformConvNorm]

    def _convs_levels(self, conv_feat, level, name=None):
        conv_func = self.conv_type[0]
        if self.use_dcn_in_tower:
            conv_func = self.conv_type[1]

        if level == 0:
            return conv_func(
                input=conv_feat,
                num_filters=self.in_channels,
                filter_size=3,
                stride=1,
                norm_type='gn',
                norm_groups=32,
                freeze_norm=False,
                act='relu',
                initializer=fluid.initializer.NormalInitializer(scale=0.01),
                norm_name=name + '.conv' + str(level) + '.gn',
                name=name + '.conv' + str(level))

        for j in range(level):
            conv_feat = conv_func(
                input=conv_feat,
                num_filters=self.in_channels,
                filter_size=3,
                stride=1,
                norm_type='gn',
                norm_groups=32,
                freeze_norm=False,
                act='relu',
                initializer=fluid.initializer.NormalInitializer(scale=0.01),
                norm_name=name + '.conv' + str(j) + '.gn',
                name=name + '.conv' + str(j))
            conv_feat = fluid.layers.resize_bilinear(
                conv_feat,
                scale=2,
                name='upsample' + str(level) + str(j),
                align_corners=False,
                align_mode=0)
        return conv_feat

    def _conv_pred(self, conv_feat):
        conv_func = self.conv_type[0]
        if self.use_dcn_in_tower:
            conv_func = self.conv_type[1]
        conv_feat = conv_func(
            input=conv_feat,
            num_filters=self.out_channels,
            filter_size=1,
            stride=1,
            norm_type='gn',
            norm_groups=32,
            freeze_norm=False,
            act='relu',
            initializer=fluid.initializer.NormalInitializer(scale=0.01),
            norm_name='mask_feat_head.conv_pred.0.gn',
            name='mask_feat_head.conv_pred.0')

        return conv_feat

    def get_output(self, inputs):
        """
        Get SOLOv2MaskHead output.

        Args:
            inputs(list[Variable]): feature map from each necks with shape of [N, C, H, W]
        Returns:
            ins_pred(Variable): Output of SOLOv2MaskHead head
        """
        range_level = self.end_level - self.start_level + 1
        feature_add_all_level = self._convs_levels(
            inputs[0], 0, name='mask_feat_head.convs_all_levels.0')
        for i in range(1, range_level):
            input_p = inputs[i]
            if i == (range_level - 1):
                input_feat = input_p
                x_range = paddle.linspace(
                    -1, 1, fluid.layers.shape(input_feat)[-1], dtype='float32')
                y_range = paddle.linspace(
                    -1, 1, fluid.layers.shape(input_feat)[-2], dtype='float32')
                y, x = paddle.tensor.meshgrid([y_range, x_range])
                x = fluid.layers.unsqueeze(x, [0, 1])
                y = fluid.layers.unsqueeze(y, [0, 1])
                y = fluid.layers.expand(
                    y,
                    expand_times=[fluid.layers.shape(input_feat)[0], 1, 1, 1])
                x = fluid.layers.expand(
                    x,
                    expand_times=[fluid.layers.shape(input_feat)[0], 1, 1, 1])
                coord_feat = fluid.layers.concat([x, y], axis=1)
                input_p = fluid.layers.concat([input_p, coord_feat], axis=1)
            feature_add_all_level = fluid.layers.elementwise_add(
                feature_add_all_level,
                self._convs_levels(
                    input_p,
                    i,
                    name='mask_feat_head.convs_all_levels.{}'.format(i)))
        ins_pred = self._conv_pred(feature_add_all_level)

        return ins_pred
