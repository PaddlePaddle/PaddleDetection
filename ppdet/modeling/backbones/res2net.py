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

from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Constant

from ppdet.core.workspace import register, serializable
from numbers import Integral

from .nonlocal_helper import add_space_nonlocal
from .name_adapter import NameAdapter
from .resnet import ResNet, ResNetC5

__all__ = ['Res2Net', 'Res2NetC5']


@register
@serializable
class Res2Net(ResNet):
    """
    Res2Net, see https://arxiv.org/abs/1904.01169
    Args:
        depth (int): Res2Net depth, should be 50, 101, 152, 200.
        width (int): Res2Net width
        scales (int): Res2Net scale
        freeze_at (int): freeze the backbone at which stage
        norm_type (str): normalization type, 'bn'/'sync_bn'/'affine_channel'
        freeze_norm (bool): freeze normalization layers
        norm_decay (float): weight decay for normalization layer weights
        variant (str): Res2Net variant, supports 'a', 'b', 'c', 'd' currently
        feature_maps (list): index of stages whose feature maps are returned
        dcn_v2_stages (list): index of stages who select deformable conv v2
        nonlocal_stages (list): index of stages who select nonlocal networks
    """
    __shared__ = ['norm_type', 'freeze_norm', 'weight_prefix_name']

    def __init__(
            self,
            depth=50,
            width=26,
            scales=4,
            freeze_at=2,
            norm_type='bn',
            freeze_norm=True,
            norm_decay=0.,
            variant='b',
            feature_maps=[2, 3, 4, 5],
            dcn_v2_stages=[],
            weight_prefix_name='',
            nonlocal_stages=[], ):
        super(Res2Net, self).__init__(
            depth=depth,
            freeze_at=freeze_at,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            norm_decay=norm_decay,
            variant=variant,
            feature_maps=feature_maps,
            dcn_v2_stages=dcn_v2_stages,
            weight_prefix_name=weight_prefix_name,
            nonlocal_stages=nonlocal_stages)

        assert depth >= 50, "just support depth>=50 in res2net, but got depth=".format(
            depth)
        # res2net config
        self.scales = scales
        self.width = width
        basic_width = self.width * self.scales
        self.num_filters1 = [basic_width * t for t in [1, 2, 4, 8]]
        self.num_filters2 = [256 * t for t in [1, 2, 4, 8]]
        self.num_filters = [64, 128, 384, 768]

    def bottleneck(self,
                   input,
                   num_filters1,
                   num_filters2,
                   stride,
                   is_first,
                   name,
                   dcn_v2=False):
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters1,
            filter_size=1,
            stride=1,
            act='relu',
            name=name + '_branch2a')

        xs = fluid.layers.split(conv0, self.scales, 1)
        ys = []
        for s in range(self.scales - 1):
            if s == 0 or stride == 2:
                ys.append(
                    self._conv_norm(
                        input=xs[s],
                        num_filters=num_filters1 // self.scales,
                        stride=stride,
                        filter_size=3,
                        act='relu',
                        name=name + '_branch2b_' + str(s + 1),
                        dcn_v2=dcn_v2))
            else:
                ys.append(
                    self._conv_norm(
                        input=xs[s] + ys[-1],
                        num_filters=num_filters1 // self.scales,
                        stride=stride,
                        filter_size=3,
                        act='relu',
                        name=name + '_branch2b_' + str(s + 1),
                        dcn_v2=dcn_v2))

        if stride == 1:
            ys.append(xs[-1])
        else:
            ys.append(
                fluid.layers.pool2d(
                    input=xs[-1],
                    pool_size=3,
                    pool_stride=stride,
                    pool_padding=1,
                    pool_type='avg'))

        conv1 = fluid.layers.concat(ys, axis=1)
        conv2 = self._conv_norm(
            input=conv1,
            num_filters=num_filters2,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self._shortcut(
            input, num_filters2, stride, is_first, name=name + "_branch1")

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

    def layer_warp(self, input, stage_num):
        """
        Args:
            input (Variable): input variable.
            stage_num (int): the stage number, should be 2, 3, 4, 5

        Returns:
            The last variable in endpoint-th stage.
        """
        assert stage_num in [2, 3, 4, 5]

        stages, block_func = self.depth_cfg[self.depth]
        count = stages[stage_num - 2]

        ch_out = self.stage_filters[stage_num - 2]
        is_first = False if stage_num != 2 else True
        dcn_v2 = True if stage_num in self.dcn_v2_stages else False

        num_filters1 = self.num_filters1[stage_num - 2]
        num_filters2 = self.num_filters2[stage_num - 2]

        nonlocal_mod = 1000
        if stage_num in self.nonlocal_stages:
            nonlocal_mod = self.nonlocal_mod_cfg[
                self.depth] if stage_num == 4 else 2

        # Make the layer name and parameter name consistent
        # with ImageNet pre-trained model
        conv = input
        for i in range(count):
            conv_name = self.na.fix_layer_warp_name(stage_num, count, i)
            if self.depth < 50:
                is_first = True if i == 0 and stage_num == 2 else False
            conv = block_func(
                input=conv,
                num_filters1=num_filters1,
                num_filters2=num_filters2,
                stride=2 if i == 0 and stage_num != 2 else 1,
                is_first=is_first,
                name=conv_name,
                dcn_v2=dcn_v2)

            # add non local model
            dim_in = conv.shape[1]
            nonlocal_name = "nonlocal_conv{}".format(stage_num)
            if i % nonlocal_mod == nonlocal_mod - 1:
                conv = add_space_nonlocal(conv, dim_in, dim_in,
                                          nonlocal_name + '_{}'.format(i),
                                          int(dim_in / 2))
        return conv


@register
@serializable
class Res2NetC5(Res2Net):
    __doc__ = Res2Net.__doc__

    def __init__(self,
                 depth=50,
                 width=26,
                 scales=4,
                 freeze_at=2,
                 norm_type='bn',
                 freeze_norm=True,
                 norm_decay=0.,
                 variant='b',
                 feature_maps=[5],
                 weight_prefix_name=''):
        super(Res2NetC5, self).__init__(depth, width, scales, freeze_at,
                                        norm_type, freeze_norm, norm_decay,
                                        variant, feature_maps)
        self.severed_head = True
