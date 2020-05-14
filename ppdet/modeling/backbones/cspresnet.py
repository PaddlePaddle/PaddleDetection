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
from .gc_block import add_gc_block
from .name_adapter import NameAdapter

__all__ = ['CSPResNet']


@register
@serializable
class CSPResNet(object):
    """
    CSPDarkNet, see https://arxiv.org/abs/1911.11929 
    Args:
        depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
        freeze_at (int): freeze the backbone at which stage
        norm_type (str): normalization type, 'bn'/'sync_bn'/'affine_channel'
        freeze_norm (bool): freeze normalization layers
        norm_decay (float): weight decay for normalization layer weights
        feature_maps (list): index of stages whose feature maps are returned
        weight_prefix_name (str): prefix name of the weights
    """
    __shared__ = ['norm_type', 'freeze_norm', 'weight_prefix_name']

    def __init__(self,
                 depth=50,
                 freeze_at=2,
                 norm_type='bn',
                 freeze_norm=True,
                 norm_decay=0.,
                 feature_maps=[2, 3, 4, 5],
                 weight_prefix_name=''):
        super(CSPResNet, self).__init__()

        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]

        assert depth in [50, 101], \
            "depth {} not in [50, 101]"
        assert 0 <= freeze_at <= 4, "freeze_at should be 0, 1, 2, 3 or 4"
        assert len(feature_maps) > 0, "need one or more feature maps"
        assert norm_type in ['bn', 'sync_bn', 'affine_channel']

        self.depth = depth
        self.freeze_at = freeze_at
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self._model_type = 'CSPResNet'
        self.feature_maps = feature_maps
        self.depth_cfg = {
            50: ([3, 3, 5, 2], self.bottleneck),
            101: ([3, 3, 22, 2], self.bottleneck),
        }
        self.stage_filters = [64, 128, 256, 512]

        self.prefix_name = weight_prefix_name
        self.end_points = []

    def net(self, input):
        depth = self.depth_cfg[self.depth][0]
        block_func = self.depth_cfg[self.depth][1]

        num_filters = self.stage_filters

        conv = self._conv_norm(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='leaky',
            name="conv1")
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=2,
            pool_stride=2,
            pool_padding=0,
            pool_type='max')

        for block in range(len(depth)):
            conv_name = "res" + str(block + 2) + chr(97)
            if block != 0:
                conv = self._conv_norm(
                    input=conv,
                    num_filters=num_filters[block],
                    filter_size=3,
                    stride=2,
                    act="leaky_relu",
                    name=conv_name + "_downsample")

            # split
            left = conv
            right = conv
            if block == 0:
                ch = num_filters[block]
            else:
                ch = num_filters[block] * 2
            right = self._conv_norm(
                input=right,
                num_filters=ch,
                filter_size=1,
                act="leaky_relu",
                name=conv_name + "_right_first_route")

            for i in range(depth[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)

                right = self.bottleneck(
                    input=right,
                    num_filters=num_filters[block],
                    stride=1,
                    name=conv_name)

            # route
            left = self._conv_norm(
                input=left,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                act="leaky_relu",
                name=conv_name + "_left_route")
            right = self._conv_norm(
                input=right,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                act="leaky_relu",
                name=conv_name + "_right_route")
            conv = fluid.layers.concat([left, right], axis=1)

            conv = self._conv_norm(
                input=conv,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                stride=1,
                act="leaky_relu",
                name=conv_name + "_merged_transition")

            self.end_points.append(conv)
        return

    def _conv_norm(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None,
                   name=None,
                   dcn_v2=False):
        _name = self.prefix_name + name if self.prefix_name != '' else name

        lr_mult = 1.0

        if not dcn_v2:
            conv = fluid.layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                act=None,
                param_attr=ParamAttr(
                    name=_name + "_weights", learning_rate=lr_mult),
                bias_attr=False,
                name=_name + '.conv2d.output.1')
        else:
            # select deformable conv"
            offset_mask = self._conv_offset(
                input=input,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                act=None,
                name=_name + "_conv_offset")
            offset_channel = filter_size**2 * 2
            mask_channel = filter_size**2
            offset, mask = fluid.layers.split(
                input=offset_mask,
                num_or_sections=[offset_channel, mask_channel],
                dim=1)
            mask = fluid.layers.sigmoid(mask)
            conv = fluid.layers.deformable_conv(
                input=input,
                offset=offset,
                mask=mask,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                deformable_groups=1,
                im2col_step=1,
                param_attr=ParamAttr(
                    name=_name + "_weights", learning_rate=lr_mult),
                bias_attr=False,
                name=_name + ".conv2d.output.1")

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        bn_name = self.prefix_name + bn_name if self.prefix_name != '' else bn_name

        norm_lr = 0. if self.freeze_norm else lr_mult
        norm_decay = self.norm_decay
        pattr = ParamAttr(
            name=bn_name + '_scale',
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        battr = ParamAttr(
            name=bn_name + '_offset',
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        if self.norm_type in ['bn', 'sync_bn']:
            global_stats = True if self.freeze_norm else False
            out = fluid.layers.batch_norm(
                input=conv,
                act=None,
                name=bn_name + '.output.1',
                param_attr=pattr,
                bias_attr=battr,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance',
                use_global_stats=global_stats)
            scale = fluid.framework._get_var(pattr.name)
            bias = fluid.framework._get_var(battr.name)
        elif self.norm_type == 'affine_channel':
            scale = fluid.layers.create_parameter(
                shape=[conv.shape[1]],
                dtype=conv.dtype,
                attr=pattr,
                default_initializer=fluid.initializer.Constant(1.))
            bias = fluid.layers.create_parameter(
                shape=[conv.shape[1]],
                dtype=conv.dtype,
                attr=battr,
                default_initializer=fluid.initializer.Constant(0.))
            out = fluid.layers.affine_channel(
                x=conv, scale=scale, bias=bias, act=None)
        if self.freeze_norm:
            scale.stop_gradient = True
            bias.stop_gradient = True

        if act == "relu":
            out = fluid.layers.relu(out)
        elif act == "leaky_relu":
            out = fluid.layers.leaky_relu(out)
        return out

    def shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first is True:
            return self._conv_norm(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck(self, input, num_filters, stride, name):
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act="leaky_relu",
            name=name + "_branch2a")
        conv1 = self._conv_norm(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="leaky_relu",
            name=name + "_branch2b")
        conv2 = self._conv_norm(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input,
            num_filters * 2,
            stride,
            is_first=False,
            name=name + "_branch1")

        ret = short + conv2
        ret = fluid.layers.leaky_relu(ret, alpha=0.1)
        return ret

    def __call__(self, input):
        assert isinstance(input, Variable)
        assert not (set(self.feature_maps) - set([2, 3, 4, 5])), \
            "feature maps {} not in [2, 3, 4, 5]".format(self.feature_maps)

        res_endpoints = []

        res = input
        feature_maps = self.feature_maps
        self.net(input)

        for i in feature_maps:
            res = self.end_points[i - 2]
            if i in self.feature_maps:
                res_endpoints.append(res)
            if self.freeze_at >= i:
                res.stop_gradient = True

        return OrderedDict(
            [('cspres{}_sum'.format(self.feature_maps[idx]), feat)
             for idx, feat in enumerate(res_endpoints)])
