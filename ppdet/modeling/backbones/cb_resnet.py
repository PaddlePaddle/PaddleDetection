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

from .name_adapter import NameAdapter
from .nonlocal_helper import add_space_nonlocal

__all__ = ['CBResNet']


@register
@serializable
class CBResNet(object):
    """
    CBNet, see https://arxiv.org/abs/1909.03625
    Args:
        depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
        freeze_at (int): freeze the backbone at which stage
        norm_type (str): normalization type, 'bn'/'sync_bn'/'affine_channel'
        freeze_norm (bool): freeze normalization layers
        norm_decay (float): weight decay for normalization layer weights
        variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
        feature_maps (list): index of stages whose feature maps are returned
        dcn_v2_stages (list): index of stages who select deformable conv v2
        nonlocal_stages (list): index of stages who select nonlocal networks
        repeat_num (int): number of repeat for backbone
    Attention:
        1. Here we set the ResNet as the base backbone.
        2. All the pretraned params are copied from corresponding names,
           but with different names to avoid name refliction.
    """

    def __init__(self,
                 depth=50,
                 freeze_at=2,
                 norm_type='bn',
                 freeze_norm=True,
                 norm_decay=0.,
                 variant='b',
                 feature_maps=[2, 3, 4, 5],
                 dcn_v2_stages=[],
                 nonlocal_stages = [],
                 repeat_num = 2):
        super(CBResNet, self).__init__()

        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]

        assert depth in [18, 34, 50, 101, 152, 200], \
            "depth {} not in [18, 34, 50, 101, 152, 200]"
        assert variant in ['a', 'b', 'c', 'd'], "invalid ResNet variant"
        assert 0 <= freeze_at <= 4, "freeze_at should be 0, 1, 2, 3 or 4"
        assert len(feature_maps) > 0, "need one or more feature maps"
        assert norm_type in ['bn', 'sync_bn', 'affine_channel']
        assert not (len(nonlocal_stages)>0 and depth<50), \
                    "non-local is not supported for resnet18 or resnet34"

        self.depth = depth
        self.dcn_v2_stages = dcn_v2_stages
        self.freeze_at = freeze_at
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.variant = variant
        self._model_type = 'ResNet'
        self.feature_maps = feature_maps
        self.repeat_num = repeat_num
        self.curr_level = 0
        self.depth_cfg = {
            18: ([2, 2, 2, 2], self.basicblock),
            34: ([3, 4, 6, 3], self.basicblock),
            50: ([3, 4, 6, 3], self.bottleneck),
            101: ([3, 4, 23, 3], self.bottleneck),
            152: ([3, 8, 36, 3], self.bottleneck),
            200: ([3, 12, 48, 3], self.bottleneck),
        }

        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_mod_cfg = {
            50  : 2,
            101 : 5,
            152 : 8,
            200 : 12,
        }

        self.stage_filters = [64, 128, 256, 512]
        self._c1_out_chan_num = 64
        self.na = NameAdapter(self)

    def _conv_offset(self, input, filter_size, stride, padding, act=None, name=None):
        out_channel = filter_size * filter_size * 3
        out = fluid.layers.conv2d(input,
            num_filters=out_channel,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            param_attr=ParamAttr(
                initializer=Constant(0.0), name=name + ".w_0"),
            bias_attr=ParamAttr(
                initializer=Constant(0.0), name=name + ".b_0"),
            act=act,
            name=name)
        return out

    def _conv_norm(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None,
                   name=None,
                   dcn=False):
        if not dcn:
            conv = fluid.layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                act=None,
                param_attr=ParamAttr(name=name + "_weights_"+str(self.curr_level)),
                bias_attr=False)
        else:
            offset_mask = self._conv_offset(
                input=input,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                act=None,
                name=name + "_conv_offset_" + str(self.curr_level))
            offset_channel = filter_size ** 2 * 2
            mask_channel = filter_size ** 2
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
                param_attr=ParamAttr(name=name + "_weights_"+str(self.curr_level)),
                bias_attr=False)

        bn_name = self.na.fix_conv_norm_name(name)

        norm_lr = 0. if self.freeze_norm else 1.
        norm_decay = self.norm_decay
        pattr = ParamAttr(
            name=bn_name + '_scale_'+str(self.curr_level),
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        battr = ParamAttr(
            name=bn_name + '_offset_'+str(self.curr_level),
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        if self.norm_type in ['bn', 'sync_bn']:
            global_stats = True if self.freeze_norm else False
            out = fluid.layers.batch_norm(
                input=conv,
                act=act,
                name=bn_name + '.output.1_'+str(self.curr_level),
                param_attr=pattr,
                bias_attr=battr,
                moving_mean_name=bn_name + '_mean_'+str(self.curr_level),
                moving_variance_name=bn_name + '_variance_'+str(self.curr_level),
                use_global_stats=global_stats)
            scale = fluid.framework._get_var(pattr.name)
            bias = fluid.framework._get_var(battr.name)
        elif self.norm_type == 'affine_channel':
            assert False, "deprecated!!!"
        if self.freeze_norm:
            scale.stop_gradient = True
            bias.stop_gradient = True
        return out

    def _shortcut(self, input, ch_out, stride, is_first, name):
        max_pooling_in_short_cut = self.variant == 'd'
        ch_in = input.shape[1]
        # the naming rule is same as pretrained weight
        name = self.na.fix_shortcut_name(name)
        if ch_in != ch_out or stride != 1 or (self.depth < 50 and is_first):
            if max_pooling_in_short_cut and not is_first:
                input = fluid.layers.pool2d(
                    input=input,
                    pool_size=2,
                    pool_stride=2,
                    pool_padding=0,
                    ceil_mode=True,
                    pool_type='avg')
                return self._conv_norm(input, ch_out, 1, 1, name=name)
            return self._conv_norm(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck(self, input, num_filters, stride, is_first, name, dcn=False):
        if self.variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        # ResNeXt
        groups = getattr(self, 'groups', 1)
        group_width = getattr(self, 'group_width', -1)
        if groups == 1:
            expand = 4
        elif (groups * group_width) == 256:
            expand = 1
        else:  # FIXME hard code for now, handles 32x4d, 64x4d and 32x8d
            num_filters = num_filters // 2
            expand = 2

        conv_name1, conv_name2, conv_name3, \
            shortcut_name = self.na.fix_bottleneck_name(name)

        conv_def = [[num_filters, 1, stride1, 'relu', 1, conv_name1],
                    [num_filters, 3, stride2, 'relu', groups, conv_name2],
                    [num_filters * expand, 1, 1, None, 1, conv_name3]]

        residual = input
        for i, (c, k, s, act, g, _name) in enumerate(conv_def):
            residual = self._conv_norm(
                input=residual,
                num_filters=c,
                filter_size=k,
                stride=s,
                act=act,
                groups=g,
                name=_name,
                dcn=(i==1 and dcn))
        short = self._shortcut(
            input,
            num_filters * expand,
            stride,
            is_first=is_first,
            name=shortcut_name)
        # Squeeze-and-Excitation
        if callable(getattr(self, '_squeeze_excitation', None)):
            residual = self._squeeze_excitation(
                input=residual, num_channels=num_filters, name='fc' + name)
        return fluid.layers.elementwise_add(
            x=short, y=residual, act='relu')

    def basicblock(self, input, num_filters, stride, is_first, name, dcn=False):
        assert dcn is False, "Not implemented yet."
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        conv1 = self._conv_norm(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")
        short = self._shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')

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
        dcn = True if stage_num in self.dcn_v2_stages else False


        nonlocal_mod = 1000
        if stage_num in self.nonlocal_stages:
            nonlocal_mod = self.nonlocal_mod_cfg[self.depth] if stage_num==4 else 2

        # Make the layer name and parameter name consistent
        # with ImageNet pre-trained model
        conv = input
        for i in range(count):
            conv_name = self.na.fix_layer_warp_name(stage_num, count, i)
            if self.depth < 50:
                is_first = True if i == 0 and stage_num == 2 else False
            conv = block_func(
                input=conv,
                num_filters=ch_out,
                stride=2 if i == 0 and stage_num != 2 else 1,
                is_first=is_first,
                name=conv_name,
                dcn=dcn)

            # add non local model
            dim_in = conv.shape[1]
            nonlocal_name = "nonlocal_conv{}_lvl{}".format( stage_num, self.curr_level )
            if i % nonlocal_mod == nonlocal_mod - 1:
                conv = add_space_nonlocal(
                    conv, dim_in, dim_in,
                    nonlocal_name + '_{}'.format(i), int(dim_in / 2) )

        return conv

    def c1_stage(self, input):
        out_chan = self._c1_out_chan_num

        conv1_name = self.na.fix_c1_stage_name()

        if self.variant in ['c', 'd']:
            conv1_1_name= "conv1_1"
            conv1_2_name= "conv1_2"
            conv1_3_name= "conv1_3"
            conv_def = [
                [out_chan // 2, 3, 2, conv1_1_name],
                [out_chan // 2, 3, 1, conv1_2_name],
                [out_chan, 3, 1, conv1_3_name],
            ]
        else:
            conv_def = [[out_chan, 7, 2, conv1_name]]

        for (c, k, s, _name) in conv_def:
            input = self._conv_norm(
                input=input,
                num_filters=c,
                filter_size=k,
                stride=s,
                act='relu',
                name=_name)

        output = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        return output

    def connect( self, left, right, name ):
        ch_right = right.shape[1]
        conv = self._conv_norm( left,
                   num_filters=ch_right,
                   filter_size=1,
                   stride=1,
                   act="relu",
                   name=name+"_connect")
        shape = fluid.layers.shape(right)
        shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
        out_shape_ = shape_hw
        out_shape = fluid.layers.cast(out_shape_, dtype='int32')
        out_shape.stop_gradient = True
        conv = fluid.layers.resize_nearest(
            conv, scale=2., actual_shape=out_shape)

        output = fluid.layers.elementwise_add(x=right, y=conv)
        return output

    def __call__(self, input):
        assert isinstance(input, Variable)
        assert not (set(self.feature_maps) - set([2, 3, 4, 5])), \
            "feature maps {} not in [2, 3, 4, 5]".format(self.feature_maps)

        res_endpoints = []

        self.curr_level = 0
        res = self.c1_stage(input)
        feature_maps = range(2, max(self.feature_maps) + 1)
        for i in feature_maps:
            res = self.layer_warp(res, i)
            if i in self.feature_maps:
                res_endpoints.append(res)

        for num in range(1, self.repeat_num):
            self.curr_level = num
            res = self.c1_stage(input)
            for i in range( len(res_endpoints) ):
                res = self.connect( res_endpoints[i], res, "test_c"+str(i+1) )
                res = self.layer_warp(res, i+2)
                res_endpoints[i] = res
                if self.freeze_at >= i+2:
                    res.stop_gradient = True

        return OrderedDict([('res{}_sum'.format(self.feature_maps[idx]), feat)
                            for idx, feat in enumerate(res_endpoints)])
