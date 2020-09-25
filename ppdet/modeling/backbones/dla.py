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
import paddle.fluid.layers as L

from ppdet.core.workspace import register
from ppdet.modeling.ops import Conv2dUnit


def get_norm(norm_type):
    assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
    bn = 0
    gn = 0
    af = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    return bn, gn, af


class BasicBlock(object):
    def __init__(self, norm_type, inplanes, planes, stride=1, name=''):
        super(BasicBlock, self).__init__()
        bn, gn, af = get_norm(norm_type)
        self.conv1 = Conv2dUnit(inplanes, planes, 3, stride=stride, bias_attr=False, bn=bn, gn=gn, af=af, act='relu', name=name+'.conv1')
        self.conv2 = Conv2dUnit(planes, planes, 3, stride=1, bias_attr=False, bn=bn, gn=gn, af=af, act=None, name=name+'.conv2')
        self.stride = stride

    def __call__(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = L.elementwise_add(x=out, y=residual, act=None)
        out = L.relu(out)
        return out



class Root(object):
    def __init__(self, norm_type, in_channels, out_channels, kernel_size, residual, name=''):
        super(Root, self).__init__()
        bn, gn, af = get_norm(norm_type)
        self.conv = Conv2dUnit(in_channels, out_channels, kernel_size, stride=1, bias_attr=False, bn=bn, gn=gn, af=af, act=None, name=name+'.conv')
        self.residual = residual

    def __call__(self, *x):
        children = x
        x = L.concat(list(x), axis=1)
        x = self.conv(x)
        if self.residual:
            x += children[0]
        x = L.relu(x)
        return x


class Tree(object):
    def __init__(self, norm_type, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1, root_residual=False, name=''):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(norm_type, in_channels, out_channels, stride, name=name+'.tree1')
            self.tree2 = block(norm_type, out_channels, out_channels, 1, name=name+'.tree2')
        else:
            self.tree1 = Tree(norm_type, levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size, root_residual=root_residual, name=name+'.tree1')
            self.tree2 = Tree(norm_type, levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, root_residual=root_residual, name=name+'.tree2')
        if levels == 1:
            self.root = Root(norm_type, root_dim, out_channels, root_kernel_size, root_residual, name=name+'.root')
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = False
        self.stride = stride
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = True
        if in_channels != out_channels:
            bn, gn, af = get_norm(norm_type)
            self.project = Conv2dUnit(in_channels, out_channels, 1, stride=1, bias_attr=False, bn=bn, gn=gn, af=af, act=None, name=name+'.project')

    def __call__(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = L.pool2d(input=x, pool_size=self.stride, pool_stride=self.stride, pool_type='max') if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


__all__ = ['DLA']


@register
class DLA(object):
    __shared__ = ['norm_type', 'levels', 'channels']

    def __init__(self,
                 norm_type="sync_bn",
                 levels=[1, 1, 1, 2, 2, 1],
                 channels=[16, 32, 64, 128, 256, 512],
                 block=BasicBlock,
                 residual_root=False,
                 feature_maps=[3, 4, 5]):
        self.norm_type = norm_type
        self.channels = channels
        self.feature_maps = feature_maps

        self._out_features = ["level{}".format(i) for i in range(6)]   # 每个特征图的名字
        self._out_feature_channels = {k: channels[i] for i, k in enumerate(self._out_features)}   # 每个特征图的输出通道数
        self._out_feature_strides = {k: 2 ** i for i, k in enumerate(self._out_features)}   # 每个特征图的下采样倍率

        bn, gn, af = get_norm(norm_type)
        self.base_layer = Conv2dUnit(3, channels[0], 7, stride=1, bias_attr=False, bn=bn, gn=gn, af=af, act='relu', name='dla.base_layer')
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], name='dla.level0')
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2, name='dla.level1')
        self.level2 = Tree(norm_type, levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root, name='dla.level2')
        self.level3 = Tree(norm_type, levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root, name='dla.level3')
        self.level4 = Tree(norm_type, levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root, name='dla.level4')
        self.level5 = Tree(norm_type, levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root, name='dla.level5')

    def _make_conv_level(self, inplanes, planes, convs, stride=1, name=''):
        modules = []
        for i in range(convs):
            bn, gn, af = get_norm(self.norm_type)
            modules.append(Conv2dUnit(inplanes, planes, 3, stride=stride if i == 0 else 1, bias_attr=False, bn=bn, gn=gn, af=af, act='relu', name=name+'.conv%d'%i))
            inplanes = planes
        return modules

    def __call__(self, x):
        outs = []
        x = self.base_layer(x)
        for i in range(6):
            name = 'level{}'.format(i)
            level = getattr(self, name)
            if isinstance(level, list):
                for ly in level:
                    x = ly(x)
            else:
                x = level(x)
            if i in self.feature_maps:
                outs.append(x)
        return OrderedDict([('res{}_sum'.format(self.feature_maps[idx]), feat)
                            for idx, feat in enumerate(outs)])