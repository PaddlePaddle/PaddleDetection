# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from paddle.nn.initializer import Constant, KaimingUniform
from paddle.vision.ops import DeformConv2D
from ..shape_spec import ShapeSpec

DLA_cfg = {34: ([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512])}


class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 name=None):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = paddle.ParamAttr(
                initializer=Constant(0.),
                name='{}.conv_offset_mask.bias'.format(name))
        else:
            offset_bias_attr = paddle.ParamAttr(
                initializer=Constant(0.),
                learning_rate=lr_scale,
                regularizer=regularizer,
                name='{}.conv_offset_mask.bias'.format(name))
        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=paddle.ParamAttr(
                initializer=Constant(0.0),
                name='{}.conv_offset_mask.weight'.format(name)),
            bias_attr=offset_bias_attr)

        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 dcn_v2=False,
                 bias=False,
                 name=None):
        super(ConvLayer, self).__init__()
        bias_attr = False
        fan_in = ch_in * kernel_size**2
        bound = 1 / math.sqrt(fan_in)
        if not dcn_v2:
            param_attr = paddle.ParamAttr(
                initializer=KaimingUniform(), name=name + ".weight")
            if bias:
                bias_attr = paddle.ParamAttr(
                    initializer=nn.initializer.Uniform(-bound, bound),
                    name=name + ".bias")
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                weight_attr=param_attr,
                bias_attr=bias_attr)
        else:
            param_attr = paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-bound, bound),
                name=name + ".weight")
            if bias:
                bias_attr = paddle.ParamAttr(
                    initializer=Constant(0), name=name + ".bias")
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                weight_attr=param_attr,
                bias_attr=bias_attr,
                name=name)

    def forward(self, inputs):
        out = self.conv(inputs)

        return out


class NormLayer(nn.Layer):
    def __init__(self, ch_out, name=None):
        super(NormLayer, self).__init__()
        param_attr = paddle.ParamAttr(name=name + ".weight")
        bias_attr = paddle.ParamAttr(name=name + ".bias")

        self.norm = nn.BatchNorm(
            ch_out,
            param_attr=param_attr,
            bias_attr=bias_attr,
            moving_mean_name=name + '.running_mean',
            moving_variance_name=name + '.running_var')

    def forward(self, inputs):
        out = self.norm(inputs)

        return out


class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, stride=1, dilation=1, name=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(
            ch_in,
            ch_out,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=dilation,
            dilation=dilation,
            name=name + ".conv1")
        self.bn1 = NormLayer(ch_out, name=name + ".bn1")
        self.conv2 = ConvLayer(
            ch_out,
            ch_out,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=dilation,
            dilation=dilation,
            name=name + ".conv2")
        self.bn2 = NormLayer(ch_out, name + ".bn2")

    def forward(self, inputs, residual=None):
        if residual is None:
            residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = paddle.add(x=out, y=residual)
        out = F.relu(out)

        return out


class Root(nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size, residual, name=None):
        super(Root, self).__init__()
        self.conv = ConvLayer(
            ch_in,
            ch_out,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
            name=name + ".conv")
        self.bn = NormLayer(ch_out, name=name + ".bn")
        self.residual = residual

    def forward(self, inputs):
        children = inputs
        out = self.conv(paddle.concat(inputs, axis=1))
        out = self.bn(out)
        if self.residual:
            out = paddle.add(x=out, y=children[0])
        out = F.relu(out)

        return out


class Tree(nn.Layer):
    def __init__(self,
                 level,
                 block,
                 ch_in,
                 ch_out,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False,
                 name=None):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * ch_out
        if level_root:
            root_dim += ch_in
        if level == 1:
            self.tree1 = block(
                ch_in, ch_out, stride, dilation, name=name + ".tree1")
            self.tree2 = block(
                ch_out, ch_out, 1, dilation, name=name + ".tree2")
        else:
            self.tree1 = Tree(
                level - 1,
                block,
                ch_in,
                ch_out,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                name=name + ".tree1")
            self.tree2 = Tree(
                level - 1,
                block,
                ch_out,
                ch_out,
                1,
                root_dim=root_dim + ch_out,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                name=name + ".tree2")

        if level == 1:
            self.root = Root(
                root_dim,
                ch_out,
                root_kernel_size,
                root_residual,
                name=name + ".root")
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.level = level
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride, stride=stride)
        if ch_in != ch_out:
            self.project = nn.Sequential(
                ConvLayer(
                    ch_in,
                    ch_out,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    name=name + ".project.0"),
                NormLayer(
                    ch_out, name=name + ".project.1"))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@register
@serializable
class DLA(nn.Layer):
    def __init__(self, depth=34, residual_root=False, name='base'):
        super(DLA, self).__init__()
        levels, channels = DLA_cfg[depth]
        if depth == 34:
            block = BasicBlock
        self.channels = channels
        self.base_layer = nn.Sequential(
            ConvLayer(
                3,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
                name=name + ".base_layer.0"),
            NormLayer(
                channels[0], name=name + ".base_layer.1"),
            nn.ReLU())
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0], name=name + ".level0")
        self.level1 = self._make_conv_level(
            channels[0],
            channels[1],
            levels[1],
            stride=2,
            name=name + ".level1")
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
            name=name + ".level2")
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
            name=name + ".level3")
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
            name=name + ".level4")
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
            name=name + ".level5")

    def _make_conv_level(self,
                         ch_in,
                         ch_out,
                         conv_num,
                         stride=1,
                         dilation=1,
                         name=None):
        modules = []
        for i in range(conv_num):
            modules.extend([
                ConvLayer(
                    ch_in,
                    ch_out,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation,
                    name=name + ".0"), NormLayer(
                        ch_out, name=name + ".1"), nn.ReLU()
            ])
            ch_in = ch_out
        return nn.Sequential(*modules)

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.channels[i]) for i in range(6)]

    def forward(self, inputs):
        outs = []
        im = inputs['image']
        feats = self.base_layer(im)
        for i in range(6):
            feats = getattr(self, 'level{}'.format(i))(feats)
            outs.append(feats)

        return outs
