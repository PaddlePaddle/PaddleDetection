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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Uniform

from ppdet.core.workspace import register

__all__ = ['Hourglass']


def kaiming_init(input, filter_size):
    fan_in = input.shape[1]
    std = (1.0 / (fan_in * filter_size * filter_size))**0.5
    return Uniform(0. - std, std)


def _conv_norm(x,
               k,
               out_dim,
               stride=1,
               pad=0,
               groups=None,
               with_bn=True,
               bn_act=None,
               ind=None,
               name=None):
    conv_name = "_conv" if ind is None else "_conv" + str(ind)
    bn_name = "_bn" if ind is None else "_bn" + str(ind)

    conv = fluid.layers.conv2d(
        input=x,
        filter_size=k,
        num_filters=out_dim,
        stride=stride,
        padding=pad,
        groups=groups,
        param_attr=ParamAttr(
            name=name + conv_name + "_weight", initializer=kaiming_init(x, k)),
        bias_attr=ParamAttr(
            name=name + conv_name + "_bias", initializer=kaiming_init(x, k))
        if not with_bn else False,
        name=name + '_output')
    if with_bn:
        pattr = ParamAttr(name=name + bn_name + '_weight')
        battr = ParamAttr(name=name + bn_name + '_bias')
        out = fluid.layers.batch_norm(
            input=conv,
            act=bn_act,
            name=name + '_bn_output',
            param_attr=pattr,
            bias_attr=battr,
            moving_mean_name=name + bn_name + '_running_mean',
            moving_variance_name=name + bn_name +
            '_running_var') if with_bn else conv
    else:
        out = fluid.layers.relu(conv)
    return out


def residual_block(x, out_dim, k=3, stride=1, name=None):
    p = (k - 1) // 2
    conv1 = _conv_norm(
        x, k, out_dim, pad=p, stride=stride, bn_act='relu', ind=1, name=name)
    conv2 = _conv_norm(conv1, k, out_dim, pad=p, ind=2, name=name)

    skip = _conv_norm(
        x, 1, out_dim, stride=stride,
        name=name + '_skip') if stride != 1 or x.shape[1] != out_dim else x
    return fluid.layers.elementwise_add(
        x=skip, y=conv2, act='relu', name=name + "_add")


def fire_block(x, out_dim, sr=2, stride=1, name=None):
    conv1 = _conv_norm(x, 1, out_dim // sr, ind=1, name=name)
    conv_1x1 = fluid.layers.conv2d(
        conv1,
        filter_size=1,
        num_filters=out_dim // 2,
        stride=stride,
        param_attr=ParamAttr(
            name=name + "_conv_1x1_weight", initializer=kaiming_init(conv1, 1)),
        bias_attr=False,
        name=name + '_conv_1x1')
    conv_3x3 = fluid.layers.conv2d(
        conv1,
        filter_size=3,
        num_filters=out_dim // 2,
        stride=stride,
        padding=1,
        groups=out_dim // sr,
        param_attr=ParamAttr(
            name=name + "_conv_3x3_weight", initializer=kaiming_init(conv1, 3)),
        bias_attr=False,
        name=name + '_conv_3x3',
        use_cudnn=False)
    conv2 = fluid.layers.concat(
        [conv_1x1, conv_3x3], axis=1, name=name + '_conv2')
    pattr = ParamAttr(name=name + '_bn2_weight')
    battr = ParamAttr(name=name + '_bn2_bias')

    bn2 = fluid.layers.batch_norm(
        input=conv2,
        name=name + '_bn2',
        param_attr=pattr,
        bias_attr=battr,
        moving_mean_name=name + '_bn2_running_mean',
        moving_variance_name=name + '_bn2_running_var')

    if stride == 1 and x.shape[1] == out_dim:
        return fluid.layers.elementwise_add(
            x=bn2, y=x, act='relu', name=name + "_add_relu")
    else:
        return fluid.layers.relu(bn2, name="_relu")


def make_layer(x, in_dim, out_dim, modules, block, name=None):
    layers = block(x, out_dim, name=name + '_0')
    for i in range(1, modules):
        layers = block(layers, out_dim, name=name + '_' + str(i))
    return layers


def make_hg_layer(x, in_dim, out_dim, modules, block, name=None):
    layers = block(x, out_dim, stride=2, name=name + '_0')
    for i in range(1, modules):
        layers = block(layers, out_dim, name=name + '_' + str(i))
    return layers


def make_layer_revr(x, in_dim, out_dim, modules, block, name=None):
    for i in range(modules - 1):
        x = block(x, in_dim, name=name + '_' + str(i))
    layers = block(x, out_dim, name=name + '_' + str(modules - 1))
    return layers


def make_unpool_layer(x, dim, name=None):
    pattr = ParamAttr(name=name + '_weight', initializer=kaiming_init(x, 4))
    battr = ParamAttr(name=name + '_bias', initializer=kaiming_init(x, 4))
    layer = fluid.layers.conv2d_transpose(
        input=x,
        num_filters=dim,
        filter_size=4,
        stride=2,
        padding=1,
        param_attr=pattr,
        bias_attr=battr)
    return layer


@register
class Hourglass(object):
    """
    Hourglass Network, see https://arxiv.org/abs/1603.06937
    Args:
        stack (int): stack of hourglass, 2 by default
        dims (list): dims of each level in hg_module
        modules (list): num of modules in each level
    """
    __shared__ = ['stack']

    def __init__(self,
                 stack=2,
                 dims=[256, 256, 384, 384, 512],
                 modules=[2, 2, 2, 2, 4],
                 block_name='fire'):
        super(Hourglass, self).__init__()
        self.stack = stack
        assert len(dims) == len(modules), \
            "Expected len of dims equal to len of modules, Receiced len of "\
            "dims: {}, len of modules: {}".format(len(dims), len(modules))
        self.dims = dims
        self.modules = modules
        self.num_level = len(dims) - 1
        block_dict = {'fire': fire_block}
        self.block = block_dict[block_name]

    def __call__(self, input, name='hg'):
        inter = self.pre(input, name + '_pre')
        cnvs = []
        for ind in range(self.stack):
            hg = self.hg_module(
                inter,
                self.num_level,
                self.dims,
                self.modules,
                name=name + '_hgs_' + str(ind))
            cnv = _conv_norm(
                hg,
                3,
                256,
                bn_act='relu',
                pad=1,
                name=name + '_cnvs_' + str(ind))
            cnvs.append(cnv)

            if ind < self.stack - 1:
                inter = _conv_norm(
                    inter, 1, 256, name=name + '_inters__' +
                    str(ind)) + _conv_norm(
                        cnv, 1, 256, name=name + '_cnvs__' + str(ind))
                inter = fluid.layers.relu(inter)
                inter = residual_block(
                    inter, 256, name=name + '_inters_' + str(ind))
        return cnvs

    def pre(self, x, name=None):
        conv = _conv_norm(
            x, 7, 128, stride=2, pad=3, bn_act='relu', name=name + '_0')
        res1 = residual_block(conv, 256, stride=2, name=name + '_1')
        res2 = residual_block(res1, 256, stride=2, name=name + '_2')
        return res2

    def hg_module(self,
                  x,
                  n=4,
                  dims=[256, 256, 384, 384, 512],
                  modules=[2, 2, 2, 2, 4],
                  make_up_layer=make_layer,
                  make_hg_layer=make_hg_layer,
                  make_low_layer=make_layer,
                  make_hg_layer_revr=make_layer_revr,
                  make_unpool_layer=make_unpool_layer,
                  name=None):
        curr_mod = modules[0]
        next_mod = modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]
        up1 = make_up_layer(
            x, curr_dim, curr_dim, curr_mod, self.block, name=name + '_up1')
        max1 = x
        low1 = make_hg_layer(
            max1, curr_dim, next_dim, curr_mod, self.block, name=name + '_low1')
        low2 = self.hg_module(
            low1,
            n - 1,
            dims[1:],
            modules[1:],
            make_up_layer=make_up_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            name=name + '_low2') if n > 1 else make_low_layer(
                low1,
                next_dim,
                next_dim,
                next_mod,
                self.block,
                name=name + '_low2')
        low3 = make_hg_layer_revr(
            low2, next_dim, curr_dim, curr_mod, self.block, name=name + '_low3')
        up2 = make_unpool_layer(low3, curr_dim, name=name + '_up2')
        merg = fluid.layers.elementwise_add(x=up1, y=up2, name=name + '_merg')
        return merg
