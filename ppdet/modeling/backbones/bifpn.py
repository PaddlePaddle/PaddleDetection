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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Constant, Xavier

from ppdet.core.workspace import register

__all__ = ['BiFPN']


class FusionConv(object):
    def __init__(self, num_chan):
        super(FusionConv, self).__init__()
        self.num_chan = num_chan

    def __call__(self, inputs, name=''):
        x = fluid.layers.swish(inputs)
        # depthwise
        x = fluid.layers.conv2d(
            x,
            self.num_chan,
            filter_size=3,
            padding='SAME',
            groups=self.num_chan,
            param_attr=ParamAttr(
                initializer=Xavier(), name=name + '_dw_w'),
            bias_attr=False)
        # pointwise
        x = fluid.layers.conv2d(
            x,
            self.num_chan,
            filter_size=1,
            param_attr=ParamAttr(
                initializer=Xavier(), name=name + '_pw_w'),
            bias_attr=ParamAttr(
                regularizer=L2Decay(0.), name=name + '_pw_b'))
        # bn + act
        x = fluid.layers.batch_norm(
            x,
            momentum=0.997,
            epsilon=1e-04,
            param_attr=ParamAttr(
                initializer=Constant(1.0),
                regularizer=L2Decay(0.),
                name=name + '_bn_w'),
            bias_attr=ParamAttr(
                regularizer=L2Decay(0.), name=name + '_bn_b'))
        return x


class BiFPNCell(object):
    def __init__(self, num_chan, levels=5, inputs_layer_num=3):
        super(BiFPNCell, self).__init__()
        self.levels = levels
        self.num_chan = num_chan
        num_trigates = levels - 2
        num_bigates = levels
        self.inputs_layer_num = inputs_layer_num
        self.trigates = fluid.layers.create_parameter(
            shape=[num_trigates, 3],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(1.))
        self.bigates = fluid.layers.create_parameter(
            shape=[num_bigates, 2],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(1.))
        self.eps = 1e-4

    def __call__(self, inputs, cell_name=''):

        def upsample(feat):
            return fluid.layers.resize_nearest(feat, scale=2.)

        def downsample(feat):
            return fluid.layers.pool2d(
                feat,
                pool_type='max',
                pool_size=3,
                pool_stride=2,
                pool_padding='SAME')

        fuse_conv = FusionConv(self.num_chan)

        # normalize weight
        trigates = fluid.layers.relu(self.trigates)
        bigates = fluid.layers.relu(self.bigates)
        trigates /= fluid.layers.reduce_sum(
            trigates, dim=1, keep_dim=True) + self.eps
        bigates /= fluid.layers.reduce_sum(
            bigates, dim=1, keep_dim=True) + self.eps

        # top down path
        feature_maps = list(inputs[:self.levels])  # make a copy
        for l in range(self.levels - 1):
            p = self.levels - l - 2
            w1 = fluid.layers.slice(
                bigates, axes=[0, 1], starts=[l, 0], ends=[l + 1, 1])
            w2 = fluid.layers.slice(
                bigates, axes=[0, 1], starts=[l, 1], ends=[l + 1, 2])
            above = upsample(feature_maps[p + 1])
            feature_maps[p] = fuse_conv(
                w1 * above + w2 * inputs[p],
                name='{}_tb_{}'.format(cell_name, l))
        # bottom up path
        for l in range(1, self.levels):
            p = l
            name = '{}_bt_{}'.format(cell_name, l)
            below = downsample(feature_maps[p - 1])
            if p == self.levels - 1:
                # handle P7
                w1 = fluid.layers.slice(
                    bigates, axes=[0, 1], starts=[p, 0], ends=[p + 1, 1])
                w2 = fluid.layers.slice(
                    bigates, axes=[0, 1], starts=[p, 1], ends=[p + 1, 2])
                feature_maps[p] = fuse_conv(
                    w1 * below + w2 * inputs[p], name=name)
            else:
                # For the first loop in BiFPN
                if len(inputs) != self.levels:
                    if p < self.inputs_layer_num:
                        w1 = fluid.layers.slice(
                            trigates,
                            axes=[0, 1],
                            starts=[p - 1, 0],
                            ends=[p, 1])
                        w2 = fluid.layers.slice(
                            trigates,
                            axes=[0, 1],
                            starts=[p - 1, 1],
                            ends=[p, 2])
                        w3 = fluid.layers.slice(
                            trigates,
                            axes=[0, 1],
                            starts=[p - 1, 2],
                            ends=[p, 3])
                        feature_maps[p] = fuse_conv(
                            w1 * feature_maps[p] + w2 * below + w3 *
                            inputs[p - 1 + self.levels],
                            name=name)
                    else:  # For P6"
                        w1 = fluid.layers.slice(
                            trigates,
                            axes=[0, 1],
                            starts=[p - 1, 0],
                            ends=[p, 1])
                        w2 = fluid.layers.slice(
                            trigates,
                            axes=[0, 1],
                            starts=[p - 1, 1],
                            ends=[p, 2])
                        w3 = fluid.layers.slice(
                            trigates,
                            axes=[0, 1],
                            starts=[p - 1, 2],
                            ends=[p, 3])
                        feature_maps[p] = fuse_conv(
                            w1 * feature_maps[p] + w2 * below + w3 * inputs[p],
                            name=name)
                else:
                    w1 = fluid.layers.slice(
                        trigates, axes=[0, 1], starts=[p - 1, 0], ends=[p, 1])
                    w2 = fluid.layers.slice(
                        trigates, axes=[0, 1], starts=[p - 1, 1], ends=[p, 2])
                    w3 = fluid.layers.slice(
                        trigates, axes=[0, 1], starts=[p - 1, 2], ends=[p, 3])
                    feature_maps[p] = fuse_conv(
                        w1 * feature_maps[p] + w2 * below + w3 * inputs[p],
                        name=name)
        return feature_maps


@register
class BiFPN(object):
    """
    Bidirectional Feature Pyramid Network, see https://arxiv.org/abs/1911.09070

    Args:
        num_chan (int): number of feature channels
        repeat (int): number of repeats of the BiFPN module
        level (int): number of FPN levels, default: 5
    """

    def __init__(self, num_chan, repeat=3, levels=5):
        super(BiFPN, self).__init__()
        self.num_chan = num_chan
        self.repeat = repeat
        self.levels = levels

    def __call__(self, inputs):
        feats = []
        # NOTE add two extra levels
        for idx in range(len(inputs)):
            if inputs[idx].shape[1] != self.num_chan:
                feat = fluid.layers.conv2d(
                    inputs[idx],
                    self.num_chan,
                    filter_size=1,
                    padding='SAME',
                    param_attr=ParamAttr(initializer=Xavier()),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                    name='resample_conv_{}'.format(idx))
                feat = fluid.layers.batch_norm(
                    feat,
                    momentum=0.997,
                    epsilon=1e-04,
                    param_attr=ParamAttr(
                        initializer=Constant(1.0), regularizer=L2Decay(0.)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                    name='resample_bn_{}'.format(idx))
            else:
                feat = inputs[idx]
            feats.append(feat)
        # Build additional input features that are not from backbone.
        # P_7 layer we just use pool2d without conv layer & bn, for the same channel with P_6.
        for idx in range(len(inputs), self.levels):
            if feats[-1].shape[1] != self.num_chan:
                feat = fluid.layers.conv2d(
                    feats[-1],
                    self.num_chan,
                    filter_size=1,
                    padding='SAME',
                    param_attr=ParamAttr(initializer=Xavier()),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                    name='resample_conv_{}'.format(idx))
                feat = fluid.layers.batch_norm(
                    feat,
                    momentum=0.997,
                    epsilon=1e-04,
                    param_attr=ParamAttr(
                        initializer=Constant(1.0), regularizer=L2Decay(0.)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                    name='resample_bn_{}'.format(idx))
            feat = fluid.layers.pool2d(
                feat,
                pool_type='max',
                pool_size=3,
                pool_stride=2,
                pool_padding='SAME',
                name='resample_downsample_{}'.format(idx))
            feats.append(feat)
        # Handle the p4_2 and p5_2 with another 1x1 conv & bn layer
        for idx in range(1, len(inputs)):
            feat = fluid.layers.conv2d(
                inputs[idx],
                self.num_chan,
                filter_size=1,
                padding='SAME',
                param_attr=ParamAttr(initializer=Xavier()),
                bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                name='resample2_conv_{}'.format(idx))
            feat = fluid.layers.batch_norm(
                feat,
                momentum=0.997,
                epsilon=1e-04,
                param_attr=ParamAttr(
                    initializer=Constant(1.0), regularizer=L2Decay(0.)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                name='resample2_bn_{}'.format(idx))
            feats.append(feat)

        biFPN = BiFPNCell(self.num_chan, self.levels, len(inputs))
        for r in range(self.repeat):
            feats = biFPN(feats, cell_name='bifpn_{}'.format(r))

        return feats
