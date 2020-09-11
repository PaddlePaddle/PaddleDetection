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
            bias_attr=False,
            use_cudnn=False)
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
        """
        # Node id starts from the input features and monotonically increase whenever

        # [Node NO.] Here is an example for level P3 - P7:
        # {3: [0,    8],
        #  4: [1, 7, 9],
        #  5: [2, 6, 10],
        #  6: [3, 5, 11],
        #  7: [4,    12]}

        #  [Related Edge]
        #   {'feat_level': 6, 'inputs_offsets': [3, 4]},      # for P6'
        #   {'feat_level': 5, 'inputs_offsets': [2, 5]},      # for P5'
        #   {'feat_level': 4, 'inputs_offsets': [1, 6]},      # for P4'
        #   {'feat_level': 3, 'inputs_offsets': [0, 7]},      # for P3"
        #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},   # for P4"
        #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},   # for P5"
        #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
        #   {'feat_level': 7, 'inputs_offsets': [4, 11]},     # for P7"

        P7 (4) --------------> P7" (12)
          |----------|           ↑
                     ↓           |
        P6 (3) --> P6' (5) --> P6" (11)
          |----------|----------↑↑
                     ↓           |
        P5 (2) --> P5' (6) --> P5" (10)
          |----------|----------↑↑
                     ↓           |
        P4 (1) --> P4' (7) --> P4" (9)
          |----------|----------↑↑
                     |----------↓|
        P3 (0) --------------> P3" (8)
        """

        super(BiFPNCell, self).__init__()
        self.levels = levels
        self.num_chan = num_chan
        self.inputs_layer_num = inputs_layer_num
        # Learnable weights of [P4", P5", P6"]
        self.trigates = fluid.layers.create_parameter(
            shape=[levels - 2, 3],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(1.))
        # Learnable weights of [P6', P5', P4', P3", P7"]
        self.bigates = fluid.layers.create_parameter(
            shape=[levels, 2],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(1.))
        self.eps = 1e-4

    def __call__(self, inputs, cell_name='', is_first_time=False, p4_2_p5_2=[]):
        assert len(inputs) == self.levels
        assert ((is_first_time) and (len(p4_2_p5_2) != 0)) or ((not is_first_time) and (len(p4_2_p5_2) == 0))

        # upsample operator
        def upsample(feat):
            return fluid.layers.resize_nearest(feat, scale=2.)

        # downsample operator
        def downsample(feat):
            return fluid.layers.pool2d(feat, pool_type='max', pool_size=3, pool_stride=2, pool_padding='SAME')

        # 3x3 fuse conv after OP combine
        fuse_conv = FusionConv(self.num_chan)

        # Normalize weight
        trigates = fluid.layers.relu(self.trigates)
        bigates = fluid.layers.relu(self.bigates)
        trigates /= fluid.layers.reduce_sum(trigates, dim=1, keep_dim=True) + self.eps
        bigates /= fluid.layers.reduce_sum(bigates, dim=1, keep_dim=True) + self.eps

        feature_maps = list(inputs)  # make a copy, 依次是 [P3, P4, P5, P6, P7]
        # top down path
        for l in range(self.levels - 1):
            p = self.levels - l - 2
            w1 = fluid.layers.slice(bigates, axes=[0, 1], starts=[l, 0], ends=[l + 1, 1])
            w2 = fluid.layers.slice(bigates, axes=[0, 1], starts=[l, 1], ends=[l + 1, 2])
            above_layer = upsample(feature_maps[p + 1])
            feature_maps[p] = fuse_conv(w1 * above_layer + w2 * inputs[p], name='{}_tb_{}'.format(cell_name, l))
        # bottom up path
        for l in range(1, self.levels):
            p = l
            name = '{}_bt_{}'.format(cell_name, l)
            below = downsample(feature_maps[p - 1])
            if p == self.levels - 1:
                # handle P7
                w1 = fluid.layers.slice(bigates, axes=[0, 1], starts=[p, 0], ends=[p + 1, 1])
                w2 = fluid.layers.slice(bigates, axes=[0, 1], starts=[p, 1], ends=[p + 1, 2])
                feature_maps[p] = fuse_conv(w1 * below + w2 * inputs[p], name=name)
            else:
                if is_first_time:
                    if p < self.inputs_layer_num:
                        w1 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 0], ends=[p, 1])
                        w2 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 1], ends=[p, 2])
                        w3 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 2], ends=[p, 3])
                        feature_maps[p] = fuse_conv(w1 * feature_maps[p] + w2 * below + w3 * p4_2_p5_2[p - 1], name=name)
                    else:  # For P6"
                        w1 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 0], ends=[p, 1])
                        w2 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 1], ends=[p, 2])
                        w3 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 2], ends=[p, 3])
                        feature_maps[p] = fuse_conv(w1 * feature_maps[p] + w2 * below + w3 * inputs[p], name=name)
                else:
                    w1 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 0], ends=[p, 1])
                    w2 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 1], ends=[p, 2])
                    w3 = fluid.layers.slice(trigates, axes=[0, 1], starts=[p - 1, 2], ends=[p, 3])
                    feature_maps[p] = fuse_conv(w1 * feature_maps[p] + w2 * below + w3 * inputs[p], name=name)
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
        # Squeeze the channel with 1x1 conv
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
                    param_attr=ParamAttr(initializer=Constant(1.0), regularizer=L2Decay(0.)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                    name='resample_bn_{}'.format(idx))
            else:
                feat = inputs[idx]
            feats.append(feat)
        # Build additional input features that are not from backbone.
        # P_7 layer we just use pool2d without conv layer & bn, for the same channel with P_6.
        # https://github.com/google/automl/blob/master/efficientdet/keras/efficientdet_keras.py#L820
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
                    param_attr=ParamAttr(initializer=Constant(1.0), regularizer=L2Decay(0.)),
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
        p4_2_p5_2 = []
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
                param_attr=ParamAttr(initializer=Constant(1.0), regularizer=L2Decay(0.)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.)),
                name='resample2_bn_{}'.format(idx))
            p4_2_p5_2.append(feat)

        # BiFPN, repeated
        biFPN = BiFPNCell(self.num_chan, self.levels, len(inputs))
        for r in range(self.repeat):
            if r == 0:
                feats = biFPN(feats, cell_name='bifpn_{}'.format(r), is_first_time=True, p4_2_p5_2=p4_2_p5_2)
            else:
                feats = biFPN(feats, cell_name='bifpn_{}'.format(r))

        return feats
