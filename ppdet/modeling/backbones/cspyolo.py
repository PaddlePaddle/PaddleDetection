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

import math
import six
import numpy as np
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register

__all__ = ['CSPYolo']


def autopad(k, p):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


@register
class CSPYolo(object):

    __shared__ = ['depth_multiple', 'width_multiple']

    def __init__(self,
                 layers=None,
                 neck=None,
                 depth_multiple=0.33,
                 width_multiple=0.50,
                 act='none',
                 yolov5=True,
                 save=[17, 20, 23],
                 conv_decay=0.0,
                 norm_type='bn',
                 norm_decay=0.0,
                 weight_prefix_name=''):

        if layers is None:
            self.layers = [
                # [from, number, module, args, kwargs]
                [-1, 1, 'Focus', [64, 3]],  # 0-P1/2
                [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                [-1, 3, 'BottleneckCSP', [128]],
                [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                [-1, 9, 'BottleneckCSP', [256]],
                [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                [-1, 9, 'BottleneckCSP', [512]],
                [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
                [-1, 1, 'SPP', [1024, [5, 9, 13]]],
                [-1, 3, 'BottleneckCSP', [1024, False]],  # 9
            ]
        else:
            self.layers = layers

        if neck is None:
            self.neck = [
                [-1, 1, 'Conv', [512, 1, 1]],
                [-1, 1, 'Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                [-1, 3, 'BottleneckCSP', [512, False]],  # 13
                [-1, 1, 'Conv', [256, 1, 1]],
                [-1, 1, 'Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                [-1, 3, 'BottleneckCSP', [256, False]],  # 17 (P3/8-small)
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 14], 1, 'Concat', [1]],  # cat head P4
                [-1, 3, 'BottleneckCSP', [512, False]],  # 20 (P4/16-medium)
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 10], 1, 'Concat', [1]],  # cat head P5
                [-1, 3, 'BottleneckCSP', [1024, False]],  # 23 (P5/32-large)
            ]
        else:
            self.neck = neck

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.act = act
        self.yolov5 = yolov5
        self.save = save
        self.conv_decay = conv_decay
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.weight_prefix_name = weight_prefix_name
        self.layer_cfg = {
            'Conv': self._conv,
            'Focus': self._focus,
            'Bottleneck': self._bottleneck,
            'BottleneckCSP': self._bottleneckcsp,
            'BottleneckCSP2': self._bottleneckcsp2,
            'SPP': self._spp,
            'SPPCSP': self._sppcsp,
            'Upsample': self._upsample,
            'Concat': self._concat
        }
        self.act_cfg = {
            'relu': fluid.layers.relu,
            'leaky_relu': lambda x: fluid.layers.leaky_relu(x, alpha=0.1),
            'hard_swish': self._hard_swish,
            'mish': self._mish,
            'none': self._identity
        }

    def _identity(self, x):
        return x

    def _hard_swish(self, x):
        return x * fluid.layers.relu6(x + 3) / 6.

    def _softplus(self, x):
        expf = fluid.layers.exp(fluid.layers.clip(x, -200, 50))
        return fluid.layers.log(1 + expf)

    def _mish(self, x):
        return x * fluid.layers.tanh(self._softplus(x))

    def _conv(self, x, c_out, k=1, s=1, p=None, g=1, act='none', name=None):
        x = fluid.layers.conv2d(
            x,
            c_out,
            k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            param_attr=ParamAttr(name=name + '.conv.weight'),
            bias_attr=False)
        x = self._bn(x, name=name)
        x = self.act_cfg[act](x)
        return x

    def _bn(self, x, name=None):
        param_attr = ParamAttr(name=name + '.{}.weight'.format(self.norm_type))
        bias_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay),
            name=name + '.{}.bias'.format(self.norm_type))

        x = fluid.layers.batch_norm(
            input=x,
            epsilon=0.001,
            param_attr=param_attr,
            bias_attr=bias_attr,
            moving_mean_name=name + '.{}.running_mean'.format(self.norm_type),
            moving_variance_name=name +
            '.{}.running_var'.format(self.norm_type))

        return x

    def _focus(self, x, c_out, k=1, s=1, p=None, g=1, act='none', name=None):
        x = fluid.layers.concat(
            [
                x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2], x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 1::2]
            ],
            axis=1)
        x = self._conv(x, c_out, k, s, p, g, act, name + '.conv')
        return x

    def _bottleneck(self,
                    x,
                    c_out,
                    shortcut=True,
                    g=1,
                    e=0.5,
                    act='none',
                    name=None):
        c_h = int(c_out * e)
        y = self._conv(x, c_h, 1, 1, act=act, name=name + '.cv1')
        y = self._conv(y, c_out, 3, 1, g=g, act=act, name=name + '.cv2')
        if shortcut:
            y = fluid.layers.elementwise_add(x=x, y=y, act=None)
        return y

    def _bottleneckcsp(self,
                       x,
                       c_out,
                       n=1,
                       shortcut=True,
                       g=1,
                       e=0.5,
                       act='none',
                       name=None):
        c_h = int(c_out * e)
        # left branch

        y1 = self._conv(x, c_h, 1, 1, act=act, name=name + '.cv1')
        # n bottle neck
        bottleneck = self._bottleneck
        for i in six.moves.xrange(n):
            y1 = bottleneck(y1, c_h, shortcut, g, 1.0, act,
                            name + '.m.{}'.format(i))
        y1 = fluid.layers.conv2d(
            y1,
            c_h,
            1,
            1,
            param_attr=ParamAttr(name=name + '.cv3.weight'),
            bias_attr=False)
        # right branch
        y2 = fluid.layers.conv2d(
            x,
            c_h,
            1,
            1,
            param_attr=ParamAttr(name=name + '.cv2.weight'),
            bias_attr=False)
        # concat
        y = fluid.layers.concat([y1, y2], axis=1)
        # bn + act
        y = self._bn(y, name=name)
        y = self.act_cfg['leaky_relu'](y) if self.yolov5 else self.act_cfg[act](
            y)
        # conv
        y = self._conv(y, c_out, 1, 1, act=act, name=name + '.cv4')
        return y

    def _bottleneckcsp2(self,
                        x,
                        c_out,
                        n=1,
                        shortcut=False,
                        g=1,
                        e=1.0,
                        act='none',
                        name=None):
        c_h = int(c_out)
        x = self._conv(x, c_h, 1, 1, act=act, name=name + '.cv1')
        # left_branch
        y1 = x
        for i in range(n):
            y1 = self._bottleneck(y1, c_h, shortcut, g, 1.0, act,
                                  name + '.m.{}'.format(i))
        # right_branch
        y2 = fluid.layers.conv2d(
            x,
            c_h,
            1,
            1,
            param_attr=ParamAttr(name=name + '.cv2.weight'),
            bias_attr=False)
        # concat
        y = fluid.layers.concat([y1, y2], axis=1)
        # bn + act
        y = self._bn(y, name=name)
        y = self.act_cfg[act](y)
        # conv
        y = self._conv(y, c_out, 1, 1, act=act, name=name + '.cv3')
        return y

    def _spp(self, x, c_out, k=(5, 9, 13), act='none', name=None):
        c_in = int(x.shape[1])
        c_h = c_in // 2
        # conv1
        x = self._conv(x, c_h, 1, 1, act=act, name=name + '.cv1')
        ys = [x]
        # pooling
        for s in k:
            ys.append(fluid.layers.pool2d(x, s, 'max', 1, s // 2))
        y = fluid.layers.concat(ys, axis=1)
        # conv2
        y = self._conv(y, c_out, 1, 1, act=act, name=name + '.cv2')
        return y

    def _sppcsp(self, x, c_out, k=(5, 9, 13), e=0.5, act='none', name=None):
        c_h = int(2 * c_out * e)
        # left branch
        y1 = self._conv(x, c_h, 1, 1, act=act, name=name + '.cv1')
        y1 = self._conv(y1, c_h, 3, 1, act=act, name=name + '.cv3')
        y1 = self._conv(y1, c_h, 1, 1, act=act, name=name + '.cv4')
        ys = [y1]
        # pooling
        for s in k:
            ys.append(fluid.layers.pool2d(y1, s, 'max', 1, s // 2))
        y1 = fluid.layers.concat(ys, axis=1)

        y1 = self._conv(y1, c_h, 1, 1, act=act, name=name + '.cv5')
        y1 = self._conv(y1, c_h, 3, 1, act=act, name=name + '.cv6')
        # right_branch
        y2 = fluid.layers.conv2d(
            x,
            c_h,
            1,
            1,
            param_attr=ParamAttr(name=name + '.cv2.weight'),
            bias_attr=False)
        # concat
        y = fluid.layers.concat([y1, y2], axis=1)
        y = self._bn(y, name=name)
        y = self.act_cfg[act](y)
        y = self._conv(y, c_out, 1, 1, act=act, name=name + '.cv7')
        return y

    def _upsample(self, x, out_shape, scale, method, name=None):
        out_shape = None if out_shape == 'None' else out_shape
        if name == 'bilinear':
            return fluid.layers.resize_bilinear(x, out_shape, scale, name=name)
        if name == 'trilinear':
            return fluid.layers.resize_trilinear(x, out_shape, scale, name=name)
        return fluid.layers.resize_nearest(x, out_shape, scale, name=name)

    def _concat(self, x, axis, name=None):
        y = fluid.layers.concat(x, axis, name=name)
        return y

    def Print(self, x):
        fluid.layers.Print(fluid.layers.reduce_max(x))
        fluid.layers.Print(fluid.layers.reduce_min(x))
        fluid.layers.Print(fluid.layers.reduce_mean(x))
        fluid.layers.Print(fluid.layers.reduce_mean(fluid.layers.abs(x)))

    def __call__(self, x):
        prefix = self.weight_prefix_name
        gw, gd = self.width_multiple, self.depth_multiple
        layers, outputs = [], []
        for i, (f, n, m, args) in enumerate(self.layers + self.neck):
            if i == 0:
                inputs = x
            else:
                if isinstance(f, int):
                    inputs = layers[f]
                else:
                    inputs = [layers[idx] for idx in f]
            n = max(round(n * gd), 1) if n > 1 else n
            if m in [
                    'Conv', 'Bottleneck', 'BottleneckCSP', 'BottleneckCSP2',
                    'SPP', 'SPPCSP', 'Focus'
            ]:
                c_out = args[0]
                args[0] = make_divisible(c_out * gw, 8)
                if m in ['BottleneckCSP', 'BottleneckCSP2']:
                    args.insert(1, n)

            if m in ['Upsample', 'Concat']:
                layers.append(self.layer_cfg[m](
                    inputs, *args, name=prefix + '.{}'.format(i)))
            else:
                layers.append(self.layer_cfg[m](
                    inputs, *args, act=self.act, name=prefix + '.{}'.format(i)))
            if i in self.save:
                outputs.append(layers[i])
        return outputs
