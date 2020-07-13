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

import math
import logging

from paddle import fluid

import paddle.fluid.optimizer as optimizer
import paddle.fluid.regularizer as regularizer
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.layers.ops import cos

from ppdet.core.workspace import register, serializable

__all__ = ['LearningRate', 'OptimizerBuilder']

logger = logging.getLogger(__name__)


@serializable
class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float | list): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self, gamma=[0.1, 0.01], milestones=[60000, 80000]):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones

    def __call__(self, base_lr=None, boundary=None, value=None):
        if boundary is not None:
            boundary.extend(self.milestones)

        if value is not None:
            for i in self.gamma:
                value.append(base_lr * i)

        return fluid.dygraph.PiecewiseDecay(boundary, value, begin=0, step=1)


@serializable
class LinearWarmup(object):
    """
    Warm up learning rate linearly

    Args:
        steps (int): warm up steps
        start_factor (float): initial learning rate factor
    """

    def __init__(self, steps=500, start_factor=1. / 3):
        super(LinearWarmup, self).__init__()
        self.steps = steps
        self.start_factor = start_factor

    def __call__(self, base_lr):
        boundary = []
        value = []
        for i in range(self.steps):
            alpha = i / self.steps
            factor = self.start_factor * (1 - alpha) + alpha
            lr = base_lr * factor
            value.append(lr)
            if i > 0:
                boundary.append(i)
        return boundary, value


@register
class LearningRate(object):
    """
    Learning Rate configuration

    Args:
        base_lr (float): base learning rate
        schedulers (list): learning rate schedulers
    """
    __category__ = 'optim'

    def __init__(self,
                 base_lr=0.01,
                 schedulers=[PiecewiseDecay(), LinearWarmup()]):
        super(LearningRate, self).__init__()
        self.base_lr = base_lr
        self.schedulers = schedulers

    def __call__(self):
        # TODO: split warmup & decay 
        # warmup
        boundary, value = self.schedulers[1](self.base_lr)
        # decay
        decay_lr = self.schedulers[0](self.base_lr, boundary, value)
        return decay_lr


@register
class OptimizerBuilder():
    """
    Build optimizer handles

    Args:
        regularizer (object): an `Regularizer` instance
        optimizer (object): an `Optimizer` instance
    """
    __category__ = 'optim'

    def __init__(self,
                 clip_grad_by_norm=None,
                 regularizer={'type': 'L2',
                              'factor': .0001},
                 optimizer={'type': 'Momentum',
                            'momentum': .9}):
        self.clip_grad_by_norm = clip_grad_by_norm
        self.regularizer = regularizer
        self.optimizer = optimizer

    def __call__(self, learning_rate, params=None):
        if self.clip_grad_by_norm is not None:
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=self.clip_grad_by_norm))

        if self.regularizer:
            reg_type = self.regularizer['type'] + 'Decay'
            reg_factor = self.regularizer['factor']
            regularization = getattr(regularizer, reg_type)(reg_factor)
        else:
            regularization = None

        optim_args = self.optimizer.copy()
        optim_type = optim_args['type']
        del optim_args['type']
        op = getattr(optimizer, optim_type)
        return op(learning_rate=learning_rate,
                  parameter_list=params,
                  regularization=regularization,
                  **optim_args)
