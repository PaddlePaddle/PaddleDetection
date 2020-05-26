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

    def __init__(self, gamma=[0.1, 0.1], milestones=[60000, 80000],
                 values=None):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones
        self.values = values

    def __call__(self, base_lr=None, learning_rate=None):
        if self.values is not None:
            return fluid.layers.piecewise_decay(self.milestones, self.values)
        assert base_lr is not None, "either base LR or values should be provided"
        values = [base_lr]
        for g in self.gamma:
            new_lr = base_lr * g
            values.append(new_lr)
        return fluid.layers.piecewise_decay(self.milestones, values)


@serializable
class PolynomialDecay(object):
    """
    Applies polynomial decay to the initial learning rate.
    Args:
        max_iter (int): The learning rate decay steps. 
        end_lr (float): End learning rate.
        power (float): Polynomial attenuation coefficient
    """

    def __init__(self, max_iter=180000, end_lr=0.0001, power=1.0):
        super(PolynomialDecay).__init__()
        self.max_iter = max_iter
        self.end_lr = end_lr
        self.power = power

    def __call__(self, base_lr=None, learning_rate=None):
        assert base_lr is not None, "either base LR or values should be provided"
        lr = fluid.layers.polynomial_decay(base_lr, self.max_iter, self.end_lr,
                                           self.power)
        return lr


@serializable
class ExponentialDecay(object):
    """
    Applies exponential decay to the learning rate.
    Args:
        max_iter (int): The learning rate decay steps. 
        decay_rate (float): The learning rate decay rate. 
    """

    def __init__(self, max_iter, decay_rate):
        super(ExponentialDecay).__init__()
        self.max_iter = max_iter
        self.decay_rate = decay_rate

    def __call__(self, base_lr=None, learning_rate=None):
        assert base_lr is not None, "either base LR or values should be provided"
        lr = fluid.layers.exponential_decay(base_lr, self.max_iter,
                                            self.decay_rate)
        return lr


@serializable
class CosineDecay(object):
    """
    Cosine learning rate decay

    Args:
        max_iters (float): max iterations for the training process.
            if you commbine cosine decay with warmup, it is recommended that
            the max_iter is much larger than the warmup iter
    """

    def __init__(self, max_iters=180000):
        self.max_iters = max_iters

    def __call__(self, base_lr=None, learning_rate=None):
        assert base_lr is not None, "either base LR or values should be provided"
        lr = fluid.layers.cosine_decay(base_lr, 1, self.max_iters)
        return lr


@serializable
class CosineDecayWithSkip(object):
    """
    Cosine decay, with explicit support for warm up

    Args:
        total_steps (int): total steps over which to apply the decay
        skip_steps (int): skip some steps at the beginning, e.g., warm up
    """

    def __init__(self, total_steps, skip_steps=None):
        super(CosineDecayWithSkip, self).__init__()
        assert (not skip_steps or skip_steps > 0), \
            "skip steps must be greater than zero"
        assert total_steps > 0, "total step must be greater than zero"
        assert (not skip_steps or skip_steps < total_steps), \
            "skip steps must be smaller than total steps"
        self.total_steps = total_steps
        self.skip_steps = skip_steps

    def __call__(self, base_lr=None, learning_rate=None):
        steps = _decay_step_counter()
        total = self.total_steps
        if self.skip_steps is not None:
            total -= self.skip_steps

        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=base_lr,
            dtype='float32',
            persistable=True,
            name="learning_rate")

        def decay():
            cos_lr = base_lr * .5 * (cos(steps * (math.pi / total)) + 1)
            fluid.layers.tensor.assign(input=cos_lr, output=lr)

        if self.skip_steps is None:
            decay()
        else:
            skipped = steps >= self.skip_steps
            fluid.layers.cond(skipped, decay)
        return lr


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

    def __call__(self, base_lr, learning_rate):
        start_lr = base_lr * self.start_factor

        return fluid.layers.linear_lr_warmup(
            learning_rate=learning_rate,
            warmup_steps=self.steps,
            start_lr=start_lr,
            end_lr=base_lr)


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
        lr = None
        for sched in self.schedulers:
            lr = sched(self.base_lr, lr)
        return lr


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

    def __call__(self, learning_rate):
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
                  regularization=regularization,
                  **optim_args)
