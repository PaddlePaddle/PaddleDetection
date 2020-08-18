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

__all__ = ['Optimize']

logger = logging.getLogger(__name__)


@serializable
@register
class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float | list): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self, gamma=[0.1, 0.01], milestones=[8, 11]):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        if boundary is not None:
            boundary.extend(self.milestones * int(step_per_epoch))

        if value is not None:
            for i in self.gamma:
                value.append(base_lr * i)

        return fluid.dygraph.PiecewiseDecay(boundary, value, begin=0, step=1)


@serializable
@register
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
        boundary.append(self.steps)
        value.append(base_lr)
        return boundary, value


@serializable
@register
class BaseLR(object):
    """
    Learning Rate configuration

    Args:
        base_lr (float): base learning rate
        schedulers (list): learning rate schedulers
    """
    __inject__ = ['decay', 'warmup']

    def __init__(self, base_lr=0.01, decay=None, warmup=None):
        super(BaseLR, self).__init__()
        self.base_lr = base_lr
        self.decay = decay
        self.warmup = warmup

    def __call__(self, step_per_epoch):
        # warmup
        boundary, value = self.warmup(self.base_lr)
        # decay
        decay_lr = self.decay(self.base_lr, boundary, value, step_per_epoch)
        return decay_lr


@register
class Optimize():
    """
    Build optimizer handles

    Args:
        regularizer (object): an `Regularizer` instance
        optimizer (object): an `Optimizer` instance
    """
    __category__ = 'optim'
    __inject__ = ['learning_rate']

    def __init__(self,
                 learning_rate,
                 optimizer={'name': 'Momentum',
                            'momentum': 0.9},
                 regularizer={'name': 'L2',
                              'factor': 0.0001},
                 clip_grad_by_norm=None):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.clip_grad_by_norm = clip_grad_by_norm

    def __call__(self, params=None, step_per_epoch=1):

        if self.regularizer:
            reg_type = self.regularizer['name'] + 'Decay'
            reg_factor = self.regularizer['factor']
            regularization = getattr(regularizer, reg_type)(reg_factor)
        else:
            regularization = None

        if self.clip_grad_by_norm is not None:
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=self.clip_grad_by_norm))

        optim_args = self.optimizer.copy()
        optim_type = optim_args['name']
        del optim_args['name']
        op = getattr(optimizer, optim_type)

        return op(learning_rate=self.learning_rate(step_per_epoch),
                  parameter_list=params,
                  regularization=regularization,
                  **optim_args)
