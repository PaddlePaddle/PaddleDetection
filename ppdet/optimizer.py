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

import sys
import math
import paddle
import paddle.nn as nn

import paddle.optimizer as optimizer
import paddle.regularizer as regularizer

from ppdet.core.workspace import register, serializable
import copy

__all__ = ['LearningRate', 'OptimizerBuilder']

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@serializable
class CosineDecay(object):
    """
    Cosine learning rate decay

    Args:
        max_epochs (int): max epochs for the training process.
            if you commbine cosine decay with warmup, it is recommended that
            the max_iters is much larger than the warmup iter
    """

    def __init__(self, max_epochs=1000, use_warmup=True):
        self.max_epochs = max_epochs
        self.use_warmup = use_warmup

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        assert base_lr is not None, "either base LR or values should be provided"

        max_iters = self.max_epochs * int(step_per_epoch)

        if boundary is not None and value is not None and self.use_warmup:
            for i in range(int(boundary[-1]), max_iters):
                boundary.append(i)

                decayed_lr = base_lr * 0.5 * (
                    math.cos(i * math.pi / max_iters) + 1)
                value.append(decayed_lr)
            return optimizer.lr.PiecewiseDecay(boundary, value)

        return optimizer.lr.CosineAnnealingDecay(base_lr, T_max=max_iters)


@serializable
class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float | list): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self,
                 gamma=[0.1, 0.01],
                 milestones=[8, 11],
                 values=None,
                 use_warmup=True):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones
        self.values = values
        self.use_warmup = use_warmup

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        if boundary is not None and self.use_warmup:
            boundary.extend([int(step_per_epoch) * i for i in self.milestones])
        else:
            # do not use LinearWarmup
            boundary = [int(step_per_epoch) * i for i in self.milestones]
            value = [base_lr]  # during step[0, boundary[0]] is base_lr

        # self.values is setted directly in config 
        if self.values is not None:
            assert len(self.milestones) + 1 == len(self.values)
            return optimizer.lr.PiecewiseDecay(boundary, self.values)

        # value is computed by self.gamma
        value = value if value is not None else [base_lr]
        for i in self.gamma:
            value.append(base_lr * i)

        return optimizer.lr.PiecewiseDecay(boundary, value)


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

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        for i in range(self.steps + 1):
            if self.steps > 0:
                alpha = i / self.steps
                factor = self.start_factor * (1 - alpha) + alpha
                lr = base_lr * factor
                value.append(lr)
            if i > 0:
                boundary.append(i)
        return boundary, value


@serializable
class BurninWarmup(object):
    """
    Warm up learning rate in burnin mode
    Args:
        steps (int): warm up steps
    """

    def __init__(self, steps=1000):
        super(BurninWarmup, self).__init__()
        self.steps = steps

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        burnin = min(self.steps, step_per_epoch)
        for i in range(burnin + 1):
            factor = (i * 1.0 / burnin)**4
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
        self.schedulers = []

        schedulers = copy.deepcopy(schedulers)
        for sched in schedulers:
            if isinstance(sched, dict):
                # support dict sched instantiate
                module = sys.modules[__name__]
                type = sched.pop("name")
                scheduler = getattr(module, type)(**sched)
                self.schedulers.append(scheduler)
            else:
                self.schedulers.append(sched)

    def __call__(self, step_per_epoch):
        assert len(self.schedulers) >= 1
        if not self.schedulers[0].use_warmup:
            return self.schedulers[0](base_lr=self.base_lr,
                                      step_per_epoch=step_per_epoch)

        # TODO: split warmup & decay 
        # warmup
        boundary, value = self.schedulers[1](self.base_lr, step_per_epoch)
        # decay
        decay_lr = self.schedulers[0](self.base_lr, boundary, value,
                                      step_per_epoch)
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

    def __call__(self, learning_rate, model=None):
        if self.clip_grad_by_norm is not None:
            grad_clip = nn.ClipGradByGlobalNorm(
                clip_norm=self.clip_grad_by_norm)
        else:
            grad_clip = None
        if self.regularizer and self.regularizer != 'None':
            reg_type = self.regularizer['type'] + 'Decay'
            reg_factor = self.regularizer['factor']
            regularization = getattr(regularizer, reg_type)(reg_factor)
        else:
            regularization = None

        optim_args = self.optimizer.copy()
        optim_type = optim_args['type']
        del optim_args['type']
        if optim_type != 'AdamW':
            optim_args['weight_decay'] = regularization
        op = getattr(optimizer, optim_type)

        if 'without_weight_decay_params' in optim_args:
            keys = optim_args['without_weight_decay_params']
            params = [{
                'params': [
                    p for n, p in model.named_parameters()
                    if any([k in n for k in keys])
                ],
                'weight_decay': 0.
            }, {
                'params': [
                    p for n, p in model.named_parameters()
                    if all([k not in n for k in keys])
                ]
            }]
            del optim_args['without_weight_decay_params']
        else:
            params = model.parameters()

        return op(learning_rate=learning_rate,
                  parameters=params,
                  grad_clip=grad_clip,
                  **optim_args)


class ModelEMA(object):
    """
    Exponential Weighted Average for Deep Neutal Networks
    Args:
        model (nn.Layer): Detector of model.
        decay (int):  The decay used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = decay * ema_param + (1 - decay) * cur_param`.
            Defaults is 0.9998.
        use_thres_step (bool): Whether set decay by thres_step or not 
        cycle_epoch (int): The epoch of interval to reset ema_param and 
            step. Defaults is -1, which means not reset. Its function is to
            add a regular effect to ema, which is set according to experience 
            and is effective when the total training epoch is large.
    """

    def __init__(self,
                 model,
                 decay=0.9998,
                 use_thres_step=False,
                 cycle_epoch=-1):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.state_dict = dict()
        for k, v in model.state_dict().items():
            self.state_dict[k] = paddle.zeros_like(v)
        self.use_thres_step = use_thres_step
        self.cycle_epoch = cycle_epoch

    def reset(self):
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            self.state_dict[k] = paddle.zeros_like(v)

    def update(self, model):
        if self.use_thres_step:
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        else:
            decay = self.decay
        self._decay = decay
        model_dict = model.state_dict()
        for k, v in self.state_dict.items():
            v = decay * v + (1 - decay) * model_dict[k]
            v.stop_gradient = True
            self.state_dict[k] = v
        self.step += 1

    def apply(self):
        if self.step == 0:
            return self.state_dict
        state_dict = dict()
        for k, v in self.state_dict.items():
            v = v / (1 - self._decay**self.step)
            v.stop_gradient = True
            state_dict[k] = v
        self.epoch += 1
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict
