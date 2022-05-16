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
import weakref
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
        use_warmup (bool): whether to use warmup. Default: True.
        min_lr_ratio (float): minimum learning rate ratio. Default: 0.
        last_plateau_epochs (int): use minimum learning rate in
            the last few epochs. Default: 0.
    """

    def __init__(self,
                 max_epochs=1000,
                 use_warmup=True,
                 min_lr_ratio=0.,
                 last_plateau_epochs=0):
        self.max_epochs = max_epochs
        self.use_warmup = use_warmup
        self.min_lr_ratio = min_lr_ratio
        self.last_plateau_epochs = last_plateau_epochs

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        assert base_lr is not None, "either base LR or values should be provided"

        max_iters = self.max_epochs * int(step_per_epoch)
        last_plateau_iters = self.last_plateau_epochs * int(step_per_epoch)
        min_lr = base_lr * self.min_lr_ratio
        if boundary is not None and value is not None and self.use_warmup:
            # use warmup
            warmup_iters = len(boundary)
            for i in range(int(boundary[-1]), max_iters):
                boundary.append(i)
                if i < max_iters - last_plateau_iters:
                    decayed_lr = min_lr + (base_lr - min_lr) * 0.5 * (math.cos(
                        (i - warmup_iters) * math.pi /
                        (max_iters - warmup_iters - last_plateau_iters)) + 1)
                    value.append(decayed_lr)
                else:
                    value.append(min_lr)
            return optimizer.lr.PiecewiseDecay(boundary, value)
        elif last_plateau_iters > 0:
            # not use warmup, but set `last_plateau_epochs` > 0
            boundary = []
            value = []
            for i in range(max_iters):
                if i < max_iters - last_plateau_iters:
                    decayed_lr = min_lr + (base_lr - min_lr) * 0.5 * (math.cos(
                        i * math.pi / (max_iters - last_plateau_iters)) + 1)
                    value.append(decayed_lr)
                else:
                    value.append(min_lr)
                if i > 0:
                    boundary.append(i)
            return optimizer.lr.PiecewiseDecay(boundary, value)

        return optimizer.lr.CosineAnnealingDecay(
            base_lr, T_max=max_iters, eta_min=min_lr)


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
        epochs (int|None): use epochs as warm up steps, the priority
            of `epochs` is higher than `steps`. Default: None.
    """

    def __init__(self, steps=500, start_factor=1. / 3, epochs=None):
        super(LinearWarmup, self).__init__()
        self.steps = steps
        self.start_factor = start_factor
        self.epochs = epochs

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        warmup_steps = self.epochs * step_per_epoch \
            if self.epochs is not None else self.steps
        for i in range(warmup_steps + 1):
            if warmup_steps > 0:
                alpha = i / warmup_steps
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


@serializable
class ExpWarmup(object):
    """
    Warm up learning rate in exponential mode
    Args:
        steps (int): warm up steps.
        epochs (int|None): use epochs as warm up steps, the priority
            of `epochs` is higher than `steps`. Default: None.
    """

    def __init__(self, steps=5, epochs=None):
        super(ExpWarmup, self).__init__()
        self.steps = steps
        self.epochs = epochs

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        warmup_steps = self.epochs * step_per_epoch if self.epochs is not None else self.steps
        for i in range(warmup_steps + 1):
            factor = (i / float(warmup_steps))**2
            value.append(base_lr * factor)
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

        if 'param_groups' in optim_args:
            assert isinstance(optim_args['param_groups'], list), ''

            param_groups = optim_args.pop('param_groups')

            params, visited = [], []
            for group in param_groups:
                assert isinstance(group,
                                  dict) and 'params' in group and isinstance(
                                      group['params'], list), ''
                _params = {
                    n: p
                    for n, p in model.named_parameters()
                    if any([k in n for k in group['params']])
                }
                _group = group.copy()
                _group.update({'params': list(_params.values())})

                params.append(_group)
                visited.extend(list(_params.keys()))

            ext_params = [
                p for n, p in model.named_parameters() if n not in visited
            ]

            if len(ext_params) < len(model.parameters()):
                params.append({'params': ext_params})

            elif len(ext_params) > len(model.parameters()):
                raise RuntimeError

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
        ema_decay_type (str): type in ['threshold', 'normal', 'exponential'],
            'threshold' as default.
        cycle_epoch (int): The epoch of interval to reset ema_param and
            step. Defaults is -1, which means not reset. Its function is to
            add a regular effect to ema, which is set according to experience
            and is effective when the total training epoch is large.
    """

    def __init__(self,
                 model,
                 decay=0.9998,
                 ema_decay_type='threshold',
                 cycle_epoch=-1):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.state_dict = dict()
        for k, v in model.state_dict().items():
            self.state_dict[k] = paddle.zeros_like(v)
        self.ema_decay_type = ema_decay_type
        self.cycle_epoch = cycle_epoch

        self._model_state = {
            k: weakref.ref(p)
            for k, p in model.state_dict().items()
        }

    def reset(self):
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            self.state_dict[k] = paddle.zeros_like(v)

    def resume(self, state_dict, step=0):
        for k, v in state_dict.items():
            if k in self.state_dict:
                self.state_dict[k] = v
        self.step = step

    def update(self, model=None):
        if self.ema_decay_type == 'threshold':
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == 'exponential':
            decay = self.decay * (1 - math.exp(-(self.step + 1) / 2000))
        else:
            decay = self.decay
        self._decay = decay

        if model is not None:
            model_dict = model.state_dict()
        else:
            model_dict = {k: p() for k, p in self._model_state.items()}
            assert all(
                [v is not None for _, v in model_dict.items()]), 'python gc.'

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
            if self.ema_decay_type != 'exponential':
                v = v / (1 - self._decay**self.step)
            v.stop_gradient = True
            state_dict[k] = v
        self.epoch += 1
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict
