# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import weakref
from copy import deepcopy

from .utils import get_bn_running_state_names

__all__ = ['ModelEMA', 'SimpleModelEMA']


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
        ema_black_list (set|list|tuple, optional): The custom EMA black_list.
            Blacklist of weight names that will not participate in EMA
            calculation. Default: None.
    """

    def __init__(self,
                 model,
                 decay=0.9998,
                 ema_decay_type='threshold',
                 cycle_epoch=-1,
                 ema_black_list=None,
                 ema_filter_no_grad=False):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.ema_decay_type = ema_decay_type
        self.cycle_epoch = cycle_epoch
        self.ema_black_list = self._match_ema_black_list(
            model.state_dict().keys(), ema_black_list)
        self.state_dict = dict()
        for k, v in model.state_dict().items():
            if k in self.ema_black_list:
                self.state_dict[k] = v
            else:
                self.state_dict[k] = paddle.zeros_like(v)

        bn_states_names = get_bn_running_state_names(model)
        if ema_filter_no_grad:
            for n, p in model.named_parameters():
                if p.stop_gradient == True and n not in bn_states_names:
                    self.ema_black_list.append(n)

        self._model_state = {
            k: weakref.ref(p)
            for k, p in model.state_dict().items()
        }

    def reset(self):
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            if k in self.ema_black_list:
                self.state_dict[k] = v
            else:
                self.state_dict[k] = paddle.zeros_like(v)

    def resume(self, state_dict, step=0):
        for k, v in state_dict.items():
            if k in self.state_dict:
                if self.state_dict[k].dtype == v.dtype:
                    self.state_dict[k] = v
                else:
                    self.state_dict[k] = v.astype(self.state_dict[k].dtype)
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
            if k not in self.ema_black_list:
                v = decay * v + (1 - decay) * model_dict[k]
                v.stop_gradient = True
                self.state_dict[k] = v
        self.step += 1

    def apply(self):
        if self.step == 0:
            return self.state_dict
        state_dict = dict()
        for k, v in self.state_dict.items():
            if k in self.ema_black_list:
                v.stop_gradient = True
                state_dict[k] = v
            else:
                if self.ema_decay_type != 'exponential':
                    v = v / (1 - self._decay**self.step)
                v.stop_gradient = True
                state_dict[k] = v
        self.epoch += 1
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict

    def _match_ema_black_list(self, weight_name, ema_black_list=None):
        out_list = set()
        if ema_black_list:
            for name in weight_name:
                for key in ema_black_list:
                    if key in name:
                        out_list.add(name)
        return out_list


class SimpleModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model=None, decay=0.9996):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
        """
        self.model = deepcopy(model)
        self.decay = decay

    def update(self, model, decay=None):
        if decay is None:
            decay = self.decay

        with paddle.no_grad():
            state = {}
            msd = model.state_dict()
            for k, v in self.model.state_dict().items():
                if paddle.is_floating_point(v):
                    v *= decay
                    v += (1.0 - decay) * msd[k].detach()
                state[k] = v
            self.model.set_state_dict(state)

    def resume(self, state_dict, step=0):
        state = {}
        msd = state_dict
        for k, v in self.model.state_dict().items():
            if paddle.is_floating_point(v):
                v = msd[k].detach()
            state[k] = v
        self.model.set_state_dict(state)
        self.step = step
