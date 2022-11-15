#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is based on https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
Ths copyright of pytorch/pytorch is a BSD-style license, as found in the LICENSE file.
"""

import math
import numpy as np

import paddle
import paddle.nn as nn

__all__ = [
    'uniform_',
    'normal_',
    'constant_',
    'ones_',
    'zeros_',
    'xavier_uniform_',
    'xavier_normal_',
    'kaiming_uniform_',
    'kaiming_normal_',
    'linear_init_',
    'conv_init_',
    'reset_initialized_parameter',
]


def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(
                shape=tensor.shape, dtype=tensor.dtype, min=a, max=b))
    return tensor


def _no_grad_normal_(tensor, mean=0., std=1.):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean=mean, std=std, shape=tensor.shape))
    return tensor


def _no_grad_fill_(tensor, value=0.):
    with paddle.no_grad():
        tensor.set_value(paddle.full_like(tensor, value, dtype=tensor.dtype))
    return tensor


def uniform_(tensor, a, b):
    """
    Modified tensor inspace using uniform_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        a (float|int): min value.
        b (float|int): max value.
    Return:
        tensor
    """
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor, mean=0., std=1.):
    """
    Modified tensor inspace using normal_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        mean (float|int): mean value.
        std (float|int): std value.
    Return:
        tensor
    """
    return _no_grad_normal_(tensor, mean, std)


def constant_(tensor, value=0.):
    """
    Modified tensor inspace using constant_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        value (float|int): value to fill tensor.
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, value)


def ones_(tensor):
    """
    Modified tensor inspace using ones_
    Args:
        tensor (paddle.Tensor): paddle Tensor
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, 1)


def zeros_(tensor):
    """
    Modified tensor inspace using zeros_
    Args:
        tensor (paddle.Tensor): paddle Tensor
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, 0)


def vector_(tensor, vector):
    with paddle.no_grad():
        tensor.set_value(paddle.to_tensor(vector, dtype=tensor.dtype))
    return tensor


def _calculate_fan_in_and_fan_out(tensor, reverse=False):
    """
    Calculate (fan_in, _fan_out) for tensor

    Args:
        tensor (Tensor): paddle.Tensor
        reverse (bool: False): tensor data format order, False by default as [fout, fin, ...]. e.g. : conv.weight [cout, cin, kh, kw] is False; linear.weight [cin, cout] is True

    Return:
        Tuple[fan_in, fan_out]
    """
    if tensor.ndim < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if reverse:
        num_input_fmaps, num_output_fmaps = tensor.shape[0], tensor.shape[1]
    else:
        num_input_fmaps, num_output_fmaps = tensor.shape[1], tensor.shape[0]

    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = np.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(tensor, gain=1., reverse=False):
    """
    Modified tensor inspace using xavier_uniform_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        gain (float): super parameter, 1. default.
        reverse (bool):  reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
    Return:
        tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    k = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -k, k)


def xavier_normal_(tensor, gain=1., reverse=False):
    """
    Modified tensor inspace using xavier_normal_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        gain (float): super parameter, 1. default.
        reverse (bool):  reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
    Return:
        tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0, std)


# reference: https://pytorch.org/docs/stable/_modules/torch/nn/init.html
def _calculate_correct_fan(tensor, mode, reverse=False):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(
            mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse)

    return fan_in if mode == 'fan_in' else fan_out


def _calculate_gain(nonlinearity, param=None):
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d'
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(
                param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(
                param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_uniform_(tensor,
                     a=0,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     reverse=False):
    """
    Modified tensor inspace using kaiming_uniform method
    Args:
        tensor (paddle.Tensor): paddle Tensor
        mode (str): ['fan_in', 'fan_out'], 'fin_in' defalut
        nonlinearity (str): nonlinearity method name
        reverse (bool):  reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
    Return:
        tensor
    """
    fan = _calculate_correct_fan(tensor, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    k = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -k, k)


def kaiming_normal_(tensor,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    reverse=False):
    """
    Modified tensor inspace using kaiming_normal_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        mode (str): ['fan_in', 'fan_out'], 'fin_in' defalut
        nonlinearity (str): nonlinearity method name
        reverse (bool):  reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
    Return:
        tensor
    """
    fan = _calculate_correct_fan(tensor, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return _no_grad_normal_(tensor, 0, std)


def linear_init_(module):
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    uniform_(module.bias, -bound, bound)


def conv_init_(module):
    bound = 1 / np.sqrt(np.prod(module.weight.shape[1:]))
    uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        uniform_(module.bias, -bound, bound)


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


@paddle.no_grad()
def reset_initialized_parameter(model, include_self=True):
    """
    Reset initialized parameter using following method for [conv, linear, embedding, bn]

    Args:
        model (paddle.Layer): paddle Layer
        include_self (bool: False): include_self for Layer.named_sublayers method. Indicate whether including itself
    Return:
        None
    """
    for _, m in model.named_sublayers(include_self=include_self):
        if isinstance(m, nn.Conv2D):
            k = float(m._groups) / (m._in_channels * m._kernel_size[0] *
                                    m._kernel_size[1])
            k = math.sqrt(k)
            _no_grad_uniform_(m.weight, -k, k)
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                _no_grad_uniform_(m.bias, -k, k)

        elif isinstance(m, nn.Linear):
            k = math.sqrt(1. / m.weight.shape[0])
            _no_grad_uniform_(m.weight, -k, k)
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                _no_grad_uniform_(m.bias, -k, k)

        elif isinstance(m, nn.Embedding):
            _no_grad_normal_(m.weight, mean=0., std=1.)

        elif isinstance(m, (nn.BatchNorm2D, nn.LayerNorm)):
            _no_grad_fill_(m.weight, 1.)
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                _no_grad_fill_(m.bias, 0)
