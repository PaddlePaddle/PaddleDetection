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
from __future__ import unicode_literals

import os
import shutil
import numpy as np

import paddle.fluid as fluid

from .download import get_weights_path

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'load_checkpoint',
    'load_and_fusebn',
    'save',
]


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def load_pretrain(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
    """
    if is_url(path):
        path = get_weights_path(path)

    if not os.path.exists(path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))

    logger.info('Loading pretrained model from {}...'.format(path))

    def _if_exist(var):
        b = os.path.exists(os.path.join(path, var.name))
        if b:
            logger.debug('load weight {}'.format(var.name))
        return b

    fluid.io.load_vars(exe, path, prog, predicate=_if_exist)


def load_checkpoint(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
    """
    if is_url(path):
        path = get_weights_path(path)

    if not os.path.exists(path):
        raise ValueError("Model checkpoint path {} does not "
                         "exists.".format(path))

    logger.info('Loading checkpoint from {}...'.format(path))
    fluid.io.load_persistables(exe, path, prog)


def global_step(scope=None):
    """
    Load global step in scope.
    Args:
        scope (fluid.Scope): load global step from which scope. If None,
            from default global_scope().

    Returns:
        global step: int.
    """
    if scope is None:
        scope = fluid.global_scope()
    v = scope.find_var('@LR_DECAY_COUNTER@')
    step = np.array(v.get_tensor())[0] if v else 0
    return step


def save(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): save weight from which Program object.
        path (string): the path to save model.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    logger.info('Save model to {}.'.format(path))
    fluid.io.save_persistables(exe, path, prog)


def load_and_fusebn(exe, prog, path):
    """
    Fuse params of batch norm to scale and bias.

    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): save weight from which Program object.
        path (string): the path to save model.
    """
    logger.info('Load model and fuse batch norm from {}...'.format(path))
    if is_url(path):
        path = get_weights_path(path)

    if not os.path.exists(path):
        raise ValueError("Model path {} does not exists.".format(path))

    def _if_exist(var):
        b = os.path.exists(os.path.join(path, var.name))
        if b:
            logger.debug('load weight {}'.format(var.name))
        return b

    all_vars = list(filter(_if_exist, prog.list_vars()))

    # Since the program uses affine-channel, there is no running mean and var
    # in the program, here append running mean and var.
    # NOTE, the params of batch norm should be like:
    #  x_scale
    #  x_offset
    #  x_mean
    #  x_variance
    #  x is any prefix
    mean_variances = set()
    bn_vars = []

    bn_in_path = True

    inner_prog = fluid.Program()
    inner_start_prog = fluid.Program()
    with fluid.program_guard(inner_prog, inner_start_prog):
        for block in prog.blocks:
            ops = list(block.ops)
            if not bn_in_path:
                break
            for op in ops:
                if op.type == 'affine_channel':
                    # remove 'scale' as prefix
                    scale_name = op.input('Scale')[0]  # _scale
                    bias_name = op.input('Bias')[0]  # _offset
                    prefix = scale_name[:-5]
                    mean_name = prefix + 'mean'
                    variance_name = prefix + 'variance'

                    if not os.path.exists(os.path.join(path, mean_name)):
                        bn_in_path = False
                        break
                    if not os.path.exists(os.path.join(path, variance_name)):
                        bn_in_path = False
                        break

                    bias = block.var(bias_name)
                    mean_vb = fluid.layers.create_parameter(
                        bias.shape, bias.dtype, mean_name)
                    variance_vb = fluid.layers.create_parameter(
                        bias.shape, bias.dtype, variance_name)
                    mean_variances.add(mean_vb)
                    mean_variances.add(variance_vb)

                    bn_vars.append(
                        [scale_name, bias_name, mean_name, variance_name])

    if not bn_in_path:
        raise ValueError("There is no params of batch norm in model {}.".format(
            path))

    # load running mean and running variance on cpu place into global scope.
    place = fluid.CPUPlace()
    exe_cpu = fluid.Executor(place)
    fluid.io.load_vars(exe_cpu, path, vars=[v for v in mean_variances])

    # load params on real place into global scope.
    fluid.io.load_vars(exe, path, prog, vars=all_vars)

    eps = 1e-5
    for names in bn_vars:
        scale_name, bias_name, mean_name, var_name = names

        scale = fluid.global_scope().find_var(scale_name).get_tensor()
        bias = fluid.global_scope().find_var(bias_name).get_tensor()
        mean = fluid.global_scope().find_var(mean_name).get_tensor()
        var = fluid.global_scope().find_var(var_name).get_tensor()

        scale_arr = np.array(scale)
        bias_arr = np.array(bias)
        mean_arr = np.array(mean)
        var_arr = np.array(var)

        bn_std = np.sqrt(np.add(var_arr, eps))
        new_scale = np.float32(np.divide(scale_arr, bn_std))
        new_bias = bias_arr - mean_arr * new_scale

        # fuse to scale and bias in affine_channel
        scale.set(new_scale, exe.place)
        bias.set(new_bias, exe.place)
