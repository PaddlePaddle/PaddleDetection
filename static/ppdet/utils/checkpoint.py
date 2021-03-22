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

import errno
import os
import shutil
import tempfile
import time
import numpy as np
import re
import paddle.fluid as fluid

from .download import get_weights_path

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'load_checkpoint',
    'load_and_fusebn',
    'load_params',
    'save',
]


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def _get_weight_path(path):
    env = os.environ
    if 'PADDLE_TRAINERS_NUM' in env and 'PADDLE_TRAINER_ID' in env:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        num_trainers = int(env['PADDLE_TRAINERS_NUM'])
        if num_trainers <= 1:
            path = get_weights_path(path)
        else:
            from ppdet.utils.download import map_path, WEIGHTS_HOME
            weight_path = map_path(path, WEIGHTS_HOME)
            lock_path = weight_path + '.lock'
            if not os.path.exists(weight_path):
                try:
                    os.makedirs(os.path.dirname(weight_path))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                with open(lock_path, 'w'):  # touch    
                    os.utime(lock_path, None)
                if trainer_id == 0:
                    get_weights_path(path)
                    os.remove(lock_path)
                else:
                    while os.path.exists(lock_path):
                        time.sleep(1)
            path = weight_path
    else:
        path = get_weights_path(path)
    return path


def _load_state(path):
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path


def load_params(exe, prog, path, ignore_params=[]):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
        ignore_params (list): ignore variable to load when finetuning.
            It can be specified by finetune_exclude_pretrained_params 
            and the usage can refer to docs/advanced_tutorials/TRANSFER_LEARNING.md
    """

    if is_url(path):
        path = _get_weight_path(path)

    path = _strip_postfix(path)
    if not (os.path.isdir(path) or os.path.isfile(path) or
            os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))

    logger.debug('Loading parameters from {}...'.format(path))

    ignore_set = set()
    state = _load_state(path)

    # ignore the parameter which mismatch the shape 
    # between the model and pretrain weight.
    all_var_shape = {}
    for block in prog.blocks:
        for param in block.all_parameters():
            all_var_shape[param.name] = param.shape
    ignore_set.update([
        name for name, shape in all_var_shape.items()
        if name in state and shape != state[name].shape
    ])

    if ignore_params:
        all_var_names = [var.name for var in prog.list_vars()]
        ignore_list = filter(
            lambda var: any([re.match(name, var) for name in ignore_params]),
            all_var_names)
        ignore_set.update(list(ignore_list))

    if len(ignore_set) > 0:
        for k in ignore_set:
            if k in state:
                logger.warning('variable {} not used'.format(k))
                del state[k]
    fluid.io.set_program_state(prog, state)


def load_checkpoint(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
    """
    if is_url(path):
        path = _get_weight_path(path)

    path = _strip_postfix(path)
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    fluid.load(prog, path, executor=exe)


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
    fluid.save(prog, path)


def load_and_fusebn(exe, prog, path):
    """
    Fuse params of batch norm to scale and bias.

    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): save weight from which Program object.
        path (string): the path to save model.
    """
    logger.debug('Load model and fuse batch norm if have from {}...'.format(
        path))

    if is_url(path):
        path = _get_weight_path(path)

    if not os.path.exists(path):
        raise ValueError("Model path {} does not exists.".format(path))

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
    state = _load_state(path)

    def check_mean_and_bias(prefix):
        m = prefix + 'mean'
        v = prefix + 'variance'
        return v in state and m in state

    has_mean_bias = True

    with fluid.program_guard(prog, fluid.Program()):
        for block in prog.blocks:
            ops = list(block.ops)
            if not has_mean_bias:
                break
            for op in ops:
                if op.type == 'affine_channel':
                    # remove 'scale' as prefix
                    scale_name = op.input('Scale')[0]  # _scale
                    bias_name = op.input('Bias')[0]  # _offset
                    prefix = scale_name[:-5]
                    mean_name = prefix + 'mean'
                    variance_name = prefix + 'variance'
                    if not check_mean_and_bias(prefix):
                        has_mean_bias = False
                        break

                    bias = block.var(bias_name)

                    mean_vb = block.create_var(
                        name=mean_name,
                        type=bias.type,
                        shape=bias.shape,
                        dtype=bias.dtype)
                    variance_vb = block.create_var(
                        name=variance_name,
                        type=bias.type,
                        shape=bias.shape,
                        dtype=bias.dtype)

                    mean_variances.add(mean_vb)
                    mean_variances.add(variance_vb)

                    bn_vars.append(
                        [scale_name, bias_name, mean_name, variance_name])

    if not has_mean_bias:
        fluid.io.set_program_state(prog, state)
        logger.warning(
            "There is no paramters of batch norm in model {}. "
            "Skip to fuse batch norm. And load paramters done.".format(path))
        return

    fluid.load(prog, path, exe)
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
