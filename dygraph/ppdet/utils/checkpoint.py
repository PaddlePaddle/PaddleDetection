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
from __future__ import unicode_literals

import errno
import os
import time
import re
import numpy as np
import paddle
import paddle.nn as nn
from .download import get_weights_path

from .logger import setup_logger
logger = setup_logger(__name__)


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') \
            or path.startswith('https://') \
            or path.startswith('ppdet://')


def _get_unique_endpoints(trainer_endpoints):
    # Sorting is to avoid different environmental variables for each card
    trainer_endpoints.sort()
    ips = set()
    unique_endpoints = set()
    for endpoint in trainer_endpoints:
        ip = endpoint.split(":")[0]
        if ip in ips:
            continue
        ips.add(ip)
        unique_endpoints.add(endpoint)
    logger.info("unique_endpoints {}".format(unique_endpoints))
    return unique_endpoints


def get_weights_path_dist(path):
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
                from paddle.distributed import ParallelEnv
                unique_endpoints = _get_unique_endpoints(ParallelEnv()
                                                         .trainer_endpoints[:])
                try:
                    os.makedirs(os.path.dirname(weight_path))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                with open(lock_path, 'w'):  # touch    
                    os.utime(lock_path, None)
                if ParallelEnv().current_endpoint in unique_endpoints:
                    get_weights_path(path)
                    os.remove(lock_path)
                else:
                    while os.path.exists(lock_path):
                        time.sleep(1)
            path = weight_path
    else:
        path = get_weights_path(path)

    return path


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path


def load_weight(model, weight, optimizer=None):
    if is_url(weight):
        weight = get_weights_path_dist(weight)

    path = _strip_postfix(weight)
    pdparam_path = path + '.pdparams'
    if not os.path.exists(pdparam_path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(pdparam_path))

    param_state_dict = paddle.load(pdparam_path)
    model_dict = model.state_dict()
    model_weight = {}
    incorrect_keys = 0

    for key in model_dict.keys():
        if key in param_state_dict.keys():
            model_weight[key] = param_state_dict[key]
        else:
            logger.info('Unmatched key: {}'.format(key))
            incorrect_keys += 1

    assert incorrect_keys == 0, "Load weight {} incorrectly, \
            {} keys unmatched, please check again.".format(weight,
                                                           incorrect_keys)
    logger.info('Finish resuming model weights: {}'.format(pdparam_path))

    model.set_dict(model_weight)

    last_epoch = 0
    if optimizer is not None and os.path.exists(path + '.pdopt'):
        optim_state_dict = paddle.load(path + '.pdopt')
        # to solve resume bug, will it be fixed in paddle 2.0
        for key in optimizer.state_dict().keys():
            if not key in optim_state_dict.keys():
                optim_state_dict[key] = optimizer.state_dict()[key]
        if 'last_epoch' in optim_state_dict:
            last_epoch = optim_state_dict.pop('last_epoch')
        optimizer.set_state_dict(optim_state_dict)

    return last_epoch


def load_pretrain_weight(model, pretrain_weight):
    if is_url(pretrain_weight):
        pretrain_weight = get_weights_path_dist(pretrain_weight)

    path = _strip_postfix(pretrain_weight)
    if not (os.path.isdir(path) or os.path.isfile(path) or
            os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path `{}` does not exists. "
                         "If you don't want to load pretrain model, "
                         "please delete `pretrain_weights` field in "
                         "config file.".format(path))

    model_dict = model.state_dict()

    weights_path = path + '.pdparams'
    param_state_dict = paddle.load(weights_path)
    ignore_set = set()
    lack_modules = set()
    for name, weight in model_dict.items():
        if name in param_state_dict.keys():
            if weight.shape != list(param_state_dict[name].shape):
                logger.info(
                    '{} not used, shape {} unmatched with {} in model.'.format(
                        name, list(param_state_dict[name].shape), weight.shape))
                param_state_dict.pop(name, None)
        else:
            lack_modules.add(name.split('.')[0])
            logger.debug('Lack weights: {}'.format(name))

    if len(lack_modules) > 0:
        logger.info('Lack weights of modules: {}'.format(', '.join(
            list(lack_modules))))

    model.set_dict(param_state_dict)
    logger.info('Finish loading model weights: {}'.format(weights_path))


def save_model(model, optimizer, save_dir, save_name, last_epoch):
    """
    save model into disk.

    Args:
        model (paddle.nn.Layer): the Layer instalce to save parameters.
        optimizer (paddle.optimizer.Optimizer): the Optimizer instance to
            save optimizer states.
        save_dir (str): the directory to be saved.
        save_name (str): the path to be saved.
        last_epoch (int): the epoch index.
    """
    if paddle.distributed.get_rank() != 0:
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    if isinstance(model, nn.Layer):
        paddle.save(model.state_dict(), save_path + ".pdparams")
    else:
        assert isinstance(model,
                          dict), 'model is not a instance of nn.layer or dict'
        paddle.save(model, save_path + ".pdparams")
    state_dict = optimizer.state_dict()
    state_dict['last_epoch'] = last_epoch
    paddle.save(state_dict, save_path + ".pdopt")
    logger.info("Save checkpoint: {}".format(save_dir))
