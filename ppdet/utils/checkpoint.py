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
import paddle.fluid as fluid
from .download import get_weights_path
import logging
logger = logging.getLogger(__name__)


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def get_weight_path(path):
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


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path


def load_weight(model, weight, optimizer=None):
    if is_url(weight):
        weight = get_weight_path(weight)

    path = _strip_postfix(weight)
    pdparam_path = path + '.pdparams'
    if not os.path.exists(pdparam_path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(pdparam_path))

    param_state_dict = paddle.load(pdparam_path)
    model.set_dict(param_state_dict)

    if optimizer is not None and os.path.exists(path + '.pdopt'):
        optim_state_dict = paddle.load(path + '.pdopt')
        optimizer.set_state_dict(optim_state_dict)
    return


def load_pretrain_weight(model,
                         pretrain_weight,
                         load_static_weights=False,
                         weight_type='pretrain'):
    assert weight_type in ['pretrain', 'finetune']
    if is_url(pretrain_weight):
        pretrain_weight = get_weight_path(pretrain_weight)

    path = _strip_postfix(pretrain_weight)
    if not (os.path.isdir(path) or os.path.isfile(path) or
            os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))

    model_dict = model.state_dict()

    if load_static_weights:
        pre_state_dict = paddle.static.load_program_state(path)
        param_state_dict = {}
        for key in model_dict.keys():
            weight_name = model_dict[key].name
            if weight_name in pre_state_dict.keys():
                logger.info('Load weight: {}, shape: {}'.format(
                    weight_name, pre_state_dict[weight_name].shape))
                param_state_dict[key] = pre_state_dict[weight_name]
            else:
                param_state_dict[key] = model_dict[key]
        model.set_dict(param_state_dict)
        return

    param_state_dict = paddle.load(path + '.pdparams')
    if weight_type == 'pretrain':
        model.backbone.set_dict(param_state_dict)
    else:
        ignore_set = set()
        for name, weight in model_dict:
            if name in param_state_dict:
                if weight.shape != param_state_dict[name].shape:
                    param_state_dict.pop(name, None)
        model.set_dict(param_state_dict)
    return


def save_model(model, optimizer, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    paddle.save(model.state_dict(), save_path + ".pdparams")
    paddle.save(optimizer.state_dict(), save_path + ".pdopt")
    logger.info("Save checkpoint: {}".format(save_dir))
