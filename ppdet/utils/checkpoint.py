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


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path


def load_weight(model, weight, optimizer=None, ema=None):
    if is_url(weight):
        weight = get_weights_path(weight)

    path = _strip_postfix(weight)
    pdparam_path = path + '.pdparams'
    if not os.path.exists(pdparam_path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(pdparam_path))

    if ema is not None and os.path.exists(path + '.pdema'):
        # Exchange model and ema_model to load
        ema_state_dict = paddle.load(pdparam_path)
        param_state_dict = paddle.load(path + '.pdema')
    else:
        ema_state_dict = None
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

        if ema_state_dict is not None:
            ema.resume(ema_state_dict,
                       optim_state_dict['LR_Scheduler']['last_epoch'])
    elif ema_state_dict is not None:
        ema.resume(ema_state_dict)
    return last_epoch


def match_state_dict(model_state_dict, weight_state_dict):
    """
    Match between the model state dict and pretrained weight state dict.
    Return the matched state dict.

    The method supposes that all the names in pretrained weight state dict are
    subclass of the names in models`, if the prefix 'backbone.' in pretrained weight
    keys is stripped. And we could get the candidates for each model key. Then we
    select the name with the longest matched size as the final match result. For
    example, the model state dict has the name of
    'backbone.res2.res2a.branch2a.conv.weight' and the pretrained weight as
    name of 'res2.res2a.branch2a.conv.weight' and 'branch2a.conv.weight'. We
    match the 'res2.res2a.branch2a.conv.weight' to the model key.
    """

    model_keys = sorted(model_state_dict.keys())
    weight_keys = sorted(weight_state_dict.keys())

    def match(a, b):
        if b.startswith('backbone.res5'):
            # In Faster RCNN, res5 pretrained weights have prefix of backbone,
            # however, the corresponding model weights have difficult prefix,
            # bbox_head.
            b = b[9:]
        return a == b or a.endswith("." + b)

    match_matrix = np.zeros([len(model_keys), len(weight_keys)])
    for i, m_k in enumerate(model_keys):
        for j, w_k in enumerate(weight_keys):
            if match(m_k, w_k):
                match_matrix[i, j] = len(w_k)
    max_id = match_matrix.argmax(1)
    max_len = match_matrix.max(1)
    max_id[max_len == 0] = -1

    load_id = set(max_id)
    load_id.discard(-1)
    not_load_weight_name = []
    for idx in range(len(weight_keys)):
        if idx not in load_id:
            not_load_weight_name.append(weight_keys[idx])

    if len(not_load_weight_name) > 0:
        logger.info('{} in pretrained weight is not used in the model, '
                    'and its will not be loaded'.format(not_load_weight_name))
    matched_keys = {}
    result_state_dict = {}
    for model_id, weight_id in enumerate(max_id):
        if weight_id == -1:
            continue
        model_key = model_keys[model_id]
        weight_key = weight_keys[weight_id]
        weight_value = weight_state_dict[weight_key]
        model_value_shape = list(model_state_dict[model_key].shape)

        if list(weight_value.shape) != model_value_shape:
            logger.info(
                'The shape {} in pretrained weight {} is unmatched with '
                'the shape {} in model {}. And the weight {} will not be '
                'loaded'.format(weight_value.shape, weight_key,
                                model_value_shape, model_key, weight_key))
            continue

        assert model_key not in result_state_dict
        result_state_dict[model_key] = weight_value
        if weight_key in matched_keys:
            raise ValueError('Ambiguity weight {} loaded, it matches at least '
                             '{} and {} in the model'.format(
                                 weight_key, model_key, matched_keys[
                                     weight_key]))
        matched_keys[weight_key] = model_key
    return result_state_dict


def load_pretrain_weight(model, pretrain_weight):
    if is_url(pretrain_weight):
        pretrain_weight = get_weights_path(pretrain_weight)

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
    param_state_dict = match_state_dict(model_dict, param_state_dict)

    model.set_dict(param_state_dict)
    logger.info('Finish loading model weights: {}'.format(weights_path))


def save_model(model,
               optimizer,
               save_dir,
               save_name,
               last_epoch,
               ema_model=None):
    """
    save model into disk.

    Args:
        model (dict): the model state_dict to save parameters.
        optimizer (paddle.optimizer.Optimizer): the Optimizer instance to
            save optimizer states.
        save_dir (str): the directory to be saved.
        save_name (str): the path to be saved.
        last_epoch (int): the epoch index.
        ema_model (dict|None): the ema_model state_dict to save parameters.
    """
    if paddle.distributed.get_rank() != 0:
        return
    assert isinstance(model, dict), ("model is not a instance of dict, "
                                     "please call model.state_dict() to get.")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    # save model
    if ema_model is None:
        paddle.save(model, save_path + ".pdparams")
    else:
        assert isinstance(ema_model,
                          dict), ("ema_model is not a instance of dict, "
                                  "please call model.state_dict() to get.")
        # Exchange model and ema_model to save
        paddle.save(ema_model, save_path + ".pdparams")
        paddle.save(model, save_path + ".pdema")
    # save optimizer
    state_dict = optimizer.state_dict()
    state_dict['last_epoch'] = last_epoch
    paddle.save(state_dict, save_path + ".pdopt")
    logger.info("Save checkpoint: {}".format(save_dir))
