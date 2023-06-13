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

import os
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


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path


def load_weight(model, weight, optimizer=None, ema=None, exchange=True):
    if is_url(weight):
        weight = get_weights_path(weight)

    path = _strip_postfix(weight)
    pdparam_path = path + '.pdparams'
    if not os.path.exists(pdparam_path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(pdparam_path))

    if ema is not None and os.path.exists(path + '.pdema'):
        if exchange:
            # Exchange model and ema_model to load
            logger.info('Exchange model and ema_model to load:')
            ema_state_dict = paddle.load(pdparam_path)
            logger.info('Loading ema_model weights from {}'.format(path +
                                                                   '.pdparams'))
            param_state_dict = paddle.load(path + '.pdema')
            logger.info('Loading model weights from {}'.format(path + '.pdema'))
        else:
            ema_state_dict = paddle.load(path + '.pdema')
            logger.info('Loading ema_model weights from {}'.format(path +
                                                                   '.pdema'))
            param_state_dict = paddle.load(pdparam_path)
            logger.info('Loading model weights from {}'.format(path +
                                                               '.pdparams'))
    else:
        ema_state_dict = None
        param_state_dict = paddle.load(pdparam_path)

    if hasattr(model, 'modelTeacher') and hasattr(model, 'modelStudent'):
        print('Loading pretrain weights for Teacher-Student framework.')
        print('Loading pretrain weights for Student model.')
        student_model_dict = model.modelStudent.state_dict()
        student_param_state_dict = match_state_dict(
            student_model_dict, param_state_dict, mode='student')
        model.modelStudent.set_dict(student_param_state_dict)
        print('Loading pretrain weights for Teacher model.')
        teacher_model_dict = model.modelTeacher.state_dict()

        teacher_param_state_dict = match_state_dict(
            teacher_model_dict, param_state_dict, mode='teacher')
        model.modelTeacher.set_dict(teacher_param_state_dict)

    else:
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


def match_state_dict(model_state_dict, weight_state_dict, mode='default'):
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

    def teacher_match(a, b):
        # skip student params
        if b.startswith('modelStudent'):
            return False
        return a == b or a.endswith("." + b) or b.endswith("." + a)

    def student_match(a, b):
        # skip teacher params
        if b.startswith('modelTeacher'):
            return False
        return a == b or a.endswith("." + b) or b.endswith("." + a)

    def match(a, b):
        if b.startswith('backbone.res5'):
            b = b[9:]
        return a == b or a.endswith("." + b)

    if mode == 'student':
        match_op = student_match
    elif mode == 'teacher':
        match_op = teacher_match
    else:
        match_op = match

    match_matrix = np.zeros([len(model_keys), len(weight_keys)])
    for i, m_k in enumerate(model_keys):
        for j, w_k in enumerate(weight_keys):
            if match_op(m_k, w_k):
                match_matrix[i, j] = len(w_k)
    max_id = match_matrix.argmax(1)
    max_len = match_matrix.max(1)
    max_id[max_len == 0] = -1
    load_id = set(max_id)
    load_id.discard(-1)
    not_load_weight_name = []
    if weight_keys[0].startswith('modelStudent') or weight_keys[0].startswith(
            'modelTeacher'):
        for match_idx in range(len(max_id)):
            if max_id[match_idx] == -1:
                not_load_weight_name.append(model_keys[match_idx])
        if len(not_load_weight_name) > 0:
            logger.info('{} in model is not matched with pretrained weights, '
                        'and its will be trained from scratch'.format(
                            not_load_weight_name))

    else:
        for idx in range(len(weight_keys)):
            if idx not in load_id:
                not_load_weight_name.append(weight_keys[idx])

        if len(not_load_weight_name) > 0:
            logger.info('{} in pretrained weight is not used in the model, '
                        'and its will not be loaded'.format(
                            not_load_weight_name))
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


def load_pretrain_weight(model, pretrain_weight, ARSL_eval=False):
    if is_url(pretrain_weight):
        pretrain_weight = get_weights_path(pretrain_weight)

    path = _strip_postfix(pretrain_weight)
    if not (os.path.isdir(path) or os.path.isfile(path) or
            os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path `{}` does not exists. "
                         "If you don't want to load pretrain model, "
                         "please delete `pretrain_weights` field in "
                         "config file.".format(path))
    teacher_student_flag = False
    if not ARSL_eval:
        if hasattr(model, 'modelTeacher') and hasattr(model, 'modelStudent'):
            print('Loading pretrain weights for Teacher-Student framework.')
            print(
                'Assert Teacher model has the same structure with Student model.'
            )
            model_dict = model.modelStudent.state_dict()
            teacher_student_flag = True
        else:
            model_dict = model.state_dict()

        weights_path = path + '.pdparams'
        param_state_dict = paddle.load(weights_path)
        param_state_dict = match_state_dict(model_dict, param_state_dict)
        for k, v in param_state_dict.items():
            if isinstance(v, np.ndarray):
                v = paddle.to_tensor(v)
            if model_dict[k].dtype != v.dtype:
                param_state_dict[k] = v.astype(model_dict[k].dtype)

        if teacher_student_flag:
            model.modelStudent.set_dict(param_state_dict)
            model.modelTeacher.set_dict(param_state_dict)
        else:
            model.set_dict(param_state_dict)
        logger.info('Finish loading model weights: {}'.format(weights_path))

    else:
        weights_path = path + '.pdparams'
        param_state_dict = paddle.load(weights_path)
        student_model_dict = model.modelStudent.state_dict()
        student_param_state_dict = match_state_dict(
            student_model_dict, param_state_dict, mode='student')
        model.modelStudent.set_dict(student_param_state_dict)
        print('Loading pretrain weights for Teacher model.')
        teacher_model_dict = model.modelTeacher.state_dict()

        teacher_param_state_dict = match_state_dict(
            teacher_model_dict, param_state_dict, mode='teacher')
        model.modelTeacher.set_dict(teacher_param_state_dict)
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    # save model
    if isinstance(model, nn.Layer):
        paddle.save(model.state_dict(), save_path + ".pdparams")
    else:
        assert isinstance(model,
                          dict), 'model is not a instance of nn.layer or dict'
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


def save_semi_model(teacher_model, student_model, optimizer, save_dir,
                    save_name, last_epoch, last_iter):
    """
    save teacher and student model into disk.
    Args:
        teacher_model (dict): the teacher_model state_dict to save parameters.
        student_model (dict): the student_model state_dict to save parameters.
        optimizer (paddle.optimizer.Optimizer): the Optimizer instance to
            save optimizer states.
        save_dir (str): the directory to be saved.
        save_name (str): the path to be saved.
        last_epoch (int): the epoch index.
        last_iter (int): the iter index.
    """
    if paddle.distributed.get_rank() != 0:
        return
    assert isinstance(teacher_model, dict), (
        "teacher_model is not a instance of dict, "
        "please call teacher_model.state_dict() to get.")
    assert isinstance(student_model, dict), (
        "student_model is not a instance of dict, "
        "please call student_model.state_dict() to get.")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    # save model
    paddle.save(teacher_model, save_path + str(last_epoch) + "epoch_t.pdparams")
    paddle.save(student_model, save_path + str(last_epoch) + "epoch_s.pdparams")

    # save optimizer
    state_dict = optimizer.state_dict()
    state_dict['last_epoch'] = last_epoch
    state_dict['last_iter'] = last_iter
    paddle.save(state_dict, save_path + str(last_epoch) + "epoch.pdopt")
    logger.info("Save checkpoint: {}".format(save_dir))
