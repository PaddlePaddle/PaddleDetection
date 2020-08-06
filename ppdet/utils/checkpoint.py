from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import re
import numpy as np
import paddle.fluid as fluid
from .download import get_weights_path


def get_ckpt_path(path):
    if path.startswith('http://') or path.startswith('https://'):
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


def load_dygraph_ckpt(model,
                      optimizer=None,
                      pretrain_ckpt=None,
                      ckpt=None,
                      ckpt_type=None,
                      exclude_params=[],
                      load_static_weights=False):

    assert ckpt_type in ['pretrain', 'resume', 'finetune', None]
    if ckpt_type == 'pretrain' and ckpt is None:
        ckpt = pretrain_ckpt
    ckpt = get_ckpt_path(ckpt)
    assert os.path.exists(ckpt), "Path {} does not exist.".format(ckpt)
    if load_static_weights:
        pre_state_dict = fluid.load_program_state(ckpt)
        param_state_dict = {}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            weight_name = model_dict[key].name
            if weight_name in pre_state_dict.keys():
                print('Load weight: {}, shape: {}'.format(
                    weight_name, pre_state_dict[weight_name].shape))
                param_state_dict[key] = pre_state_dict[weight_name]
            else:
                param_state_dict[key] = model_dict[key]
        model.set_dict(param_state_dict)
        return model
    param_state_dict, optim_state_dict = fluid.load_dygraph(ckpt)

    if len(exclude_params) != 0:
        for k in exclude_params:
            param_state_dict.pop(k, None)

    if ckpt_type == 'pretrain':
        model.backbone.set_dict(param_state_dict)
    else:
        model.set_dict(param_state_dict)

    if ckpt_type == 'resume':
        assert optim_state_dict, "Can't Resume Last Training's Optimizer State!!!"
        optimizer.set_dict(optim_state_dict)
    return model


def save_dygraph_ckpt(model, optimizer, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fluid.dygraph.save_dygraph(model.state_dict(), save_dir)
    fluid.dygraph.save_dygraph(optimizer.state_dict(), save_dir)
    print("Save checkpoint:", save_dir)
