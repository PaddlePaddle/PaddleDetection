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
                      optimizer,
                      pretrain_ckpt=None,
                      ckpt=None,
                      ckpt_type='pretrain',
                      exclude_params=[],
                      open_debug=False):

    if ckpt_type == 'pretrain':
        ckpt = pretrain_ckpt
    ckpt = get_ckpt_path(ckpt)
    if ckpt is not None and os.path.exists(ckpt):
        param_state_dict, optim_state_dict = fluid.load_dygraph(ckpt)
        if open_debug:
            print("Loading Weights: ", param_state_dict.keys())

        if len(exclude_params) != 0:
            for k in exclude_params:
                param_state_dict.pop(k, None)

        if ckpt_type == 'pretrain':
            model.backbone.set_dict(param_state_dict)
        elif ckpt_type == 'finetune':
            model.set_dict(param_state_dict, use_structured_name=True)
        else:
            model.set_dict(param_state_dict)

        if ckpt_type == 'resume':
            if optim_state_dict is None:
                print("Can't Resume Last Training's Optimizer State!!!")
            else:
                optimizer.set_dict(optim_state_dict)
    return model


def save_dygraph_ckpt(model, optimizer, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fluid.dygraph.save_dygraph(model.state_dict(), save_dir)
    fluid.dygraph.save_dygraph(optimizer.state_dict(), save_dir)
    print("Save checkpoint:", save_dir)
