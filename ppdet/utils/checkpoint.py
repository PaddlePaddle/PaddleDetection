from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import re
import numpy as np
import paddle.fluid as fluid


def load_dygraph_ckpt(model,
                      optimizer,
                      pretrain_ckpt=None,
                      ckpt=None,
                      ckpt_type='pretrain',
                      exclude_params=[],
                      open_debug=False):

    if 'npz' in pretrain_ckpt and os.path.exists(pretrain_ckpt):
        w_dict = np.load(pretrain_ckpt)
        new_w_dict = {}
        for k, v in w_dict.items():
            if 'bn_conv1' in k:
                nk = k[3:]
            elif k == 'conv1_weights':
                nk = 'conv1_conv_weight'
            elif 'bn' in k and 'conv1' not in k:
                nk = 'res' + k[2:]
            elif 'weights' in k:
                nk = k[:-7] + 'conv_weight'
            else:
                nk = k

            ks = nk.split('_')
            new_k = ''
            for i in ks:
                new_k += i + '.'
            new_k = new_k[:-1]

            if open_debug:
                print("Rename weight: ", k, " ---> ", new_k)

            new_w_dict[new_k] = v

        weight_keys = new_w_dict.keys()
        model_states = model.state_dict()
        model_keys = model_states.keys()
        for mk in model_keys:
            for wk in weight_keys:
                res = re.search(wk, mk)
                if res is not None:
                    model_states[mk] = v
                    if open_debug:
                        print("Loading weight: ", mk, model_states[mk].shape,
                              " <--- ", wk, new_w_dict[wk].shape)

        pretrain_ckpt = "./output/resnet50.pdparams"
        if not os.path.exists(pretrain_ckpt):
            fluid.dygraph.save_dygraph(model.backbone.state_dict(),
                                       pretrain_ckpt)

    if ckpt_type == 'pretrain':
        ckpt = pretrain_ckpt

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

        if type == 'resume':
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
