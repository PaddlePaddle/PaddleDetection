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

from . import prune
from . import quant
from . import distill
from . import unstructured_prune

from .prune import *
from .quant import *
from .distill import *
from .unstructured_prune import *
from .ofa import *

import yaml
from ppdet.core.workspace import load_config
from ppdet.utils.checkpoint import load_pretrain_weight


def build_slim_model(cfg, slim_cfg, mode='train'):
    with open(slim_cfg) as f:
        slim_load_cfg = yaml.load(f, Loader=yaml.Loader)
    if mode != 'train' and slim_load_cfg['slim'] == 'Distill':
        return cfg

    if slim_load_cfg['slim'] == 'Distill':
        model = DistillModel(cfg, slim_cfg)
        cfg['model'] = model
        cfg['slim_type'] = cfg.slim
    elif slim_load_cfg['slim'] == 'OFA':
        load_config(slim_cfg)
        model = create(cfg.architecture)
        load_pretrain_weight(model, cfg.weights)
        slim = create(cfg.slim)
        cfg['slim'] = slim
        cfg['model'] = slim(model, model.state_dict())
        cfg['slim_type'] = cfg.slim
    elif slim_load_cfg['slim'] == 'DistillPrune':
        if mode == 'train':
            model = DistillModel(cfg, slim_cfg)
            pruner = create(cfg.pruner)
            pruner(model.student_model)
        else:
            model = create(cfg.architecture)
            weights = cfg.weights
            load_config(slim_cfg)
            pruner = create(cfg.pruner)
            model = pruner(model)
            load_pretrain_weight(model, weights)
        cfg['model'] = model
        cfg['slim_type'] = cfg.slim
    elif slim_load_cfg['slim'] == 'PTQ':
        model = create(cfg.architecture)
        load_config(slim_cfg)
        load_pretrain_weight(model, cfg.weights)
        slim = create(cfg.slim)
        cfg['slim'] = slim
        cfg['model'] = slim(model)
        cfg['slim_type'] = cfg.slim
    elif slim_load_cfg['slim'] == 'UnstructuredPruner':
        load_config(slim_cfg)
        slim = create(cfg.slim)
        cfg['slim_type'] = cfg.slim
        cfg['slim'] = slim
        cfg['unstructured_prune'] = True
    else:
        load_config(slim_cfg)
        model = create(cfg.architecture)
        if mode == 'train':
            load_pretrain_weight(model, cfg.pretrain_weights)
        slim = create(cfg.slim)
        cfg['slim_type'] = cfg.slim
        # TODO: fix quant export model in framework.
        if mode == 'test' and slim_load_cfg['slim'] == 'QAT':
            slim.quant_config['activation_preprocess_type'] = None
        cfg['model'] = slim(model)
        cfg['slim'] = slim
        if mode != 'train':
            load_pretrain_weight(cfg['model'], cfg.weights)

    return cfg
