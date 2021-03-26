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

from .prune import *
from .quant import *
from .distill import *

import yaml
from paddle.distributed import ParallelEnv
from ppdet.core.workspace import load_config
from ppdet.utils.checkpoint import load_pretrain_weight


def build_slim_model(config, slim_cfg, mode='train'):
    cfg = load_config(config)
    with open(slim_cfg) as f:
        slim_load_cfg = yaml.load(f, Loader=yaml.Loader)

    if slim_load_cfg['slim'] == 'Distill':
        model = DistillModel(cfg, slim_cfg)
        cfg['model'] = model
    else:
        load_config(slim_cfg)
        place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
        place = paddle.set_device(place)
        model = create(cfg.architecture)
        if mode == 'train':
            load_pretrain_weight(model, cfg.pretrain_weights)
        slim = create(cfg.slim)
        cfg['model'] = slim(model)

    return cfg
