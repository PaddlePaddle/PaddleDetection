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

import os
import random
import numpy as np

import paddle
from paddle.distributed import fleet

__all__ = ['init_parallel_env', 'set_random_seed', 'init_fleet_env']


def init_fleet_env(find_unused_parameters=False):
    strategy = fleet.DistributedStrategy()
    strategy.find_unused_parameters = find_unused_parameters
    fleet.init(is_collective=True, strategy=strategy)


def init_parallel_env():
    env = os.environ
    dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    paddle.distributed.init_parallel_env()


def set_random_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
