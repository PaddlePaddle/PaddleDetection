# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.utils import try_import

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class UnstructuredPruner(object):
    def __init__(self,
                 stable_epochs,
                 pruning_epochs,
                 tunning_epochs,
                 pruning_steps,
                 ratio,
                 initial_ratio,
                 prune_params_type=None):
        self.stable_epochs = stable_epochs
        self.pruning_epochs = pruning_epochs
        self.tunning_epochs = tunning_epochs
        self.ratio = ratio
        self.prune_params_type = prune_params_type
        self.initial_ratio = initial_ratio
        self.pruning_steps = pruning_steps

    def __call__(self, model, steps_per_epoch, skip_params_func=None):
        paddleslim = try_import('paddleslim')
        from paddleslim import GMPUnstructuredPruner
        configs = {
            'pruning_strategy': 'gmp',
            'stable_iterations': self.stable_epochs * steps_per_epoch,
            'pruning_iterations': self.pruning_epochs * steps_per_epoch,
            'tunning_iterations': self.tunning_epochs * steps_per_epoch,
            'resume_iteration': 0,
            'pruning_steps': self.pruning_steps,
            'initial_ratio': self.initial_ratio,
        }

        pruner = GMPUnstructuredPruner(
            model,
            ratio=self.ratio,
            skip_params_func=skip_params_func,
            prune_params_type=self.prune_params_type,
            local_sparsity=True,
            configs=configs)

        return pruner
