# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from ppdet.core.workspace import create
from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

from . import Trainer
__all__ = ['TrainerCot']

class TrainerCot(Trainer):
    """
    Trainer for label-cotuning
    calculate the relationship between base_classes and novel_classes
    """
    def __init__(self, cfg, mode='train'):
        super(TrainerCot, self).__init__(cfg, mode)
        self.cotuning_init()

    def cotuning_init(self):    
        num_classes_novel = self.cfg['num_classes']

        self.load_weights(self.cfg.pretrain_weights)

        self.model.eval()
        relationship = self.model.relationship_learning(self.loader, num_classes_novel)
    
        self.model.init_cot_head(relationship)
        self.optimizer = create('OptimizerBuilder')(self.lr, self.model)


