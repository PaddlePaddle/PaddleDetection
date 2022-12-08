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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import time

import numpy as np
from PIL import Image, ImageOps

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import amp
from paddle.static import InputSpec
from ppdet.optimizer import ModelEMA

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.metrics import Metric, COCOMetric, VOCMetric, WiderFaceMetric, get_infer_results, KeyPointTopDownCOCOEval, KeyPointTopDownMPIIEval
from ppdet.metrics import RBoxMetric, JDEDetMetric, SNIPERCOCOMetric
from ppdet.data.source.sniper_coco import SniperCOCODataSet
from ppdet.data.source.category import get_categories
import ppdet.utils.stats as stats
from ppdet.utils import profiler

from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter, SniperProposalsGenerator
from .export_utils import _dump_infer_config, _prune_input_spec
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


