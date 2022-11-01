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

def softmax_np(x):
    max_el = np.max(x, axis=1, keepdims=True)
    x = x - max_el
    x = np.exp(x)
    s = np.sum(x, axis=1, keepdims=True)
    return x / s

def read_label_names(filepath):
    return [x.strip() for x in open(filepath).readlines()]

class TrainerCot(Trainer):
    def __init__(self, cfg, mode='train'):
        super(TrainerCot, self).__init__(cfg, mode)
        # assert not self.use_ema
        self.cotuning_init()

    def cotuning_init(self):    
        # num_classes_novel = self.cfg['num_classes']
        
        # self.cfg['num_classes'] = 80
        # self.rebuild_model()
        self.load_weights(self.cfg.pretrain_weights)

        cot_lambda = self.cfg['cot_lambda']
        cot_scale = self.cfg['cot_scale']
        self.coco_labels = read_label_names('./coco_label_list.txt')
        if self.cfg['metric'] == 'VOC':
            self.novel_labels = read_label_names(os.path.join(self.dataset.dataset_dir, self.dataset.label_list))
        elif os.path.exists(os.path.join(self.dataset.dataset_dir, 'novel_label_list.txt')):
            self.novel_labels = read_label_names(os.path.join(self.dataset.dataset_dir, 'novel_label_list.txt'))
        else:
            self.novel_labels = [str(i) for i in range(4)]
        # self.model.eval()
        relationship, cot_head_dict = self.model.relationship_learning(self.loader, 4, cot_scale, self.coco_labels, self.novel_labels)
    
        self.model.init_cot_head(relationship, cot_lambda, cot_scale)
        self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

    def get_cot_head_dict(self):
        cot_head_dict = copy.deepcopy(self.model.yolo_head.state_dict())
        keys = list(cot_head_dict.keys())
        for k in keys:
            if 'yolo_output.0' in k:
                new_k = k.split('.')[-1]
                cot_head_dict.update({new_k:cot_head_dict.pop(k)})
            else:
                cot_head_dict.pop(k)
        return cot_head_dict

    def rebuild_model(self):
        # build model
        if 'model' not in self.cfg:
            self.model = create(self.cfg.architecture)
        else:
            self.model = self.cfg.model
        self.is_loaded_weights = False
