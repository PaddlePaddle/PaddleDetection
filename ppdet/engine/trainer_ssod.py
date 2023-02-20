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
from paddle import amp
import os
import sys
import copy
import time
import typing
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from ppdet.optimizer import ModelEMA, SimpleModelEMA

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
import ppdet.utils.stats as stats
from ppdet.utils import profiler
from ppdet.modeling.ssod_utils import align_weak_strong_shape
from .trainer import Trainer

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['Trainer_DenseTeacher']


class Trainer_DenseTeacher(Trainer):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)

        # build data loader
        capital_mode = self.mode.capitalize()
        self.dataset = self.cfg['{}Dataset'.format(capital_mode)] = create(
            '{}Dataset'.format(capital_mode))()

        if self.mode == 'train':
            self.dataset_unlabel = self.cfg['UnsupTrainDataset'] = create(
                'UnsupTrainDataset')
            self.loader = create('SemiTrainReader')(
                self.dataset, self.dataset_unlabel, cfg.worker_num)

        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.SSOD)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            self._eval_batch_sampler = paddle.io.BatchSampler(
                self.dataset, batch_size=self.cfg.EvalReader['batch_size'])
            # If metric is VOC, need to be set collate_batch=False.
            if cfg.metric == 'VOC':
                cfg['EvalReader']['collate_batch'] = False
            self.loader = create('EvalReader')(self.dataset, cfg.worker_num,
                                               self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

            # Unstructured pruner is only enabled in the train mode.
            if self.cfg.get('unstructured_prune'):
                self.pruner = create('UnstructuredPruner')(self.model,
                                                           steps_per_epoch)
        if self.use_amp and self.amp_level == 'O2':
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.amp_level)

        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_black_list = self.cfg.get('ema_black_list', None)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list)
            self.ema_start_iters = self.cfg.get('ema_start_iters', 0)

        # simple_ema for SSOD
        self.use_simple_ema = ('use_simple_ema' in cfg and
                               cfg['use_simple_ema'])
        if self.use_simple_ema:
            self.use_ema = True
            ema_decay = self.cfg.get('ema_decay', 0.9996)
            self.ema = SimpleModelEMA(self.model, decay=ema_decay)
            self.ema_start_iters = self.cfg.get('ema_start_iters', 0)

        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()

        self.status = {}

        self.start_epoch = 0
        self.start_iter = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()




    def load_semi_weights(self, t_weights, s_weights):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model.teacher, t_weights)
        load_pretrain_weight(self.model.student, s_weights)
        logger.info("Load teacher weights {} to start training".format(t_weights))
        logger.info("Load student weights {} to start training".format(s_weights))


    def resume_weights(self, weights, exchange=True):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights,
                                           self.optimizer, exchange)
        else:
            self.start_iter,self.start_epoch = load_weight(self.model, weights, self.optimizer,
                                           self.ema
                                           if self.use_ema else None, exchange)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))
        logger.debug("Resume weights of iter {}".format(self.start_iter))

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False
        if validate:
            self.cfg.EvalDataset = create("EvalDataset")()

        model = self.model
        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and
                   self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            # self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
            #     self.model)
            model.teacher = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                model.teacher)
            model.student = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model.student)

        
        if self.cfg.get('fleet', False):
            # model = fleet.distributed_model(model)
            model = fleet.distributed_model(model)

            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            model = paddle.DataParallel(
                model, find_unused_parameters=find_unused_parameters)

        if self.cfg.get('amp', False):
            scaler = amp.GradScaler(
                enable=self.cfg.use_gpu or self.cfg.use_npu,
                init_loss_scaling=1024)

        self.status.update({
            'epoch_id': self.start_epoch,
            'iter_id': self.start_iter,
            # 'step_id': self.start_step,
            'steps_per_epoch': len(self.loader),
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)
        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)
        iter_id = self.start_iter
        self.status['iter_id'] = iter_id
        self.status['eval_interval'] = self.cfg.eval_interval
        self.status['save_interval'] = self.cfg.save_interval
        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset_label.set_epoch(epoch_id)
            self.loader.dataset_unlabel.set_epoch(epoch_id)
            iter_tic = time.time()
            if self._nranks > 1:
                # print(model)
                model._layers.teacher.eval()
                model._layers.student.train()
            else:
                model.teacher.eval()
                model.student.train()
            iter_tic = time.time()
            for step_id in range(len(self.loader)):
                data = next(self.loader)
                data_sup_w, data_sup_s, data_unsup_w, data_unsup_s = data
                data_sup_w['epoch_id'] = epoch_id
                data_sup_s['epoch_id'] = epoch_id
                data_unsup_w['epoch_id'] = epoch_id
                data_unsup_s['epoch_id'] = epoch_id
                data=[data_sup_w, data_sup_s, data_unsup_w, data_unsup_s] 
                iter_id += 1
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                self.status['iter_id'] = iter_id
                data.append(iter_id)
                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                if self.cfg.get('amp', False):
                    with amp.auto_cast(enable=self.cfg.use_gpu):
                        # model forward
                        if self._nranks > 1:
                            outputs = model._layers(data)
                        else:
                            outputs = model(data)
                        loss = outputs['loss']

                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    scaler.minimize(self.optimizer, scaled_loss)
                else:
                    outputs = model(data)                    
                    loss = outputs['loss']
                    # model backward
                    loss.backward()
                    self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                if self.cfg.get('unstructured_prune'):
                    self.pruner.step()
                self.optimizer.clear_grad()
                # print(outputs)
                # outputs=reduce_dict(outputs)
                # if self.model.debug:
                #     check_gradient(model)
                # self.check_gradient()
                self.status['learning_rate'] = curr_lr
                if self._nranks < 2 or self._local_rank == 0:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)

                if validate and (self._nranks < 2 or self._local_rank == 0) and \
                                ((iter_id + 1) % self.cfg.eval_interval == 0):                    
                    if not hasattr(self, '_eval_loader'):
                        # build evaluation dataset and loader
                        self._eval_dataset = self.cfg.EvalDataset
                        self._eval_batch_sampler = \
                            paddle.io.BatchSampler(
                                self._eval_dataset,
                                batch_size=self.cfg.EvalReader['batch_size'])
                        # If metric is VOC, need to be set collate_batch=False.
                        if self.cfg.metric == 'VOC':
                            self.cfg['EvalReader']['collate_batch'] = False
                        self._eval_loader = create('EvalReader')(
                            self._eval_dataset,
                            self.cfg.worker_num,
                            batch_sampler=self._eval_batch_sampler)
                    # if validation in training is enabled, metrics should be re-init
                    # Init_mark makes sure this code will only execute once
                    if validate and Init_mark == False:
                        Init_mark = True
                        self._init_metrics(validate=validate)
                        self._reset_metrics()

                    with paddle.no_grad():
                        self.status['save_best_model'] = True
                        self._eval_with_loader(self._eval_loader)
                    model._layers.student.train()

                self._compose_callback.on_step_end(self.status)

                iter_tic = time.time()
                    
            if self.cfg.get('unstructured_prune'):
                self.pruner.update_params()
            self._compose_callback.on_epoch_end(self.status)

        self._compose_callback.on_train_end(self.status)

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)
        print("*****teacher evaluate*****")
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            outs = self.model.teacher(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

        print("*****student evaluate*****")
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            outs = self.model.student(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        # reset metric states for metric may performed multiple times
        self._reset_metrics()
        self.status['mode'] = 'train'

    def evaluate(self):
        with paddle.no_grad():
            self._eval_with_loader(self.loader)

