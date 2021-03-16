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
import datetime
import six
import numpy as np

import paddle
from paddle.distributed import ParallelEnv

from ppdet.utils.checkpoint import save_model
from ppdet.optimizer import ModelEMA

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['Callback', 'ComposeCallback', 'LogPrinter', 'Checkpointer']


class Callback(object):
    def __init__(self, model):
        self.model = model

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        pass

    def on_epoch_begin(self, status):
        pass

    def on_epoch_end(self, status):
        pass

    def on_eval_end(self, status):
        pass

    def on_predict_end(self, status):
        pass


class ComposeCallback(object):
    def __init__(self, callbacks):
        callbacks = [h for h in list(callbacks) if h is not None]
        for h in callbacks:
            assert isinstance(h,
                              Callback), "hook shoule be subclass of Callback"
        self._callbacks = callbacks

    def on_step_begin(self, status):
        for h in self._callbacks:
            h.on_step_begin(status)

    def on_step_end(self, status):
        for h in self._callbacks:
            h.on_step_end(status)

    def on_epoch_begin(self, status):
        for h in self._callbacks:
            h.on_epoch_begin(status)

    def on_epoch_end(self, status):
        for h in self._callbacks:
            h.on_epoch_end(status)

    def on_eval_end(self, status):
        for h in self._callbacks:
            h.on_eval_end(status)

    def on_predict_end(self, status):
        for h in self._callbacks:
            h.on_predict_end(status)


class LogPrinter(Callback):
    def __init__(self, model):
        super(LogPrinter, self).__init__(model)
        # use VisualDL to log data or image
        if model.cfg.use_vdl:
            assert six.PY3, "VisualDL requires Python >= 3.5"
            try:
                from visualdl import LogWriter
            except Exception as e:
                logger.error('visualdl not found, plaese install visualdl. '
                             'for example: `pip install visualdl`.')
                raise e
            self.vdl_writer = LogWriter(model.cfg.vdl_log_dir)
            self.vdl_loss_step = 0
            self.vdl_mAP_step = 0
            self.vdl_image_step = 0
            self.vdl_image_frame = 0

    def on_step_end(self, status):
        if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
            mode = status['mode']
            if mode == 'train':
                epoch_id = status['epoch_id']
                step_id = status['step_id']
                steps_per_epoch = status['steps_per_epoch']
                training_staus = status['training_staus']
                batch_time = status['batch_time']
                data_time = status['data_time']

                epoches = self.model.cfg.epoch
                batch_size = self.model.cfg['{}Reader'.format(mode.capitalize(
                ))]['batch_size']

                logs = training_staus.log()
                space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
                if step_id % self.model.cfg.log_iter == 0:
                    eta_steps = (epoches - epoch_id) * steps_per_epoch - step_id
                    eta_sec = eta_steps * batch_time.global_avg
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                    ips = float(batch_size) / batch_time.avg
                    fmt = ' '.join([
                        'Epoch: [{}]',
                        '[{' + space_fmt + '}/{}]',
                        'learning_rate: {lr:.6f}',
                        '{meters}',
                        'eta: {eta}',
                        'batch_cost: {btime}',
                        'data_cost: {dtime}',
                        'ips: {ips:.4f} images/s',
                    ])
                    fmt = fmt.format(
                        epoch_id,
                        step_id,
                        steps_per_epoch,
                        lr=status['learning_rate'],
                        meters=logs,
                        eta=eta_str,
                        btime=str(batch_time),
                        dtime=str(data_time),
                        ips=ips)
                    logger.info(fmt)
                    if self.model.cfg.use_vdl:
                        for loss_name, loss_value in training_staus.get().items(
                        ):
                            self.vdl_writer.add_scalar(loss_name, loss_value,
                                                       self.vdl_loss_step)
                        self.vdl_loss_step += 1
            if mode == 'eval':
                step_id = status['step_id']
                if step_id % 100 == 0:
                    logger.info("Eval iter: {}".format(step_id))

    def on_epoch_end(self, status):
        if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
            mode = status['mode']
            if mode == 'eval':
                sample_num = status['sample_num']
                cost_time = status['cost_time']
                logger.info('Total sample number: {}, averge FPS: {}'.format(
                    sample_num, sample_num / cost_time))

    def on_eval_end(self, status):
        if status['mode'] != 'eval':
            return
        if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
            if self.model.cfg.use_vdl:
                for metric in self.model._metrics:
                    for key, map_value in metric.get_results().items():
                        self.vdl_writer.add_scalar("{}-mAP".format(key),
                                                   map_value[0],
                                                   self.vdl_mAP_step)
                self.vdl_mAP_step += 1

    def on_predict_end(self, status):
        if status['mode'] != 'test':
            return
        if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
            if self.model.cfg.use_vdl:
                ori_image = status['original_image']
                result_image = status['result_image']
                self.vdl_writer.add_image(
                    "original/frame_{}".format(self.vdl_image_frame), ori_image,
                    self.vdl_image_step)
                self.vdl_writer.add_image(
                    "result/frame_{}".format(self.vdl_image_frame),
                    result_image, self.vdl_image_step)
                self.vdl_image_step += 1
                # each frame can display ten pictures at most.
                if self.vdl_image_step % 10 == 0:
                    self.vdl_image_step = 0
                    self.vdl_image_frame += 1


class Checkpointer(Callback):
    def __init__(self, model):
        super(Checkpointer, self).__init__(model)
        cfg = self.model.cfg
        self.best_ap = 0.
        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        self.save_dir = os.path.join(self.model.cfg.save_dir,
                                     self.model.cfg.filename)
        if self.use_ema:
            self.ema = ModelEMA(
                cfg['ema_decay'], self.model.model, use_thres_step=True)

    def on_step_end(self, status):
        if self.use_ema:
            self.ema.update(self.model.model)

    def on_epoch_end(self, status):
        # Checkpointer only performed during training
        mode = status['mode']
        if mode != 'train':
            return

        if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
            epoch_id = status['epoch_id']
            end_epoch = self.model.cfg.epoch
            if epoch_id % self.model.cfg.snapshot_epoch == 0 or epoch_id == end_epoch - 1:
                save_name = str(
                    epoch_id) if epoch_id != end_epoch - 1 else "model_final"
                if self.use_ema:
                    state_dict = self.ema.apply()
                    save_model(state_dict, self.model.optimizer, self.save_dir,
                               save_name, epoch_id + 1)
                else:
                    save_model(self.model.model, self.model.optimizer,
                               self.save_dir, save_name, epoch_id + 1)

    def on_eval_end(self, status):
        if status['mode'] != 'eval':
            return
        if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
            if 'save_best_model' in status and status['save_best_model']:
                for metric in self.model._metrics:
                    map_res = metric.get_results()
                    key = 'bbox' if 'bbox' in map_res else 'mask'
                    if map_res[key][0] > self.best_ap:
                        self.best_ap = map_res[key][0]
                        epoch_id = status['epoch_id']
                        if self.use_ema:
                            state_dict = self.ema.apply()
                            save_model(state_dict, self.model.optimizer,
                                       self.save_dir, 'best_model',
                                       epoch_id + 1)
                        else:
                            save_model(self.model.model, self.model.optimizer,
                                       self.save_dir, 'best_model',
                                       epoch_id + 1)
                    logger.info("Best test {} ap is {:0.3f}.".format(
                        key, self.best_ap))


class WiferFaceEval(Callback):
    def __init__(self, model):
        super(WiferFaceEval, self).__init__(model)

    def on_epoch_begin(self, status):
        assert self.model.mode == 'eval', \
            "WiferFaceEval can only be set during evaluation"
        for metric in self.model._metrics:
            metric.update(self.model.model)
        sys.exit()
