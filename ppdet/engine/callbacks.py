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
import copy
import json

import paddle
import paddle.distributed as dist

from ppdet.utils.checkpoint import save_model
from ppdet.metrics import get_infer_results

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = [
    'Callback', 'ComposeCallback', 'LogPrinter', 'Checkpointer',
    'VisualDLWriter', 'SniperProposalsGenerator'
]


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

    def on_train_begin(self, status):
        pass

    def on_train_end(self, status):
        pass


class ComposeCallback(object):
    def __init__(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(
                c, Callback), "callback should be subclass of Callback"
        self._callbacks = callbacks

    def on_step_begin(self, status):
        for c in self._callbacks:
            c.on_step_begin(status)

    def on_step_end(self, status):
        for c in self._callbacks:
            c.on_step_end(status)

    def on_epoch_begin(self, status):
        for c in self._callbacks:
            c.on_epoch_begin(status)

    def on_epoch_end(self, status):
        for c in self._callbacks:
            c.on_epoch_end(status)

    def on_train_begin(self, status):
        for c in self._callbacks:
            c.on_train_begin(status)

    def on_train_end(self, status):
        for c in self._callbacks:
            c.on_train_end(status)


class LogPrinter(Callback):
    def __init__(self, model):
        super(LogPrinter, self).__init__(model)

    def on_step_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
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
            if mode == 'eval':
                step_id = status['step_id']
                if step_id % 100 == 0:
                    logger.info("Eval iter: {}".format(step_id))

    def on_epoch_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            mode = status['mode']
            if mode == 'eval':
                sample_num = status['sample_num']
                cost_time = status['cost_time']
                logger.info('Total sample number: {}, averge FPS: {}'.format(
                    sample_num, sample_num / cost_time))


class Checkpointer(Callback):
    def __init__(self, model):
        super(Checkpointer, self).__init__(model)
        cfg = self.model.cfg
        self.best_ap = 0.
        self.save_dir = os.path.join(self.model.cfg.save_dir,
                                     self.model.cfg.filename)
        if hasattr(self.model.model, 'student_model'):
            self.weight = self.model.model.student_model
        else:
            self.weight = self.model.model

    def on_epoch_end(self, status):
        # Checkpointer only performed during training
        mode = status['mode']
        epoch_id = status['epoch_id']
        weight = None
        save_name = None
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if mode == 'train':
                end_epoch = self.model.cfg.epoch
                if (
                        epoch_id + 1
                ) % self.model.cfg.snapshot_epoch == 0 or epoch_id == end_epoch - 1:
                    save_name = str(
                        epoch_id) if epoch_id != end_epoch - 1 else "model_final"
                    weight = self.weight.state_dict()
            elif mode == 'eval':
                if 'save_best_model' in status and status['save_best_model']:
                    for metric in self.model._metrics:
                        map_res = metric.get_results()
                        if 'bbox' in map_res:
                            key = 'bbox'
                        elif 'keypoint' in map_res:
                            key = 'keypoint'
                        else:
                            key = 'mask'
                        if key not in map_res:
                            logger.warning("Evaluation results empty, this may be due to " \
                                        "training iterations being too few or not " \
                                        "loading the correct weights.")
                            return
                        if map_res[key][0] > self.best_ap:
                            self.best_ap = map_res[key][0]
                            save_name = 'best_model'
                            weight = self.weight.state_dict()
                        logger.info("Best test {} ap is {:0.3f}.".format(
                            key, self.best_ap))
            if weight:
                if self.model.use_ema:
                    # save model and ema_model
                    save_model(
                        status['weight'],
                        self.model.optimizer,
                        self.save_dir,
                        save_name,
                        epoch_id + 1,
                        ema_model=weight)
                else:
                    save_model(weight, self.model.optimizer, self.save_dir,
                               save_name, epoch_id + 1)


class WiferFaceEval(Callback):
    def __init__(self, model):
        super(WiferFaceEval, self).__init__(model)

    def on_epoch_begin(self, status):
        assert self.model.mode == 'eval', \
            "WiferFaceEval can only be set during evaluation"
        for metric in self.model._metrics:
            metric.update(self.model.model)
        sys.exit()


class VisualDLWriter(Callback):
    """
    Use VisualDL to log data or image
    """

    def __init__(self, model):
        super(VisualDLWriter, self).__init__(model)

        assert six.PY3, "VisualDL requires Python >= 3.5"
        try:
            from visualdl import LogWriter
        except Exception as e:
            logger.error('visualdl not found, plaese install visualdl. '
                         'for example: `pip install visualdl`.')
            raise e
        self.vdl_writer = LogWriter(
            model.cfg.get('vdl_log_dir', 'vdl_log_dir/scalar'))
        self.vdl_loss_step = 0
        self.vdl_mAP_step = 0
        self.vdl_image_step = 0
        self.vdl_image_frame = 0

    def on_step_end(self, status):
        mode = status['mode']
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if mode == 'train':
                training_staus = status['training_staus']
                for loss_name, loss_value in training_staus.get().items():
                    self.vdl_writer.add_scalar(loss_name, loss_value,
                                               self.vdl_loss_step)
                self.vdl_loss_step += 1
            elif mode == 'test':
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

    def on_epoch_end(self, status):
        mode = status['mode']
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if mode == 'eval':
                for metric in self.model._metrics:
                    for key, map_value in metric.get_results().items():
                        self.vdl_writer.add_scalar("{}-mAP".format(key),
                                                   map_value[0],
                                                   self.vdl_mAP_step)
                self.vdl_mAP_step += 1


class SniperProposalsGenerator(Callback):
    def __init__(self, model):
        super(SniperProposalsGenerator, self).__init__(model)
        ori_dataset = self.model.dataset
        self.dataset = self._create_new_dataset(ori_dataset)
        self.loader = self.model.loader
        self.cfg = self.model.cfg
        self.infer_model = self.model.model

    def _create_new_dataset(self, ori_dataset):
        dataset = copy.deepcopy(ori_dataset)
        # init anno_cropper
        dataset.init_anno_cropper()
        # generate infer roidbs
        ori_roidbs = dataset.get_ori_roidbs()
        roidbs = dataset.anno_cropper.crop_infer_anno_records(ori_roidbs)
        # set new roidbs
        dataset.set_roidbs(roidbs)

        return dataset

    def _eval_with_loader(self, loader):
        results = []
        with paddle.no_grad():
            self.infer_model.eval()
            for step_id, data in enumerate(loader):
                outs = self.infer_model(data)
                for key in ['im_shape', 'scale_factor', 'im_id']:
                    outs[key] = data[key]
                for key, value in outs.items():
                    if hasattr(value, 'numpy'):
                        outs[key] = value.numpy()

                results.append(outs)

        return results

    def on_train_end(self, status):
        self.loader.dataset = self.dataset
        results = self._eval_with_loader(self.loader)
        results = self.dataset.anno_cropper.aggregate_chips_detections(results)
        # sniper
        proposals = []
        clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()}
        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            start = 0
            for i, im_id in enumerate(outs['im_id']):
                bbox_num = outs['bbox_num']
                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                    if 'bbox' in batch_res else None
                if bbox_res:
                    proposals += bbox_res
        logger.info("save proposals in {}".format(self.cfg.proposals_path))
        with open(self.cfg.proposals_path, 'w') as f:
            json.dump(proposals, f)
