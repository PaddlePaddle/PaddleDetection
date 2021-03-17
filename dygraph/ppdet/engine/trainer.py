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
import time
import random
import datetime
import numpy as np
from PIL import Image
import glob

from IPython import embed
import paddle
from paddle.distributed import ParallelEnv, fleet
from paddle import amp
from paddle.static import InputSpec

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.visualizer import visualize_results
from ppdet.tracker.multitracker import JDETracker
from ppdet.tracking_utils.timer import Timer
from ppdet.tracking_utils import mot_visualization as mot_vis

from ppdet.metrics import Metric, COCOMetric, VOCMetric, JDEDetMetric, JDEReIDMetric, MOTMetric
from ppdet.metrics import get_categories, get_infer_results
import ppdet.utils.stats as stats

from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer
from .export_utils import _dump_infer_config

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test', 'track'], \
                "mode should be 'train', 'eval', 'test' or 'track'"
        self.mode = mode.lower()
        self.optimizer = None

        # model slim build
        if 'slim' in cfg and cfg.slim:
            if self.mode == 'train':
                self.load_weights(cfg.pretrain_weights, cfg.weight_type)
            slim = create(cfg.slim)
            slim(self.model)

        # build data loader
        self.dataset = cfg['{}Dataset'.format(self.mode.capitalize())]
        if self.mode == 'train':
            self.loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, cfg.worker_num)

        if cfg.architecture == 'JDE' and self.mode == 'train':
            cfg['JEDEmbeddingHead']['num_identifiers'] = self.dataset.nID

        # build model
        self.model = create(cfg.architecture)

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            self._eval_batch_sampler = paddle.io.BatchSampler(
                self.dataset, batch_size=self.cfg.EvalReader['batch_size'])
            self.loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, cfg.worker_num, self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr,
                                                        self.model.parameters())

        self._nranks = ParallelEnv().nranks
        self._local_rank = ParallelEnv().local_rank

        self.status = {}

        self.start_epoch = 0
        self.end_epoch = cfg.epoch

        self._weights_loaded = False

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def _init_callbacks(self):
        if self.mode == 'train':
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self):
        if self.mode == 'test':
            self._metrics = []
            return
        if self.cfg.metric == 'COCO':
            # TODO: bias should be unified
            bias = self.cfg['bias'] if 'bias' in self.cfg else 0
            self._metrics = [
                COCOMetric(
                    anno_file=self.dataset.get_anno(), bias=bias)
            ]
        elif self.cfg.metric == 'VOC':
            self._metrics = [
                VOCMetric(
                    anno_file=self.dataset.get_anno(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type)
            ]
        elif self.cfg.metric == 'MOTDet':
            self._metrics = [JDEDetMetric(), ]
        elif self.cfg.metric == 'ReID':
            self._metrics = [JDEReIDMetric(), ]
        elif self.cfg.metric == 'MOT':
            self._metrics = [MOTMetric(), ]
        else:
            logger.warn("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def register_callbacks(self, callbacks):
        callbacks = [h for h in list(callbacks) if h is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def register_metrics(self, metrics):
        metrics = [m for m in list(metrics) if m is not None]
        for m in metrics:
            assert isinstance(m, Metric), \
                    "metrics shoule be instances of subclass of Metric"
        self._metrics.extend(metrics)

    def load_weights(self, weights, weight_type='pretrain'):
        assert weight_type in ['pretrain', 'resume', 'finetune'], \
                "weight_type can only be 'pretrain', 'resume', 'finetune'"
        if weight_type == 'resume':
            self.start_epoch = load_weight(self.model, weights, self.optimizer)
            logger.debug("Resume weights of epoch {}".format(self.start_epoch))
        else:
            self.start_epoch = 0
            load_pretrain_weight(self.model, weights,
                                 self.cfg.get('load_static_weights', False),
                                 weight_type)
            logger.debug("Load {} weights {} to start training".format(
                weight_type, weights))
        self._weights_loaded = True

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"

        # if no given weights loaded, load backbone pretrain weights as default
        if not self._weights_loaded:
            self.load_weights(self.cfg.pretrain_weights)

        model = self.model
        if self.cfg.fleet:
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(
                self.optimizer).user_defined_optimizer
        elif self._nranks > 1:
            model = paddle.DataParallel(self.model)

        # initial fp16
        if self.cfg.fp16:
            scaler = amp.GradScaler(
                enable=self.cfg.use_gpu, init_loss_scaling=1024)

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                self._compose_callback.on_step_begin(self.status)

                if self.cfg.fp16:
                    with amp.auto_cast(enable=self.cfg.use_gpu):
                        # model forward
                        outputs = model(data)
                        loss = outputs['loss']

                    # model backward
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    # in dygraph mode, optimizer.minimize is equal to optimizer.step
                    scaler.minimize(self.optimizer, scaled_loss)
                else:
                    # model forward
                    outputs = model(data)
                    loss = outputs['loss']
                    # model backward
                    loss.backward()
                    self.optimizer.step()

                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                self.optimizer.clear_grad()
                self.status['learning_rate'] = curr_lr

                if self._nranks < 2 or self._local_rank == 0:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                iter_tic = time.time()

            self._compose_callback.on_epoch_end(self.status)

            if validate and (self._nranks < 2 or self._local_rank == 0) \
                    and (epoch_id % self.cfg.snapshot_epoch == 0 \
                             or epoch_id == self.end_epoch - 1):
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    self._eval_loader = create('EvalReader')(
                        self._eval_dataset,
                        self.cfg.worker_num,
                        batch_sampler=self._eval_batch_sampler)
                with paddle.no_grad():
                    self._eval_with_loader(self._eval_loader)

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            outs = self.model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic
        self._compose_callback.on_epoch_end(self.status)

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate(self):
        self._eval_with_loader(self.loader)

    def track(self,
              target_shape=[1088, 608],
              min_box_area=200,
              det_thresh=0.5,
              track_buffer=30,
              data_root='./dataset/MOT/MOT16/train',
              seqs=('MOT16-02', ),
              exp_name='demo',
              save_images=False,
              save_videos=False,
              show_image=False,
              save_dir=None):
        result_root = os.path.join(data_root, '..', 'results', exp_name)
        if save_dir:
            os.makedirs(result_root, exist_ok=True)
        data_type = 'mot'
        # run tracking
        accs = []
        n_frame = 0
        timer_avgs, timer_calls = [], []
        for seq in seqs:
            output_dir = os.path.join(
                data_root, '..', 'outputs', exp_name,
                seq) if save_images or save_videos else None
            logger.info('start seq: {}'.format(seq))
            infer_dir = os.path.join(data_root, seq, 'img1')
            images = self.get_tracking_images(infer_dir)

            self.dataset.set_images(images)
            dataloader = create('TestReader')(self.dataset, 0)  ### todo
            meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
            frame_rate = int(meta_info[meta_info.find('frameRate') + 10:
                                       meta_info.find('\nseqLength')])

            tracker = JDETracker(
                img_size=target_shape,
                det_thresh=det_thresh,
                frame_rate=frame_rate,
                track_buffer=track_buffer)
            timer = Timer()
            results = []
            frame_id = 0
            self.status['mode'] = 'track'
            self.model.eval()
            for step_id, data in enumerate(dataloader):
                self.status['step_id'] = step_id
                if frame_id % 20 == 0:
                    logger.info('Processing frame {} ({:.2f} fps)'.format(
                        frame_id, 1. / max(1e-5, timer.average_time)))

                # forward
                timer.tic()
                outs = self.model(data)
                pred_boxes = outs['bbox'][:, 2:].numpy()
                pred_scores = outs['bbox'][:, 1:2].numpy()
                pred_embs = outs['embedding'].numpy()
                img0_shape = outs['img0_shape'].numpy()[0]

                pred_dets = np.concatenate((pred_boxes, pred_scores), axis=1)
                online_targets = tracker.update(target_shape, pred_dets,
                                                pred_embs, img0_shape)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                timer.toc()

                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                if show_image or save_dir is not None:
                    online_im = mot_vis.plot_tracking(
                        img0,
                        online_tlwhs,
                        online_ids,
                        frame_id=frame_id,
                        fps=1. / timer.average_time)
                if show_image:
                    cv2.imshow('online_im', online_im)
                if save_dir is not None:
                    cv2.imwrite(
                        os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)),
                        online_im)
                frame_id += 1

            result_filename = os.path.join(result_root, '{}.txt'.format(seq))
            self.write_mot_results(result_filename, results, data_type)
            n_frame += frame_id
            timer_avgs.append(timer.average_time)
            timer_calls.append(timer.calls)
            if save_videos:
                output_video_path = os.path.join(output_dir,
                                                 '{}.mp4'.format(seq))
                cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
                    output_dir, output_video_path)
                os.system(cmd_str)

            logger.info('Evaluate seq: {}'.format(seq))
            # update metrics
            for metric in self._metrics:
                metric.update(data_root, seq, data_type, result_root,
                              result_filename, exp_name)

        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
            all_time, 1.0 / avg_time))

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def get_tracking_images(self, infer_dir):
        assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
        images = set()
        assert os.path.isdir(infer_dir), \
            "infer_dir {} is not a directory".format(infer_dir)
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
        images = list(images)
        images.sort()
        assert len(images) > 0, "no image found in {}".format(infer_dir)
        logger.info("Found {} inference images in total.".format(len(images)))
        return images

    def write_mot_results(self, filename, results, data_type='mot'):
        if data_type in ['mot', 'mcmot', 'lab']:
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(
                        frame=frame_id,
                        id=track_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        w=w,
                        h=h)
                    f.write(line)
        logger.info('MOT results save in {}'.format(filename))

    def predict(self, images, draw_threshold=0.5, output_dir='output'):
        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(self.cfg.metric, anno_file)

        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)
            for key in ['im_shape', 'scale_factor', 'im_id']:
                outs[key] = data[key]
            for key, value in outs.items():
                outs[key] = value.numpy()

            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']
            start = 0
            for i, im_id in enumerate(outs['im_id']):
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                end = start + bbox_num[i]

                bbox_res = batch_res['bbox'][start:end] \
                        if 'bbox' in batch_res else None
                mask_res = batch_res['mask'][start:end] \
                        if 'mask' in batch_res else None
                segm_res = batch_res['segm'][start:end] \
                        if 'segm' in batch_res else None
                image = visualize_results(image, bbox_res, mask_res, segm_res,
                                          int(outs['im_id']), catid2name,
                                          draw_threshold)

                # save image with detection
                save_name = self._get_save_image_name(output_dir, image_path)
                logger.info("Detection bbox results save in {}".format(
                    save_name))
                image.save(save_name, quality=95)
                start = end

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext

    def export(self, output_dir='output_inference'):
        self.model.eval()
        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_shape = None
        if 'inputs_def' in self.cfg['TestReader']:
            inputs_def = self.cfg['TestReader']['inputs_def']
            image_shape = inputs_def.get('image_shape', None)
        # set image_shape=[3, -1, -1] as default
        if image_shape is None:
            image_shape = [3, -1, -1]

        self.model.eval()

        # Save infer cfg
        _dump_infer_config(self.cfg,
                           os.path.join(save_dir, 'infer_cfg.yml'), image_shape,
                           self.model)

        input_spec = [{
            "image": InputSpec(
                shape=[None] + image_shape, name='image'),
            "im_shape": InputSpec(
                shape=[None, 2], name='im_shape'),
            "scale_factor": InputSpec(
                shape=[None, 2], name='scale_factor')
        }]

        # dy2st and save model
        static_model = paddle.jit.to_static(self.model, input_spec=input_spec)
        # NOTE: dy2st do not pruned program, but jit.save will prune program
        # input spec, prune input spec here and save with pruned input spec
        pruned_input_spec = self._prune_input_spec(
            input_spec, static_model.forward.main_program,
            static_model.forward.outputs)
        paddle.jit.save(
            static_model,
            os.path.join(save_dir, 'model'),
            input_spec=pruned_input_spec)
        logger.info("Export model and saved in {}".format(save_dir))

    def _prune_input_spec(self, input_spec, program, targets):
        # try to prune static program to figure out pruned input spec
        # so we perform following operations in static mode
        paddle.enable_static()
        pruned_input_spec = [{}]
        program = program.clone()
        program = program._prune(targets=targets)
        global_block = program.global_block()
        for name, spec in input_spec[0].items():
            try:
                v = global_block.var(name)
                pruned_input_spec[0][name] = spec
            except Exception:
                pass
        paddle.disable_static()
        return pruned_input_spec
