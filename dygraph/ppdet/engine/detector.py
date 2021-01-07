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

import paddle
from paddle.distributed import ParallelEnv
from paddle.static import InputSpec

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.eval_utils import get_infer_results, eval_results
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.stats as stats

from .hooks import HookBase, ComposeHook, LogPrinter, Checkpointer
from .export_utils import _dump_infer_config

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['Detector']


class Detector(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()

        # build model
        self.model = create(cfg.architecture)
        if ParallelEnv().nranks > 1:
            self.model = paddle.DataParallel(self.model)
        
        # build data loader
        self.dataset = cfg['{}Dataset'.format(self.mode.capitalize())]
        # TestDataset build after user set images, skip loader creation here
        if self.mode != 'test':
            self.loader = create('{}Reader'.format(self.mode.capitalize()))(self.dataset, cfg.worker_num)

        # build optimizer in train mode
        self.optimizer = None
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model.parameters())
        
        self.status = {}

        self.start_epoch = 0
        self.end_epoch = cfg.epoch

        self._weights_loaded = False

        # initial default hooks
        self._init_hooks()

    def _init_hooks(self):
        if self.mode == 'train':
            self._hooks = [LogPrinter(self), Checkpointer(self)]
            self._compose_hook = ComposeHook(self._hooks)
        if self.mode == 'eval':
            self._hooks = [LogPrinter(self)]
            self._compose_hook = ComposeHook(self._hooks)

    def load_weights(self, weights, weight_type='pretrain'):
        assert weight_type in ['pretrain', 'resume', 'finetune'], \
                "weight_type can only be 'pretrain', 'resume', 'finetune'"
        if weight_type == 'resume':
            self.start_epoch = load_weight(self.model, weights, self.optimizer)
            logger.debug("Resume weights of epoch {}".format(self.start_epoch))
        else:
            self.start_epoch = 0
            load_pretrain_weight(self.model, weights, self.cfg.get('load_static_weights', False), weight_type)
            logger.debug("Load {} weights {} to start training".format(weight_type, weights))
        self._weights_loaded = True

    def register_hooks(self, hooks):
        hooks = [h for h in list(hooks) if h is not None]
        self._hooks.extend(hooks)
        self._compose_hook = ComposeHook(self._hooks)

    def train(self):
        assert self.mode == 'train', "Model not in 'train' mode"

        # if no given weights loaded, load backbone pretrain weights as default
        if not self._weights_loaded:
            self.load_weights(self.cfg.pretrain_weights)

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)})
        
        self.status['batch_time'] = stats.SmoothedValue(self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['epoch_id'] = epoch_id
            self._compose_hook.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id 
                self._compose_hook.on_step_begin(self.status)

                # model forward
                self.model.train()
                outputs = self.model(data, mode='train')
                loss = outputs['loss']

                # model backward
                loss.backward()
                self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                self.optimizer.clear_grad()

                if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_hook.on_step_end(self.status)

            self._compose_hook.on_epoch_end(self.status)

    def evaluate(self):
        extra_key = ['im_shape', 'scale_factor', 'im_id']
        if self.cfg.metric == 'VOC':
            extra_key += ['gt_bbox', 'gt_class', 'difficult']

        # Run Eval
        outs_res = []
        sample_num = 0
        tic = time.time()
        self._compose_hook.on_epoch_begin(self.status)
        for step_id, data in enumerate(self.loader):
            self.status['step_id'] = step_id
            self._compose_hook.on_step_begin(self.status)
            # forward
            self.model.eval()
            outs = self.model(data, mode='infer')
            for key in extra_key:
                outs[key] = data[key]
            for key, value in outs.items():
                outs[key] = value.numpy()

            if 'mask' in outs and 'bbox' in outs:
                mask_resolution = self.model.mask_post_process.mask_resolution
                from ppdet.py_op.post_process import mask_post_process
                outs['mask'] = mask_post_process(
                    outs, outs['im_shape'], outs['scale_factor'], mask_resolution)

            outs_res.append(outs)
            sample_num += outs['im_id'].shape[0]
            self._compose_hook.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic
        self._compose_hook.on_epoch_end(self.status)

        eval_type = [t for t in ['bbox', 'mask'] if t in outs]

        # Metric
        # TODO: compate as ppdet.metric module
        with_background = self.cfg.with_background
        use_default_label = self.loader.dataset.use_default_label
        if self.cfg.metric == 'COCO':
            from ppdet.utils.coco_eval import get_category_info
            clsid2catid, catid2name = get_category_info(
                self.loader.dataset.get_anno(), with_background, use_default_label)

            infer_res = get_infer_results(outs_res, eval_type, clsid2catid)

        elif self.cfg.metric == 'VOC':
            from ppdet.utils.voc_eval import get_category_info
            clsid2catid, catid2name = get_category_info(
                self.loader.dataset.get_label_list(), with_background, use_default_label)
            infer_res = outs_res

        eval_results(infer_res, self.cfg.metric, self.loader.dataset)

    def predict(self, images, draw_threshold=0.5, output_dir='output'):
        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        extra_key = ['im_shape', 'scale_factor', 'im_id']

        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        with_background = self.cfg.with_background
        use_default_label = self.dataset.use_default_label

        if self.cfg.metric == 'COCO':
            from ppdet.utils.coco_eval import get_category_info
        elif self.cfg.metric == 'VOC':
            from ppdet.utils.voc_eval import get_category_info
        else:
            raise ValueError("unrecongnized metric type: {}".format(self.cfg.metric))
        clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                    use_default_label)

        # Run Infer 
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            # forward
            self.model.eval()
            outs = self.model(data, mode='infer')
            for key in extra_key:
                outs[key] = data[key]
            for key, value in outs.items():
                outs[key] = value.numpy()

            if 'mask' in outs and 'bbox' in outs:
                # FIXME: for more elegent coding
                mask_resolution = model.mask_post_process.mask_resolution
                from ppdet.py_op.post_process import mask_post_process
                outs['mask'] = mask_post_process(
                    outs, outs['im_shape'], outs['scale_factor'], mask_resolution)

            eval_type = [t for t in ['bbox', 'mask'] if t in outs]
            batch_res = get_infer_results([outs], eval_type, clsid2catid)
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
                image = visualize_results(image, bbox_res, mask_res,
                                          int(outs['im_id']), catid2name,
                                          draw_threshold)

                # save image with detection
                save_name = self._get_save_image_name(output_dir, image_path)
                logger.info("Detection bbox results save in {}".format(save_name))
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

    def export(self, output_dir='output'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_shape = None
        if 'inputs_def' in self.cfg['TestReader']:
            inputs_def = self.cfg['TestReader']['inputs_def']
            image_shape = inputs_def.get('image_shape', None)
        if image_shape is None:
            image_shape = [3, None, None]

        # Save infer cfg
        _dump_infer_config(self.cfg,
                           os.path.join(output_dir, 'infer_cfg.yml'), image_shape,
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
        paddle.jit.save(static_model, os.path.join(output_dir, 'model'))

