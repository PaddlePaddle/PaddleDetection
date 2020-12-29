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
import datetime

import paddle
from paddle.distributed import ParallelEnv

from ppdet.utils.eval_utils import get_infer_results, eval_results
from ppdet.core.workspace import create

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['eval_detector']


def eval_detector(model, loader, cfg):
    extra_key = ['im_shape', 'scale_factor', 'im_id']
    if cfg.metric == 'VOC':
        extra_key += ['gt_bbox', 'gt_class', 'difficult']

    # Run Eval
    outs_res = []
    sample_num = 0
    start_time = time.time()
    for iter_id, data in enumerate(loader):
        # forward
        model.eval()
        outs = model(data, mode='infer')
        for key in extra_key:
            outs[key] = data[key]
        for key, value in outs.items():
            outs[key] = value.numpy()

        if 'mask' in outs and 'bbox' in outs:
            mask_resolution = model.mask_post_process.mask_resolution
            from ppdet.py_op.post_process import mask_post_process
            outs['mask'] = mask_post_process(
                outs, outs['im_shape'], outs['scale_factor'], mask_resolution)

        outs_res.append(outs)
        # log
        sample_num += outs['im_id'].shape[0]
        if iter_id % 100 == 0:
            logger.info("Eval iter: {}".format(iter_id))

    cost_time = time.time() - start_time
    logger.info('Total sample number: {}, averge FPS: {}'.format(
        sample_num, sample_num / cost_time))

    eval_type = []
    if 'bbox' in outs:
        eval_type.append('bbox')
    if 'mask' in outs:
        eval_type.append('mask')
    # Metric
    # TODO: support other metric
    with_background = cfg.with_background
    use_default_label = loader.dataset.use_default_label
    if cfg.metric == 'COCO':
        from ppdet.utils.coco_eval import get_category_info
        clsid2catid, catid2name = get_category_info(
            loader.dataset.get_anno(), with_background, use_default_label)

        infer_res = get_infer_results(outs_res, eval_type, clsid2catid)

    elif cfg.metric == 'VOC':
        from ppdet.utils.voc_eval import get_category_info
        clsid2catid, catid2name = get_category_info(
            loader.dataset.get_label_list(), with_background, use_default_label)
        infer_res = outs_res

    eval_results(infer_res, cfg.metric, loader.dataset)
