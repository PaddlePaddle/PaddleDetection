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
import json
import paddle
import numpy as np

from .category import get_categories
from .map_utils import prune_zero_padding, DetectionMAP
from .coco_utils import get_infer_results, cocoapi_eval

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['Metric', 'COCOMetric', 'VOCMetric', 'get_infer_results']


class Metric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # abstract method for logging metric results
    def log(self):
        pass

    # abstract method for getting metric results
    def get_results(self):
        pass


class COCOMetric(Metric):
    def __init__(self, anno_file, with_background=True, mask_resolution=None):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.with_background = with_background
        self.mask_resolution = mask_resolution
        self.clsid2catid, self.catid2name = get_categories('COCO', anno_file,
                                                           with_background)

        self.reset()

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        # some input fields also needed
        for k in ['im_id', 'scale_factor', 'im_shape']:
            v = inputs[k]
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        if 'mask' in outs and 'bbox' in outs:
            from ppdet.py_op.post_process import mask_post_process
            outs['mask'] = mask_post_process(outs, outs['im_shape'],
                                             outs['scale_factor'],
                                             self.mask_resolution)

        infer_results = get_infer_results(outs, self.clsid2catid)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        self.results['mask'] += infer_results[
            'mask'] if 'mask' in infer_results else []
        self.results['segm'] += infer_results[
            'segm'] if 'segm' in infer_results else []

    def accumulate(self):
        if len(self.results['bbox']) > 0:
            with open("bbox.json", 'w') as f:
                json.dump(self.results['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            bbox_stats = cocoapi_eval(
                'bbox.json', 'bbox', anno_file=self.anno_file)
            self.eval_results['bbox'] = bbox_stats
            sys.stdout.flush()

        if len(self.results['mask']) > 0:
            with open("mask.json", 'w') as f:
                json.dump(self.results['mask'], f)
                logger.info('The mask result is saved to mask.json.')

            seg_stats = cocoapi_eval(
                'mask.json', 'segm', anno_file=self.anno_file)
            self.eval_results['mask'] = seg_stats
            sys.stdout.flush()

        if len(self.results['segm']) > 0:
            with open("segm.json", 'w') as f:
                json.dump(self.results['segm'], f)
                logger.info('The segm result is saved to segm.json.')

            seg_stats = cocoapi_eval(
                'segm.json', 'segm', anno_file=self.anno_file)
            self.eval_results['mask'] = seg_stats
            sys.stdout.flush()

    def log(self):
        pass

    def get_results(self):
        return self.eval_results


class VOCMetric(Metric):
    def __init__(self,
                 anno_file,
                 with_background=True,
                 class_num=20,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.with_background = with_background
        self.clsid2catid, self.catid2name = get_categories('VOC', anno_file,
                                                           with_background)

        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult)

        self.reset()

    def reset(self):
        self.detection_map.reset()

    def update(self, inputs, outputs):
        bboxes = outputs['bbox'].numpy()
        bbox_lengths = outputs['bbox_num'].numpy()

        if bboxes.shape == (1, 1) or bboxes is None:
            return
        gt_boxes = inputs['gt_bbox'].numpy()
        gt_labels = inputs['gt_class'].numpy()
        difficults = inputs['difficult'].numpy() if not self.evaluate_difficult \
                            else None

        scale_factor = inputs['scale_factor'].numpy(
        ) if 'scale_factor' in inputs else np.ones(
            (gt_boxes.shape[0], 2)).astype('float32')

        bbox_idx = 0
        for i in range(gt_boxes.shape[0]):
            gt_box = gt_boxes[i]
            h, w = scale_factor[i]
            gt_box = gt_box / np.array([w, h, w, h])
            gt_label = gt_labels[i]
            difficult = None if difficults is None \
                            else difficults[i]
            bbox_num = bbox_lengths[i]
            bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                             difficult)
            self.detection_map.update(bbox, gt_box, gt_label, difficult)
            bbox_idx += bbox_num

    def accumulate(self):
        logger.info("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                       self.map_type, map_stat))

    def get_results(self):
        self.detection_map.get_map()
