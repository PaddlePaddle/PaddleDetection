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
from sklearn import metrics
from scipy import interpolate
import paddle.nn.functional as F
import motmetrics as mm
from .map_utils import ap_per_class, bbox_iou_expand
from .mot_eval_utils import MOTEvaluator

from .map_utils import prune_zero_padding, DetectionMAP
from .coco_utils import get_infer_results, cocoapi_eval
from .widerface_utils import face_eval_run
from ppdet.data.source.category import get_categories

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'Metric',
    'COCOMetric',
    'VOCMetric',
    'WiderFaceMetric',
    'get_infer_results',
    'JDEDetMetric',
    'JDEReIDMetric',
    'MOTMetric',
]


class Metric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # abstract method for logging metric results
    def log(self):
        pass

    # abstract method for getting metric results
    def get_results(self):
        pass


class COCOMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get('clsid2catid', None)
        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories('COCO', anno_file)
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
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

        im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        self.results['mask'] += infer_results[
            'mask'] if 'mask' in infer_results else []
        self.results['segm'] += infer_results[
            'segm'] if 'segm' in infer_results else []

    def accumulate(self):
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                logger.info('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                bbox_stats = cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

        if len(self.results['mask']) > 0:
            output = "mask.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['mask'], f)
                logger.info('The mask result is saved to mask.json.')

            if self.save_prediction_only:
                logger.info('The mask result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['segm']) > 0:
            output = "segm.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['segm'], f)
                logger.info('The segm result is saved to segm.json.')

            if self.save_prediction_only:
                logger.info('The segm result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

    def log(self):
        pass

    def get_results(self):
        return self.eval_results


class VOCMetric(Metric):
    def __init__(self,
                 label_list,
                 class_num=20,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False):
        assert os.path.isfile(label_list), \
                "label_list {} not a file".format(label_list)
        self.clsid2catid, self.catid2name = get_categories('VOC', label_list)

        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult,
            catid2name=self.catid2name,
            classwise=classwise)

        self.reset()

    def reset(self):
        self.detection_map.reset()

    def update(self, inputs, outputs):
        bboxes = outputs['bbox'][:, 2:].numpy()
        scores = outputs['bbox'][:, 1].numpy()
        labels = outputs['bbox'][:, 0].numpy()
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
            score = scores[bbox_idx:bbox_idx + bbox_num]
            label = labels[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                             difficult)
            self.detection_map.update(bbox, score, label, gt_box, gt_label,
                                      difficult)
            bbox_idx += bbox_num

    def accumulate(self):
        logger.info("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                       self.map_type, map_stat))

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}


class WiderFaceMetric(Metric):
    def __init__(self, image_dir, anno_file, multi_scale=True):
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.multi_scale = multi_scale
        self.clsid2catid, self.catid2name = get_categories('widerface')

    def update(self, model):

        face_eval_run(
            model,
            self.image_dir,
            self.anno_file,
            pred_dir='output/pred',
            eval_mode='widerface',
            multi_scale=self.multi_scale)


class JDEDetMetric(Metric):
    def __init__(self, overlap_thresh=0.5):
        self.overlap_thresh = overlap_thresh
        self.reset()

    def reset(self):
        self.AP_accum = np.zeros(1)
        self.AP_accum_count = np.zeros(1)

    def update(self, inputs, outputs):
        bboxes = outputs['bbox'][:, 2:].numpy()
        scores = outputs['bbox'][:, 1].numpy()
        labels = outputs['bbox'][:, 0].numpy()
        bbox_lengths = outputs['bbox_num'].numpy()
        if bboxes.shape[0] == 1 and bboxes.sum() == 0.0:
            return

        gt_boxes = inputs['gt_bbox'].numpy()[0]
        gt_labels = inputs['gt_class'].numpy()[0]
        if gt_labels.shape[0] == 0:
            return

        correct = []
        detected = []
        for i in range(bboxes.shape[0]):
            obj_pred = 0
            pred_bbox = bboxes[i].reshape(1, 4)
            # Compute iou with target boxes
            iou = bbox_iou_expand(pred_bbox, gt_boxes, x1y1x2y2=True)[0]
            # Extract index of largest overlap
            best_i = np.argmax(iou)
            # If overlap exceeds threshold and classification is correct mark as correct
            if iou[best_i] > self.overlap_thresh and obj_pred == gt_labels[
                    best_i] and best_i not in detected:
                correct.append(1)
                detected.append(best_i)
            else:
                correct.append(0)

        # Compute Average Precision (AP) per class
        target_cls = list(gt_labels.T[0])
        AP, AP_class, R, P = ap_per_class(
            tp=correct,
            conf=scores,
            pred_cls=np.zeros_like(scores),
            target_cls=target_cls)
        self.AP_accum_count += np.bincount(AP_class, minlength=1)
        self.AP_accum += np.bincount(AP_class, minlength=1, weights=AP)

    def accumulate(self):
        logger.info("Accumulating evaluatation results...")
        self.map_stat = self.AP_accum[0] / (self.AP_accum_count[0] + 1E-16)

    def log(self):
        map_stat = 100. * self.map_stat
        logger.info("mAP({:.2f}) = {:.2f}%".format(self.overlap_thresh,
                                                   map_stat))

    def get_results(self):
        return self.map_stat


class JDEReIDMetric(Metric):
    def __init__(self, far_levels=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
        self.far_levels = far_levels
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.embedding = []
        self.id_labels = []
        self.eval_results = {}

    def update(self, inputs, outputs):
        for out in outputs:
            feat, label = out[:-1].clone().detach(), int(out[-1])
            if label != -1:
                self.embedding.append(feat)
                self.id_labels.append(label)

    def accumulate(self):
        logger.info("Computing pairwise similairity...")
        assert len(self.embedding) == len(self.id_labels)
        if len(self.embedding) < 1:
            return None
        embedding = paddle.stack(self.embedding, axis=0)
        emb = F.normalize(embedding, axis=1).numpy()
        pdist = np.matmul(emb, emb.T)

        id_labels = np.array(self.id_labels, dtype='int32').reshape(-1, 1)
        n = len(id_labels)
        id_lbl = np.tile(id_labels, n).T
        gt = id_lbl == id_lbl.T

        up_triangle = np.where(np.triu(pdist) - np.eye(n) * pdist != 0)
        pdist = pdist[up_triangle]
        gt = gt[up_triangle]

        far, tar, threshold = self.metrics.roc_curve(gt, pdist)
        interp = interpolate.interp1d(far, tar)
        tar_at_far = [interp(x) for x in self.far_levels]

        for f, fa in enumerate(self.far_levels):
            self.eval_results['TPR@FAR={:.7f}'.format(fa)] = ' {:.4f}'.format(
                tar_at_far[f])

    def log(self):
        for k, v in self.eval_results.items():
            logger.info('{}: {}'.format(k, v))

    def get_results(self):
        return self.eval_results


class MOTMetric(Metric):
    def __init__(self, ):
        self.MOTEvaluator = MOTEvaluator
        self.reset()

    def reset(self):
        self.accs = []
        self.seqs = []

    def update(self, data_root, seq, data_type, result_root, result_filename):
        evaluator = self.MOTEvaluator(data_root, seq, data_type)
        self.accs.append(evaluator.eval_file(result_filename))
        self.seqs.append(seq)

    def accumulate(self):
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = self.MOTEvaluator.get_summary(self.accs, self.seqs, metrics)
        self.strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names)
        # self.MOTEvaluator.save_summary(summary, os.path.join(result_root, 'summary.xlsx'))

    def log(self):
        print(self.strsummary)

    def get_results(self):
        return self.strsummary
