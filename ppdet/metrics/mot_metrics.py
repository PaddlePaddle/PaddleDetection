# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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
import paddle
import numpy as np
from scipy import interpolate
import paddle.nn.functional as F
from .map_utils import ap_per_class
from ppdet.modeling.bbox_utils import bbox_iou_np_expand
from .mot_eval_utils import MOTEvaluator
from .metrics import Metric

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['JDEDetMetric', 'JDEReIDMetric', 'MOTMetric']


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
            iou = bbox_iou_np_expand(pred_bbox, gt_boxes, x1y1x2y2=True)[0]
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

        # lazy import metrics here
        from sklearn import metrics
        far, tar, threshold = metrics.roc_curve(gt, pdist)
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
    def __init__(self, save_summary=False):
        self.save_summary = save_summary
        self.MOTEvaluator = MOTEvaluator
        self.result_root = None
        self.reset()

    def reset(self):
        self.accs = []
        self.seqs = []

    def update(self, data_root, seq, data_type, result_root, result_filename):
        evaluator = self.MOTEvaluator(data_root, seq, data_type)
        self.accs.append(evaluator.eval_file(result_filename))
        self.seqs.append(seq)
        self.result_root = result_root

    def accumulate(self):
        import motmetrics as mm
        import openpyxl
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = self.MOTEvaluator.get_summary(self.accs, self.seqs, metrics)
        self.strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names)
        if self.save_summary:
            self.MOTEvaluator.save_summary(
                summary, os.path.join(self.result_root, 'summary.xlsx'))

    def log(self):
        print(self.strsummary)

    def get_results(self):
        return self.strsummary
