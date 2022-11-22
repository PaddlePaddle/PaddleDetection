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
from __future__ import unicode_literals

import os
import sys
import numpy as np
import itertools
import paddle
from ppdet.modeling.rbox_utils import poly2rbox_np

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'draw_pr_curve',
    'bbox_area',
    'jaccard_overlap',
    'prune_zero_padding',
    'DetectionMAP',
    'ap_per_class',
    'compute_ap',
]


def draw_pr_curve(precision,
                  recall,
                  iou=0.5,
                  out_dir='pr_curve',
                  file_name='precision_recall_curve.jpg'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_path = os.path.join(out_dir, file_name)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error('Matplotlib not found, plaese install matplotlib.'
                     'for example: `pip install matplotlib`.')
        raise e
    plt.cla()
    plt.figure('P-R Curve')
    plt.title('Precision/Recall Curve(IoU={})'.format(iou))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.plot(recall, precision)
    plt.savefig(output_path)


def bbox_area(bbox, is_bbox_normalized):
    """
    Calculate area of a bounding box
    """
    norm = 1. - float(is_bbox_normalized)
    width = bbox[2] - bbox[0] + norm
    height = bbox[3] - bbox[1] + norm
    return width * height


def jaccard_overlap(pred, gt, is_bbox_normalized=False):
    """
    Calculate jaccard overlap ratio between two bounding box
    """
    if pred[0] >= gt[2] or pred[2] <= gt[0] or \
        pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.
    inter_xmin = max(pred[0], gt[0])
    inter_ymin = max(pred[1], gt[1])
    inter_xmax = min(pred[2], gt[2])
    inter_ymax = min(pred[3], gt[3])
    inter_size = bbox_area([inter_xmin, inter_ymin, inter_xmax, inter_ymax],
                           is_bbox_normalized)
    pred_size = bbox_area(pred, is_bbox_normalized)
    gt_size = bbox_area(gt, is_bbox_normalized)
    overlap = float(inter_size) / (pred_size + gt_size - inter_size)
    return overlap


def calc_rbox_iou(pred, gt_poly):
    """
    calc iou between rotated bbox
    """
    # calc iou of bounding box for speedup
    pred = np.array(pred, np.float32).reshape(-1, 2)
    gt_poly = np.array(gt_poly, np.float32).reshape(-1, 2)
    pred_rect = [
        np.min(pred[:, 0]), np.min(pred[:, 1]), np.max(pred[:, 0]),
        np.max(pred[:, 1])
    ]
    gt_rect = [
        np.min(gt_poly[:, 0]), np.min(gt_poly[:, 1]), np.max(gt_poly[:, 0]),
        np.max(gt_poly[:, 1])
    ]
    iou = jaccard_overlap(pred_rect, gt_rect, False)

    if iou <= 0:
        return iou

    # calc rbox iou
    pred_rbox = poly2rbox_np(pred.reshape(-1, 8)).reshape(-1, 5)
    gt_rbox = poly2rbox_np(gt_poly.reshape(-1, 8)).reshape(-1, 5)
    try:
        from ext_op import rbox_iou
    except Exception as e:
        print("import custom_ops error, try install ext_op " \
                  "following ppdet/ext_op/README.md", e)
        sys.stdout.flush()
        sys.exit(-1)
    pd_gt_rbox = paddle.to_tensor(gt_rbox, dtype='float32')
    pd_pred_rbox = paddle.to_tensor(pred_rbox, dtype='float32')
    iou = rbox_iou(pd_gt_rbox, pd_pred_rbox)
    iou = iou.numpy()
    return iou[0][0]


def prune_zero_padding(gt_box, gt_label, difficult=None):
    valid_cnt = 0
    for i in range(len(gt_box)):
        if (gt_box[i] == 0).all():
            break
        valid_cnt += 1
    return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
            if difficult is not None else None)


class DetectionMAP(object):
    """
    Calculate detection mean average precision.
    Currently support two types: 11point and integral

    Args:
        class_num (int): The class number.
        overlap_thresh (float): The threshold of overlap
            ratio between prediction bounding box and 
            ground truth bounding box for deciding 
            true/false positive. Default 0.5.
        map_type (str): Calculation method of mean average
            precision, currently support '11point' and
            'integral'. Default '11point'.
        is_bbox_normalized (bool): Whether bounding boxes
            is normalized to range[0, 1]. Default False.
        evaluate_difficult (bool): Whether to evaluate
            difficult bounding boxes. Default False.
        catid2name (dict): Mapping between category id and category name.
        classwise (bool): Whether per-category AP and draw
            P-R Curve or not.
    """

    def __init__(self,
                 class_num,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 catid2name=None,
                 classwise=False):
        self.class_num = class_num
        self.overlap_thresh = overlap_thresh
        assert map_type in ['11point', 'integral'], \
                "map_type currently only support '11point' "\
                "and 'integral'"
        self.map_type = map_type
        self.is_bbox_normalized = is_bbox_normalized
        self.evaluate_difficult = evaluate_difficult
        self.classwise = classwise
        self.classes = []
        for cname in catid2name.values():
            self.classes.append(cname)
        self.reset()

    def update(self, bbox, score, label, gt_box, gt_label, difficult=None):
        """
        Update metric statics from given prediction and ground
        truth infomations.
        """
        if difficult is None:
            difficult = np.zeros_like(gt_label)

        # record class gt count
        for gtl, diff in zip(gt_label, difficult):
            if self.evaluate_difficult or int(diff) == 0:
                self.class_gt_counts[int(np.array(gtl))] += 1

        # record class score positive
        visited = [False] * len(gt_label)
        for b, s, l in zip(bbox, score, label):
            pred = b.tolist() if isinstance(b, np.ndarray) else b
            max_idx = -1
            max_overlap = -1.0
            for i, gl in enumerate(gt_label):
                if int(gl) == int(l):
                    if len(gt_box[i]) == 8:
                        overlap = calc_rbox_iou(pred, gt_box[i])
                    else:
                        overlap = jaccard_overlap(pred, gt_box[i],
                                                  self.is_bbox_normalized)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = i

            if max_overlap > self.overlap_thresh:
                if self.evaluate_difficult or \
                        int(np.array(difficult[max_idx])) == 0:
                    if not visited[max_idx]:
                        self.class_score_poss[int(l)].append([s, 1.0])
                        visited[max_idx] = True
                    else:
                        self.class_score_poss[int(l)].append([s, 0.0])
            else:
                self.class_score_poss[int(l)].append([s, 0.0])

    def reset(self):
        """
        Reset metric statics
        """
        self.class_score_poss = [[] for _ in range(self.class_num)]
        self.class_gt_counts = [0] * self.class_num
        self.mAP = 0.0

    def accumulate(self):
        """
        Accumulate metric results and calculate mAP
        """
        mAP = 0.
        valid_cnt = 0
        eval_results = []
        for score_pos, count in zip(self.class_score_poss,
                                    self.class_gt_counts):
            if count == 0: continue
            if len(score_pos) == 0:
                valid_cnt += 1
                continue

            accum_tp_list, accum_fp_list = \
                    self._get_tp_fp_accum(score_pos)
            precision = []
            recall = []
            for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
                precision.append(float(ac_tp) / (ac_tp + ac_fp))
                recall.append(float(ac_tp) / count)

            one_class_ap = 0.0
            if self.map_type == '11point':
                max_precisions = [0.] * 11
                start_idx = len(precision) - 1
                for j in range(10, -1, -1):
                    for i in range(start_idx, -1, -1):
                        if recall[i] < float(j) / 10.:
                            start_idx = i
                            if j > 0:
                                max_precisions[j - 1] = max_precisions[j]
                                break
                        else:
                            if max_precisions[j] < precision[i]:
                                max_precisions[j] = precision[i]
                one_class_ap = sum(max_precisions) / 11.
                mAP += one_class_ap
                valid_cnt += 1
            elif self.map_type == 'integral':
                import math
                prev_recall = 0.
                for i in range(len(precision)):
                    recall_gap = math.fabs(recall[i] - prev_recall)
                    if recall_gap > 1e-6:
                        one_class_ap += precision[i] * recall_gap
                        prev_recall = recall[i]
                mAP += one_class_ap
                valid_cnt += 1
            else:
                logger.error("Unspported mAP type {}".format(self.map_type))
                sys.exit(1)
            eval_results.append({
                'class': self.classes[valid_cnt - 1],
                'ap': one_class_ap,
                'precision': precision,
                'recall': recall,
            })
        self.eval_results = eval_results
        self.mAP = mAP / float(valid_cnt) if valid_cnt > 0 else mAP

    def get_map(self):
        """
        Get mAP result
        """
        if self.mAP is None:
            logger.error("mAP is not calculated.")
        if self.classwise:
            # Compute per-category AP and PR curve
            try:
                from terminaltables import AsciiTable
            except Exception as e:
                logger.error(
                    'terminaltables not found, plaese install terminaltables. '
                    'for example: `pip install terminaltables`.')
                raise e
            results_per_category = []
            for eval_result in self.eval_results:
                results_per_category.append(
                    (str(eval_result['class']),
                     '{:0.3f}'.format(float(eval_result['ap']))))
                draw_pr_curve(
                    eval_result['precision'],
                    eval_result['recall'],
                    out_dir='voc_pr_curve',
                    file_name='{}_precision_recall_curve.jpg'.format(
                        eval_result['class']))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(* [
                results_flatten[i::num_columns] for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            logger.info('Per-category of VOC AP: \n{}'.format(table.table))
            logger.info(
                "per-category PR curve has output to voc_pr_curve folder.")
        return self.mAP

    def _get_tp_fp_accum(self, score_pos_list):
        """
        Calculate accumulating true/false positive results from
        [score, pos] records
        """
        sorted_list = sorted(score_pos_list, key=lambda s: s[0], reverse=True)
        accum_tp = 0
        accum_fp = 0
        accum_tp_list = []
        accum_fp_list = []
        for (score, pos) in sorted_list:
            accum_tp += int(pos)
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)
            accum_fp_list.append(accum_fp)
        return accum_tp_list, accum_fp_list


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Computes the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    
    Args:
        tp (list): True positives.
        conf (list): Objectness value from 0-1.
        pred_cls (list): Predicted object classes.
        target_cls (list): Target object classes.
    """
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(
        pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(
        p)


def compute_ap(recall, precision):
    """
    Computes the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    
    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
