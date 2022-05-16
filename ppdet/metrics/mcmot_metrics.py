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
import copy
import sys
import math
from collections import defaultdict
from motmetrics.math_util import quiet_divide

import numpy as np
import pandas as pd

from .metrics import Metric
import motmetrics as mm
import openpyxl
metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['MCMOTEvaluator', 'MCMOTMetric']

METRICS_LIST = [
    'num_frames', 'num_matches', 'num_switches', 'num_transfer', 'num_ascend',
    'num_migrate', 'num_false_positives', 'num_misses', 'num_detections',
    'num_objects', 'num_predictions', 'num_unique_objects', 'mostly_tracked',
    'partially_tracked', 'mostly_lost', 'num_fragmentations', 'motp', 'mota',
    'precision', 'recall', 'idfp', 'idfn', 'idtp', 'idp', 'idr', 'idf1'
]

NAME_MAP = {
    'num_frames': 'num_frames',
    'num_matches': 'num_matches',
    'num_switches': 'IDs',
    'num_transfer': 'IDt',
    'num_ascend': 'IDa',
    'num_migrate': 'IDm',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_detections': 'num_detections',
    'num_objects': 'num_objects',
    'num_predictions': 'num_predictions',
    'num_unique_objects': 'GT',
    'mostly_tracked': 'MT',
    'partially_tracked': 'partially_tracked',
    'mostly_lost': 'ML',
    'num_fragmentations': 'FM',
    'motp': 'MOTP',
    'mota': 'MOTA',
    'precision': 'Prcn',
    'recall': 'Rcll',
    'idfp': 'idfp',
    'idfn': 'idfn',
    'idtp': 'idtp',
    'idp': 'IDP',
    'idr': 'IDR',
    'idf1': 'IDF1'
}


def parse_accs_metrics(seq_acc, index_name, verbose=False):
    """
    Parse the evaluation indicators of multiple MOTAccumulator 
    """
    mh = mm.metrics.create()
    summary = MCMOTEvaluator.get_summary(seq_acc, index_name, METRICS_LIST)
    summary.loc['OVERALL', 'motp'] = (summary['motp'] * summary['num_detections']).sum() / \
                                     summary.loc['OVERALL', 'num_detections']
    if verbose:
        strsummary = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=NAME_MAP)
        print(strsummary)

    return summary


def seqs_overall_metrics(summary_df, verbose=False):
    """
    Calculate overall metrics for multiple sequences
    """
    add_col = [
        'num_frames', 'num_matches', 'num_switches', 'num_transfer',
        'num_ascend', 'num_migrate', 'num_false_positives', 'num_misses',
        'num_detections', 'num_objects', 'num_predictions',
        'num_unique_objects', 'mostly_tracked', 'partially_tracked',
        'mostly_lost', 'num_fragmentations', 'idfp', 'idfn', 'idtp'
    ]
    calc_col = ['motp', 'mota', 'precision', 'recall', 'idp', 'idr', 'idf1']
    calc_df = summary_df.copy()

    overall_dic = {}
    for col in add_col:
        overall_dic[col] = calc_df[col].sum()

    for col in calc_col:
        overall_dic[col] = getattr(MCMOTMetricOverall, col + '_overall')(
            calc_df, overall_dic)

    overall_df = pd.DataFrame(overall_dic, index=['overall_calc'])
    calc_df = pd.concat([calc_df, overall_df])

    if verbose:
        mh = mm.metrics.create()
        str_calc_df = mm.io.render_summary(
            calc_df, formatters=mh.formatters, namemap=NAME_MAP)
        print(str_calc_df)

    return calc_df


class MCMOTMetricOverall(object):
    def motp_overall(summary_df, overall_dic):
        motp = quiet_divide((summary_df['motp'] *
                             summary_df['num_detections']).sum(),
                            overall_dic['num_detections'])
        return motp

    def mota_overall(summary_df, overall_dic):
        del summary_df
        mota = 1. - quiet_divide(
            (overall_dic['num_misses'] + overall_dic['num_switches'] +
             overall_dic['num_false_positives']), overall_dic['num_objects'])
        return mota

    def precision_overall(summary_df, overall_dic):
        del summary_df
        precision = quiet_divide(overall_dic['num_detections'], (
            overall_dic['num_false_positives'] + overall_dic['num_detections']))
        return precision

    def recall_overall(summary_df, overall_dic):
        del summary_df
        recall = quiet_divide(overall_dic['num_detections'],
                              overall_dic['num_objects'])
        return recall

    def idp_overall(summary_df, overall_dic):
        del summary_df
        idp = quiet_divide(overall_dic['idtp'],
                           (overall_dic['idtp'] + overall_dic['idfp']))
        return idp

    def idr_overall(summary_df, overall_dic):
        del summary_df
        idr = quiet_divide(overall_dic['idtp'],
                           (overall_dic['idtp'] + overall_dic['idfn']))
        return idr

    def idf1_overall(summary_df, overall_dic):
        del summary_df
        idf1 = quiet_divide(2. * overall_dic['idtp'], (
            overall_dic['num_objects'] + overall_dic['num_predictions']))
        return idf1


def read_mcmot_results_union(filename, is_gt, is_ignore):
    results_dict = dict()
    if os.path.isfile(filename):
        all_result = np.loadtxt(filename, delimiter=',')
        if all_result.shape[0] == 0 or all_result.shape[1] < 7:
            return results_dict
        if is_ignore:
            return results_dict
        if is_gt:
            # only for test use
            all_result = all_result[all_result[:, 7] != 0]
            all_result[:, 7] = all_result[:, 7] - 1

        if all_result.shape[0] == 0:
            return results_dict

        class_unique = np.unique(all_result[:, 7])

        last_max_id = 0
        result_cls_list = []
        for cls in class_unique:
            result_cls_split = all_result[all_result[:, 7] == cls]
            result_cls_split[:, 1] = result_cls_split[:, 1] + last_max_id
            # make sure track id different between every category
            last_max_id = max(np.unique(result_cls_split[:, 1])) + 1
            result_cls_list.append(result_cls_split)

        results_con = np.concatenate(result_cls_list)

        for line in range(len(results_con)):
            linelist = results_con[line]
            fid = int(linelist[0])
            if fid < 1:
                continue
            results_dict.setdefault(fid, list())

            if is_gt:
                score = 1
            else:
                score = float(linelist[6])

            tlwh = tuple(map(float, linelist[2:6]))
            target_id = int(linelist[1])
            cls = int(linelist[7])

            results_dict[fid].append((tlwh, target_id, cls, score))

        return results_dict


def read_mcmot_results(filename, is_gt, is_ignore):
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.strip().split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                cid = int(linelist[7])
                if is_gt:
                    score = 1
                    # only for test use
                    cid -= 1
                else:
                    score = float(linelist[6])

                cls_result_dict = results_dict.setdefault(cid, dict())
                cls_result_dict.setdefault(fid, list())

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])
                cls_result_dict[fid].append((tlwh, target_id, score))
    return results_dict


def read_results(filename,
                 data_type,
                 is_gt=False,
                 is_ignore=False,
                 multi_class=False,
                 union=False):
    if data_type in ['mcmot', 'lab']:
        if multi_class:
            if union:
                # The results are evaluated by union all the categories.
                # Track IDs between different categories cannot be duplicate.
                read_fun = read_mcmot_results_union
            else:
                # The results are evaluated separately by category.
                read_fun = read_mcmot_results
        else:
            raise ValueError('multi_class: {}, MCMOT should have cls_id.'.
                             format(multi_class))
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores


def unzip_objs_cls(objs):
    if len(objs) > 0:
        tlwhs, ids, cls, scores = zip(*objs)
    else:
        tlwhs, ids, cls, scores = [], [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    ids = np.array(ids)
    cls = np.array(cls)
    scores = np.array(scores)
    return tlwhs, ids, cls, scores


class MCMOTEvaluator(object):
    def __init__(self, data_root, seq_name, data_type, num_classes):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.num_classes = num_classes

        self.load_annotations()
        self.reset_accumulator()

        self.class_accs = []

    def load_annotations(self):
        assert self.data_type == 'mcmot'
        self.gt_filename = os.path.join(self.data_root, '../', 'sequences',
                                        '{}.txt'.format(self.seq_name))
        if not os.path.exists(self.gt_filename):
            logger.warning(
                "gt_filename '{}' of MCMOTEvaluator is not exist, so the MOTA will be -INF."
            )

    def reset_accumulator(self):
        import motmetrics as mm
        mm.lap.default_solver = 'lap'
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame_dict(self, trk_objs, gt_objs, rtn_events=False, union=False):
        import motmetrics as mm
        mm.lap.default_solver = 'lap'
        if union:
            trk_tlwhs, trk_ids, trk_cls = unzip_objs_cls(trk_objs)[:3]
            gt_tlwhs, gt_ids, gt_cls = unzip_objs_cls(gt_objs)[:3]

            # get distance matrix
            iou_distance = mm.distances.iou_matrix(
                gt_tlwhs, trk_tlwhs, max_iou=0.5)

            # Set the distance between objects of different categories to nan
            gt_cls_len = len(gt_cls)
            trk_cls_len = len(trk_cls)
            # When the number of GT or Trk is 0, iou_distance dimension is (0,0)
            if gt_cls_len != 0 and trk_cls_len != 0:
                gt_cls = gt_cls.reshape(gt_cls_len, 1)
                gt_cls = np.repeat(gt_cls, trk_cls_len, axis=1)
                trk_cls = trk_cls.reshape(1, trk_cls_len)
                trk_cls = np.repeat(trk_cls, gt_cls_len, axis=0)
                iou_distance = np.where(gt_cls == trk_cls, iou_distance, np.nan)

        else:
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

            # get distance matrix
            iou_distance = mm.distances.iou_matrix(
                gt_tlwhs, trk_tlwhs, max_iou=0.5)

        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc,
                                                            'mot_events'):
            events = self.acc.mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, result_filename):
        # evaluation of each category
        gt_frame_dict = read_results(
            self.gt_filename,
            self.data_type,
            is_gt=True,
            multi_class=True,
            union=False)
        result_frame_dict = read_results(
            result_filename,
            self.data_type,
            is_gt=False,
            multi_class=True,
            union=False)

        for cid in range(self.num_classes):
            self.reset_accumulator()
            cls_result_frame_dict = result_frame_dict.setdefault(cid, dict())
            cls_gt_frame_dict = gt_frame_dict.setdefault(cid, dict())

            # only labeled frames will be evaluated
            frames = sorted(list(set(cls_gt_frame_dict.keys())))

            for frame_id in frames:
                trk_objs = cls_result_frame_dict.get(frame_id, [])
                gt_objs = cls_gt_frame_dict.get(frame_id, [])
                self.eval_frame_dict(trk_objs, gt_objs, rtn_events=False)

            self.class_accs.append(self.acc)

        return self.class_accs

    @staticmethod
    def get_summary(accs,
                    names,
                    metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1',
                             'precision', 'recall')):
        import motmetrics as mm
        mm.lap.default_solver = 'lap'

        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs, metrics=metrics, names=names, generate_overall=True)

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


class MCMOTMetric(Metric):
    def __init__(self, num_classes, save_summary=False):
        self.num_classes = num_classes
        self.save_summary = save_summary
        self.MCMOTEvaluator = MCMOTEvaluator
        self.result_root = None
        self.reset()

        self.seqs_overall = defaultdict(list)

    def reset(self):
        self.accs = []
        self.seqs = []

    def update(self, data_root, seq, data_type, result_root, result_filename):
        evaluator = self.MCMOTEvaluator(data_root, seq, data_type,
                                        self.num_classes)
        seq_acc = evaluator.eval_file(result_filename)
        self.accs.append(seq_acc)
        self.seqs.append(seq)
        self.result_root = result_root

        cls_index_name = [
            '{}_{}'.format(seq, i) for i in range(self.num_classes)
        ]
        summary = parse_accs_metrics(seq_acc, cls_index_name)
        summary.rename(
            index={'OVERALL': '{}_OVERALL'.format(seq)}, inplace=True)
        for row in range(len(summary)):
            self.seqs_overall[row].append(summary.iloc[row:row + 1])

    def accumulate(self):
        self.cls_summary_list = []
        for row in range(self.num_classes):
            seqs_cls_df = pd.concat(self.seqs_overall[row])
            seqs_cls_summary = seqs_overall_metrics(seqs_cls_df)
            cls_summary_overall = seqs_cls_summary.iloc[-1:].copy()
            cls_summary_overall.rename(
                index={'overall_calc': 'overall_calc_{}'.format(row)},
                inplace=True)
            self.cls_summary_list.append(cls_summary_overall)

    def log(self):
        seqs_summary = seqs_overall_metrics(
            pd.concat(self.seqs_overall[self.num_classes]), verbose=True)
        class_summary = seqs_overall_metrics(
            pd.concat(self.cls_summary_list), verbose=True)

    def get_results(self):
        return 1
