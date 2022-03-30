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
import numpy as np

from ppdet.modeling.bbox_utils import bbox_iou_np_expand
from .map_utils import ap_per_class
from .metrics import Metric
from .munkres import Munkres

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['MOTEvaluator', 'MOTMetric', 'JDEDetMetric', 'KITTIMOTMetric']


def read_mot_results(filename, is_gt=False, is_ignore=False):
    valid_label = [1]
    ignore_labels = [2, 7, 8, 12]  # only in motchallenge datasets like 'MOT16'
    if is_gt:
        logger.info(
            "In MOT16/17 dataset the valid_label of ground truth is '{}', "
            "in other dataset it should be '0' for single classs MOT.".format(
                valid_label[0]))
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
                    label = int(float(linelist[7]))
                    mark = int(float(linelist[6]))
                    if mark == 0 or label not in valid_label:
                        continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename or 'MOT15-' in filename or 'MOT20-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))
    return results_dict


"""
MOT dataset label list, see in https://motchallenge.net
labels={'ped', ...			    % 1
        'person_on_vhcl', ...	% 2
        'car', ...				% 3
        'bicycle', ...			% 4
        'mbike', ...			% 5
        'non_mot_vhcl', ...		% 6
        'static_person', ...	% 7
        'distractor', ...		% 8
        'occluder', ...			% 9
        'occluder_on_grnd', ...	% 10
        'occluder_full', ...	% 11
        'reflection', ...		% 12
        'crowd' ...			    % 13
};
"""


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores


class MOTEvaluator(object):
    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'
        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt',
                                   'gt.txt')
        if not os.path.exists(gt_filename):
            logger.warning(
                "gt_filename '{}' of MOTEvaluator is not exist, so the MOTA will be -INF."
            )
        self.gt_frame_dict = read_mot_results(gt_filename, is_gt=True)
        self.gt_ignore_frame_dict = read_mot_results(
            gt_filename, is_ignore=True)

    def reset_accumulator(self):
        import motmetrics as mm
        mm.lap.default_solver = 'lap'
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        import motmetrics as mm
        mm.lap.default_solver = 'lap'
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(
            ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc,
                                                            'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_mot_results(filename, is_gt=False)
        frames = sorted(list(set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

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


class JDEDetMetric(Metric):
    # Note this detection AP metric is different from COCOMetric or VOCMetric,
    # and the bboxes coordinates are not scaled to the original image
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


"""
Following code is borrow from https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/eval_kitti_track/evaluate_tracking.py
"""


class tData:
    """
        Utility class to load data.
    """
    def __init__(self,frame=-1,obj_type="unset",truncation=-1,occlusion=-1,\
                 obs_angle=-10,x1=-1,y1=-1,x2=-1,y2=-1,w=-1,h=-1,l=-1,\
                 X=-1000,Y=-1000,Z=-1000,yaw=-10,score=-1000,track_id=-1):
        """
            Constructor, initializes the object given the parameters.
        """
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.tracker = -1

    def __str__(self):
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


class KITTIEvaluation(object):
    """ KITTI tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	- Multi-object tracking accuracy in [0,100]
             MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
             MOTAL	- Multi-object tracking accuracy in [0,100] with log10(id-switches)

             id-switches - number of id switches
             fragments   - number of fragmentations

             MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories

             recall	        - recall = percentage of detected targets
             precision	    - precision = percentage of correctly detected targets
             FAR		    - number of false alarms per frame
             falsepositives - number of false positives (FP)
             missed         - number of missed targets (FN)
    """
    def __init__(self, result_path, gt_path, min_overlap=0.5, max_truncation = 0,\
                min_height = 25, max_occlusion = 2, cls="car",\
                n_frames=[], seqs=[], n_sequences=0):
        # get number of sequences and
        # get number of frames per sequence from test mapping
        # (created while extracting the benchmark)
        self.gt_path = os.path.join(gt_path, "../labels")
        self.n_frames = n_frames
        self.sequence_name = seqs
        self.n_sequences = n_sequences

        self.cls = cls  # class to evaluate, i.e. pedestrian or car

        self.result_path = result_path

        # statistics and numbers for evaluation
        self.n_gt = 0  # number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # number of ignored ground truth detections
        self.n_gts = [
        ]  # number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_igts = [
        ]  # number of ground ignored truth detections PER SEQUENCE
        self.n_gt_trajectories = 0
        self.n_gt_seq = []
        self.n_tr = 0  # number of tracker detections minus ignored tracker detections
        self.n_trs = [
        ]  # number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_itr = 0  # number of ignored tracker detections
        self.n_itrs = []  # number of ignored tracker detections PER SEQUENCE
        self.n_igttr = 0  # number of ignored ground truth detections where the corresponding associated tracker detection is also ignored
        self.n_tr_trajectories = 0
        self.n_tr_seq = []
        self.MOTA = 0
        self.MOTP = 0
        self.MOTAL = 0
        self.MODA = 0
        self.MODP = 0
        self.MODP_t = []
        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.FAR = 0
        self.total_cost = 0
        self.itp = 0  # number of ignored true positives
        self.itps = []  # number of ignored true positives PER SEQUENCE
        self.tp = 0  # number of true positives including ignored true positives!
        self.tps = [
        ]  # number of true positives including ignored true positives PER SEQUENCE
        self.fn = 0  # number of false negatives WITHOUT ignored false negatives
        self.fns = [
        ]  # number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self.ifn = 0  # number of ignored false negatives
        self.ifns = []  # number of ignored false negatives PER SEQUENCE
        self.fp = 0  # number of false positives
        # a bit tricky, the number of ignored false negatives and ignored true positives 
        # is subtracted, but if both tracker detection and ground truth detection
        # are ignored this number is added again to avoid double counting
        self.fps = []  # above PER SEQUENCE
        self.mme = 0
        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.PT = 0
        self.ML = 0

        self.min_overlap = min_overlap  # minimum bounding box overlap for 3rd party metrics
        self.max_truncation = max_truncation  # maximum truncation of an object for evaluation
        self.max_occlusion = max_occlusion  # maximum occlusion of an object for evaluation
        self.min_height = min_height  # minimum height of an object for evaluation
        self.n_sample_points = 500

        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories = [[] for x in range(self.n_sequences)]
        self.ign_trajectories = [[] for x in range(self.n_sequences)]

    def loadGroundtruth(self):
        try:
            self._loadData(self.gt_path, cls=self.cls, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def loadTracker(self):
        try:
            if not self._loadData(
                    self.result_path, cls=self.cls, loading_groundtruth=False):
                return False
        except IOError:
            return False
        return True

    def _loadData(self,
                  root_dir,
                  cls,
                  min_score=-1000,
                  loading_groundtruth=False):
        """
            Generic loader for ground truth and tracking data.
            Use loadGroundtruth() or loadTracker() to load this data.
            Loads detections in KITTI format from textfiles.
        """
        # construct objectDetections object to hold detection data
        t_data = tData()
        data = []
        eval_2d = True
        eval_3d = True

        seq_data = []
        n_trajectories = 0
        n_trajectories_seq = []
        for seq, s_name in enumerate(self.sequence_name):
            i = 0
            filename = os.path.join(root_dir, "%s.txt" % s_name)
            f = open(filename, "r")

            f_data = [
                [] for x in range(self.n_frames[seq])
            ]  # current set has only 1059 entries, sufficient length is checked anyway
            ids = []
            n_in_seq = 0
            id_frame_cache = []
            for line in f:
                # KITTI tracking benchmark data format:
                # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
                line = line.strip()
                fields = line.split(" ")
                # classes that should be loaded (ignored neighboring classes)
                if "car" in cls.lower():
                    classes = ["car", "van"]
                elif "pedestrian" in cls.lower():
                    classes = ["pedestrian", "person_sitting"]
                else:
                    classes = [cls.lower()]
                classes += ["dontcare"]
                if not any([s for s in classes if s in fields[2].lower()]):
                    continue
                # get fields from table
                t_data.frame = int(float(fields[0]))  # frame
                t_data.track_id = int(float(fields[1]))  # id
                t_data.obj_type = fields[
                    2].lower()  # object type [car, pedestrian, cyclist, ...]
                t_data.truncation = int(
                    float(fields[3]))  # truncation [-1,0,1,2]
                t_data.occlusion = int(
                    float(fields[4]))  # occlusion  [-1,0,1,2]
                t_data.obs_angle = float(fields[5])  # observation angle [rad]
                t_data.x1 = float(fields[6])  # left   [px]
                t_data.y1 = float(fields[7])  # top    [px]
                t_data.x2 = float(fields[8])  # right  [px]
                t_data.y2 = float(fields[9])  # bottom [px]
                t_data.h = float(fields[10])  # height [m]
                t_data.w = float(fields[11])  # width  [m]
                t_data.l = float(fields[12])  # length [m]
                t_data.X = float(fields[13])  # X [m]
                t_data.Y = float(fields[14])  # Y [m]
                t_data.Z = float(fields[15])  # Z [m]
                t_data.yaw = float(fields[16])  # yaw angle [rad]
                if not loading_groundtruth:
                    if len(fields) == 17:
                        t_data.score = -1
                    elif len(fields) == 18:
                        t_data.score = float(fields[17])  # detection score
                    else:
                        logger.info("file is not in KITTI format")
                        return

                # do not consider objects marked as invalid
                if t_data.track_id is -1 and t_data.obj_type != "dontcare":
                    continue

                idx = t_data.frame
                # check if length for frame data is sufficient
                if idx >= len(f_data):
                    print("extend f_data", idx, len(f_data))
                    f_data += [[] for x in range(max(500, idx - len(f_data)))]
                try:
                    id_frame = (t_data.frame, t_data.track_id)
                    if id_frame in id_frame_cache and not loading_groundtruth:
                        logger.info(
                            "track ids are not unique for sequence %d: frame %d"
                            % (seq, t_data.frame))
                        logger.info(
                            "track id %d occured at least twice for this frame"
                            % t_data.track_id)
                        logger.info("Exiting...")
                        #continue # this allows to evaluate non-unique result files
                        return False
                    id_frame_cache.append(id_frame)
                    f_data[t_data.frame].append(copy.copy(t_data))
                except:
                    print(len(f_data), idx)
                    raise

                if t_data.track_id not in ids and t_data.obj_type != "dontcare":
                    ids.append(t_data.track_id)
                    n_trajectories += 1
                    n_in_seq += 1

                # check if uploaded data provides information for 2D and 3D evaluation
                if not loading_groundtruth and eval_2d is True and (
                        t_data.x1 == -1 or t_data.x2 == -1 or t_data.y1 == -1 or
                        t_data.y2 == -1):
                    eval_2d = False
                if not loading_groundtruth and eval_3d is True and (
                        t_data.X == -1000 or t_data.Y == -1000 or
                        t_data.Z == -1000):
                    eval_3d = False

            # only add existing frames
            n_trajectories_seq.append(n_in_seq)
            seq_data.append(f_data)
            f.close()

        if not loading_groundtruth:
            self.tracker = seq_data
            self.n_tr_trajectories = n_trajectories
            self.eval_2d = eval_2d
            self.eval_3d = eval_3d
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories == 0:
                return False
        else:
            # split ground truth and DontCare areas
            self.dcareas = []
            self.groundtruth = []
            for seq_idx in range(len(seq_data)):
                seq_gt = seq_data[seq_idx]
                s_g, s_dc = [], []
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g, dc = [], []
                    for gg in all_gt:
                        if gg.obj_type == "dontcare":
                            dc.append(gg)
                        else:
                            g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas.append(s_dc)
                self.groundtruth.append(s_g)
            self.n_gt_seq = n_trajectories_seq
            self.n_gt_trajectories = n_trajectories
        return True

    def boxoverlap(self, a, b, criterion="union"):
        """
            boxoverlap computes intersection over union for bbox a and b in KITTI format.
            If the criterion is 'union', overlap = (a inter b) / a union b).
            If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
        """
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        w = x2 - x1
        h = y2 - y1

        if w <= 0. or h <= 0.:
            return 0.
        inter = w * h
        aarea = (a.x2 - a.x1) * (a.y2 - a.y1)
        barea = (b.x2 - b.x1) * (b.y2 - b.y1)
        # intersection over union overlap
        if criterion.lower() == "union":
            o = inter / float(aarea + barea - inter)
        elif criterion.lower() == "a":
            o = float(inter) / float(aarea)
        else:
            raise TypeError("Unkown type for criterion")
        return o

    def compute3rdPartyMetrics(self):
        """
            Computes the metrics defined in
                - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
                  MOTA, MOTAL, MOTP
                - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
                  MT/PT/ML
        """
        # construct Munkres object for Hungarian Method association
        hm = Munkres()
        max_cost = 1e9

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0, 0
        for seq_idx in range(len(self.groundtruth)):
            seq_gt = self.groundtruth[seq_idx]
            seq_dc = self.dcareas[seq_idx]  # don't care areas
            seq_tracker = self.tracker[seq_idx]
            seq_trajectories = defaultdict(list)
            seq_ignored = defaultdict(list)

            # statistics over the current sequence, check the corresponding
            # variable comments in __init__ to get their meaning
            seqtp = 0
            seqitp = 0
            seqfn = 0
            seqifn = 0
            seqfp = 0
            seqigt = 0
            seqitr = 0

            last_ids = [[], []]
            n_gts = 0
            n_trs = 0

            for f in range(len(seq_gt)):
                g = seq_gt[f]
                dc = seq_dc[f]

                t = seq_tracker[f]
                # counting total number of ground truth and tracker objects
                self.n_gt += len(g)
                self.n_tr += len(t)

                n_gts += len(g)
                n_trs += len(t)

                # use hungarian method to associate, using boxoverlap 0..1 as cost
                # build cost matrix
                cost_matrix = []
                this_ids = [[], []]
                for gg in g:
                    # save current ids
                    this_ids[0].append(gg.track_id)
                    this_ids[1].append(-1)
                    gg.tracker = -1
                    gg.id_switch = 0
                    gg.fragmentation = 0
                    cost_row = []
                    for tt in t:
                        # overlap == 1 is cost ==0
                        c = 1 - self.boxoverlap(gg, tt)
                        # gating for boxoverlap
                        if c <= self.min_overlap:
                            cost_row.append(c)
                        else:
                            cost_row.append(max_cost)  # = 1e9
                    cost_matrix.append(cost_row)
                    # all ground truth trajectories are initially not associated
                    # extend groundtruth trajectories lists (merge lists)
                    seq_trajectories[gg.track_id].append(-1)
                    seq_ignored[gg.track_id].append(False)

                if len(g) is 0:
                    cost_matrix = [[]]
                # associate
                association_matrix = hm.compute(cost_matrix)

                # tmp variables for sanity checks and MODP computation
                tmptp = 0
                tmpfp = 0
                tmpfn = 0
                tmpc = 0  # this will sum up the overlaps for all true positives
                tmpcs = [0] * len(
                    g)  # this will save the overlaps for all true positives
                # the reason is that some true positives might be ignored
                # later such that the corrsponding overlaps can
                # be subtracted from tmpc for MODP computation

                # mapping for tracker ids and ground truth ids
                for row, col in association_matrix:
                    # apply gating on boxoverlap
                    c = cost_matrix[row][col]
                    if c < max_cost:
                        g[row].tracker = t[col].track_id
                        this_ids[1][row] = t[col].track_id
                        t[col].valid = True
                        g[row].distance = c
                        self.total_cost += 1 - c
                        tmpc += 1 - c
                        tmpcs[row] = 1 - c
                        seq_trajectories[g[row].track_id][-1] = t[col].track_id

                        # true positives are only valid associations
                        self.tp += 1
                        tmptp += 1
                    else:
                        g[row].tracker = -1
                        self.fn += 1
                        tmpfn += 1

                # associate tracker and DontCare areas
                # ignore tracker in neighboring classes
                nignoredtracker = 0  # number of ignored tracker detections
                ignoredtrackers = dict()  # will associate the track_id with -1
                # if it is not ignored and 1 if it is
                # ignored;
                # this is used to avoid double counting ignored
                # cases, see the next loop

                for tt in t:
                    ignoredtrackers[tt.track_id] = -1
                    # ignore detection if it belongs to a neighboring class or is
                    # smaller or equal to the minimum height

                    tt_height = abs(tt.y1 - tt.y2)
                    if ((self.cls == "car" and tt.obj_type == "van") or
                        (self.cls == "pedestrian" and
                         tt.obj_type == "person_sitting") or
                            tt_height <= self.min_height) and not tt.valid:
                        nignoredtracker += 1
                        tt.ignored = True
                        ignoredtrackers[tt.track_id] = 1
                        continue
                    for d in dc:
                        overlap = self.boxoverlap(tt, d, "a")
                        if overlap > 0.5 and not tt.valid:
                            tt.ignored = True
                            nignoredtracker += 1
                            ignoredtrackers[tt.track_id] = 1
                            break

                # check for ignored FN/TP (truncation or neighboring object class)
                ignoredfn = 0  # the number of ignored false negatives
                nignoredtp = 0  # the number of ignored true positives
                nignoredpairs = 0  # the number of ignored pairs, i.e. a true positive
                # which is ignored but where the associated tracker
                # detection has already been ignored

                gi = 0
                for gg in g:
                    if gg.tracker < 0:
                        if gg.occlusion>self.max_occlusion or gg.truncation>self.max_truncation\
                                or (self.cls=="car" and gg.obj_type=="van") or (self.cls=="pedestrian" and gg.obj_type=="person_sitting"):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1

                    elif gg.tracker >= 0:
                        if gg.occlusion>self.max_occlusion or gg.truncation>self.max_truncation\
                                or (self.cls=="car" and gg.obj_type=="van") or (self.cls=="pedestrian" and gg.obj_type=="person_sitting"):

                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1

                            # if the associated tracker detection is already ignored,
                            # we want to avoid double counting ignored detections
                            if ignoredtrackers[gg.tracker] > 0:
                                nignoredpairs += 1

                            # for computing MODP, the overlaps from ignored detections
                            # are subtracted
                            tmpc -= tmpcs[gi]
                    gi += 1

                # the below might be confusion, check the comments in __init__
                # to see what the individual statistics represent

                # correct TP by number of ignored TP due to truncation
                # ignored TP are shown as tracked in visualization
                tmptp -= nignoredtp

                # count the number of ignored true positives
                self.itp += nignoredtp

                # adjust the number of ground truth objects considered
                self.n_gt -= (ignoredfn + nignoredtp)

                # count the number of ignored ground truth objects
                self.n_igt += ignoredfn + nignoredtp

                # count the number of ignored tracker objects
                self.n_itr += nignoredtracker

                # count the number of ignored pairs, i.e. associated tracker and
                # ground truth objects that are both ignored
                self.n_igttr += nignoredpairs

                # false negatives = associated gt bboxes exceding association threshold + non-associated gt bboxes
                tmpfn += len(g) - len(association_matrix) - ignoredfn
                self.fn += len(g) - len(association_matrix) - ignoredfn
                self.ifn += ignoredfn

                # false positives = tracker bboxes - associated tracker bboxes
                # mismatches (mme_t)
                tmpfp += len(
                    t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs
                self.fp += len(
                    t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs

                # update sequence data
                seqtp += tmptp
                seqitp += nignoredtp
                seqfp += tmpfp
                seqfn += tmpfn
                seqifn += ignoredfn
                seqigt += ignoredfn + nignoredtp
                seqitr += nignoredtracker

                # sanity checks
                # - the number of true positives minues ignored true positives
                #   should be greater or equal to 0
                # - the number of false negatives should be greater or equal to 0
                # - the number of false positives needs to be greater or equal to 0
                #   otherwise ignored detections might be counted double
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false negatives (plus ignored ones)
                #   should match the total number of ground truth objects
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false positives
                #   plus the number of ignored tracker detections should
                #   match the total number of tracker detections; note that
                #   nignoredpairs is subtracted here to avoid double counting
                #   of ignored detection sin nignoredtp and nignoredtracker
                if tmptp < 0:
                    print(tmptp, nignoredtp)
                    raise NameError("Something went wrong! TP is negative")
                if tmpfn < 0:
                    print(tmpfn,
                          len(g),
                          len(association_matrix), ignoredfn, nignoredpairs)
                    raise NameError("Something went wrong! FN is negative")
                if tmpfp < 0:
                    print(tmpfp,
                          len(t), tmptp, nignoredtracker, nignoredtp,
                          nignoredpairs)
                    raise NameError("Something went wrong! FP is negative")
                if tmptp + tmpfn is not len(g) - ignoredfn - nignoredtp:
                    print("seqidx", seq_idx)
                    print("frame ", f)
                    print("TP    ", tmptp)
                    print("FN    ", tmpfn)
                    print("FP    ", tmpfp)
                    print("nGT   ", len(g))
                    print("nAss  ", len(association_matrix))
                    print("ign GT", ignoredfn)
                    print("ign TP", nignoredtp)
                    raise NameError(
                        "Something went wrong! nGroundtruth is not TP+FN")
                if tmptp + tmpfp + nignoredtp + nignoredtracker - nignoredpairs is not len(
                        t):
                    print(seq_idx, f, len(t), tmptp, tmpfp)
                    print(len(association_matrix), association_matrix)
                    raise NameError(
                        "Something went wrong! nTracker is not TP+FP")

                # check for id switches or fragmentations
                for i, tt in enumerate(this_ids[0]):
                    if tt in last_ids[0]:
                        idx = last_ids[0].index(tt)
                        tid = this_ids[1][i]
                        lid = last_ids[1][idx]
                        if tid != lid and lid != -1 and tid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].id_switch = 1
                                ids += 1
                        if tid != lid and lid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].fragmentation = 1
                                fr += 1

                # save current index
                last_ids = this_ids
                # compute MOTP_t
                MODP_t = 1
                if tmptp != 0:
                    MODP_t = tmpc / float(tmptp)
                self.MODP_t.append(MODP_t)

            # remove empty lists for current gt trajectories
            self.gt_trajectories[seq_idx] = seq_trajectories
            self.ign_trajectories[seq_idx] = seq_ignored

            # gather statistics for "per sequence" statistics.
            self.n_gts.append(n_gts)
            self.n_trs.append(n_trs)
            self.tps.append(seqtp)
            self.itps.append(seqitp)
            self.fps.append(seqfp)
            self.fns.append(seqfn)
            self.ifns.append(seqifn)
            self.n_igts.append(seqigt)
            self.n_itrs.append(seqitr)

        # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
        n_ignored_tr_total = 0
        for seq_idx, (
                seq_trajectories, seq_ignored
        ) in enumerate(zip(self.gt_trajectories, self.ign_trajectories)):
            if len(seq_trajectories) == 0:
                continue
            tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0] * 5
            n_ignored_tr = 0
            for g, ign_g in zip(seq_trajectories.values(),
                                seq_ignored.values()):
                # all frames of this gt trajectory are ignored
                if all(ign_g):
                    n_ignored_tr += 1
                    n_ignored_tr_total += 1
                    continue
                # all frames of this gt trajectory are not assigned to any detections
                if all([this == -1 for this in g]):
                    tmpML += 1
                    self.ML += 1
                    continue
                # compute tracked frames in trajectory
                last_id = g[0]
                # first detection (necessary to be in gt_trajectories) is always tracked
                tracked = 1 if g[0] >= 0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1, len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt += 1
                    if last_id != g[f] and last_id != -1 and g[f] != -1 and g[
                            f - 1] != -1:
                        tmpId_switches += 1
                        self.id_switches += 1
                    if f < len(g) - 1 and g[f - 1] != g[
                            f] and last_id != -1 and g[f] != -1 and g[f +
                                                                      1] != -1:
                        tmpFragments += 1
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]
                # handle last frame; tracked state is handled in for loop (g[f]!=-1)
                if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[
                        f] != -1 and not ign_g[f]:
                    tmpFragments += 1
                    self.fragments += 1

                # compute MT/PT/ML
                tracking_ratio = tracked / float(len(g) - sum(ign_g))
                if tracking_ratio > 0.8:
                    tmpMT += 1
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    tmpML += 1
                    self.ML += 1
                else:  # 0.2 <= tracking_ratio <= 0.8
                    tmpPT += 1
                    self.PT += 1

        if (self.n_gt_trajectories - n_ignored_tr_total) == 0:
            self.MT = 0.
            self.PT = 0.
            self.ML = 0.
        else:
            self.MT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories - n_ignored_tr_total)

        # precision/recall etc.
        if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp / float(self.tp + self.fn)
            self.precision = self.tp / float(self.fp + self.tp)
        if (self.recall + self.precision) == 0:
            self.F1 = 0.
        else:
            self.F1 = 2. * (self.precision * self.recall) / (
                self.precision + self.recall)
        if sum(self.n_frames) == 0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp / float(sum(self.n_frames))

        # compute CLEARMOT
        if self.n_gt == 0:
            self.MOTA = -float("inf")
            self.MODA = -float("inf")
        else:
            self.MOTA = 1 - (self.fn + self.fp + self.id_switches
                             ) / float(self.n_gt)
            self.MODA = 1 - (self.fn + self.fp) / float(self.n_gt)
        if self.tp == 0:
            self.MOTP = float("inf")
        else:
            self.MOTP = self.total_cost / float(self.tp)
        if self.n_gt != 0:
            if self.id_switches == 0:
                self.MOTAL = 1 - (self.fn + self.fp + self.id_switches
                                  ) / float(self.n_gt)
            else:
                self.MOTAL = 1 - (self.fn + self.fp +
                                  math.log10(self.id_switches)
                                  ) / float(self.n_gt)
        else:
            self.MOTAL = -float("inf")
        if sum(self.n_frames) == 0:
            self.MODP = "n/a"
        else:
            self.MODP = sum(self.MODP_t) / float(sum(self.n_frames))
        return True

    def createSummary(self):
        summary = ""
        summary += "tracking evaluation summary".center(80, "=") + "\n"
        summary += self.printEntry("Multiple Object Tracking Accuracy (MOTA)",
                                   self.MOTA) + "\n"
        summary += self.printEntry("Multiple Object Tracking Precision (MOTP)",
                                   self.MOTP) + "\n"
        summary += self.printEntry("Multiple Object Tracking Accuracy (MOTAL)",
                                   self.MOTAL) + "\n"
        summary += self.printEntry("Multiple Object Detection Accuracy (MODA)",
                                   self.MODA) + "\n"
        summary += self.printEntry("Multiple Object Detection Precision (MODP)",
                                   self.MODP) + "\n"
        summary += "\n"
        summary += self.printEntry("Recall", self.recall) + "\n"
        summary += self.printEntry("Precision", self.precision) + "\n"
        summary += self.printEntry("F1", self.F1) + "\n"
        summary += self.printEntry("False Alarm Rate", self.FAR) + "\n"
        summary += "\n"
        summary += self.printEntry("Mostly Tracked", self.MT) + "\n"
        summary += self.printEntry("Partly Tracked", self.PT) + "\n"
        summary += self.printEntry("Mostly Lost", self.ML) + "\n"
        summary += "\n"
        summary += self.printEntry("True Positives", self.tp) + "\n"
        #summary += self.printEntry("True Positives per Sequence", self.tps) + "\n"
        summary += self.printEntry("Ignored True Positives", self.itp) + "\n"
        #summary += self.printEntry("Ignored True Positives per Sequence", self.itps) + "\n"

        summary += self.printEntry("False Positives", self.fp) + "\n"
        #summary += self.printEntry("False Positives per Sequence", self.fps) + "\n"
        summary += self.printEntry("False Negatives", self.fn) + "\n"
        #summary += self.printEntry("False Negatives per Sequence", self.fns) + "\n"
        summary += self.printEntry("ID-switches", self.id_switches) + "\n"
        self.fp = self.fp / self.n_gt
        self.fn = self.fn / self.n_gt
        self.id_switches = self.id_switches / self.n_gt
        summary += self.printEntry("False Positives Ratio", self.fp) + "\n"
        #summary += self.printEntry("False Positives per Sequence", self.fps) + "\n"
        summary += self.printEntry("False Negatives Ratio", self.fn) + "\n"
        #summary += self.printEntry("False Negatives per Sequence", self.fns) + "\n"
        summary += self.printEntry("Ignored False Negatives Ratio",
                                   self.ifn) + "\n"

        #summary += self.printEntry("Ignored False Negatives per Sequence", self.ifns) + "\n"
        summary += self.printEntry("Missed Targets", self.fn) + "\n"
        summary += self.printEntry("ID-switches", self.id_switches) + "\n"
        summary += self.printEntry("Fragmentations", self.fragments) + "\n"
        summary += "\n"
        summary += self.printEntry("Ground Truth Objects (Total)", self.n_gt +
                                   self.n_igt) + "\n"
        #summary += self.printEntry("Ground Truth Objects (Total) per Sequence", self.n_gts) + "\n"
        summary += self.printEntry("Ignored Ground Truth Objects",
                                   self.n_igt) + "\n"
        #summary += self.printEntry("Ignored Ground Truth Objects per Sequence", self.n_igts) + "\n"
        summary += self.printEntry("Ground Truth Trajectories",
                                   self.n_gt_trajectories) + "\n"
        summary += "\n"
        summary += self.printEntry("Tracker Objects (Total)", self.n_tr) + "\n"
        #summary += self.printEntry("Tracker Objects (Total) per Sequence", self.n_trs) + "\n"
        summary += self.printEntry("Ignored Tracker Objects", self.n_itr) + "\n"
        #summary += self.printEntry("Ignored Tracker Objects per Sequence", self.n_itrs) + "\n"
        summary += self.printEntry("Tracker Trajectories",
                                   self.n_tr_trajectories) + "\n"
        #summary += "\n"
        #summary += self.printEntry("Ignored Tracker Objects with Associated Ignored Ground Truth Objects", self.n_igttr) + "\n"
        summary += "=" * 80
        return summary

    def printEntry(self, key, val, width=(70, 10)):
        """
            Pretty print an entry in a table fashion.
        """
        s_out = key.ljust(width[0])
        if type(val) == int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val) == float:
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s" % val).rjust(width[1])
        return s_out

    def saveToStats(self, save_summary):
        """
            Save the statistics in a whitespace separate file.
        """
        summary = self.createSummary()
        if save_summary:
            filename = os.path.join(self.result_path,
                                    "summary_%s.txt" % self.cls)
            dump = open(filename, "w+")
            dump.write(summary)
            dump.close()
        return summary


class KITTIMOTMetric(Metric):
    def __init__(self, save_summary=True):
        self.save_summary = save_summary
        self.MOTEvaluator = KITTIEvaluation
        self.result_root = None
        self.reset()

    def reset(self):
        self.seqs = []
        self.n_sequences = 0
        self.n_frames = []
        self.strsummary = ''

    def update(self, data_root, seq, data_type, result_root, result_filename):
        assert data_type == 'kitti', "data_type should 'kitti'"
        self.result_root = result_root
        self.gt_path = data_root
        gt_path = '{}/../labels/{}.txt'.format(data_root, seq)
        gt = open(gt_path, "r")
        max_frame = 0
        for line in gt:
            line = line.strip()
            line_list = line.split(" ")
            if int(line_list[0]) > max_frame:
                max_frame = int(line_list[0])
        rs = open(result_filename, "r")
        for line in rs:
            line = line.strip()
            line_list = line.split(" ")
            if int(line_list[0]) > max_frame:
                max_frame = int(line_list[0])
        gt.close()
        rs.close()
        self.n_frames.append(max_frame + 1)
        self.seqs.append(seq)
        self.n_sequences += 1

    def accumulate(self):
        logger.info("Processing Result for KITTI Tracking Benchmark")
        e = self.MOTEvaluator(result_path=self.result_root, gt_path=self.gt_path,\
            n_frames=self.n_frames, seqs=self.seqs, n_sequences=self.n_sequences)
        try:
            if not e.loadTracker():
                return
            logger.info("Loading Results - Success")
            logger.info("Evaluate Object Class: %s" % c.upper())
        except:
            logger.info("Caught exception while loading result data.")
        if not e.loadGroundtruth():
            raise ValueError("Ground truth not found.")
        logger.info("Loading Groundtruth - Success")
        # sanity checks
        if len(e.groundtruth) is not len(e.tracker):
            logger.info(
                "The uploaded data does not provide results for every sequence.")
            return False
        logger.info("Loaded %d Sequences." % len(e.groundtruth))
        logger.info("Start Evaluation...")

        if e.compute3rdPartyMetrics():
            self.strsummary = e.saveToStats(self.save_summary)
        else:
            logger.info(
                "There seem to be no true positives or false positives at all in the submitted data."
            )

    def log(self):
        print(self.strsummary)

    def get_results(self):
        return self.strsummary
