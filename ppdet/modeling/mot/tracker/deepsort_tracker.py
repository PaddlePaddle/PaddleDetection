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
"""
This code is based on https://github.com/nwojke/deep_sort/blob/master/deep_sort/tracker.py
"""

import numpy as np

from ..motion import KalmanFilter
from ..matching.deepsort_matching import NearestNeighborDistanceMetric
from ..matching.deepsort_matching import iou_cost, min_cost_matching, matching_cascade, gate_cost_matrix
from .base_sde_tracker import Track
from ..utils import Detection

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['DeepSORTTracker']


@register
@serializable
class DeepSORTTracker(object):
    """
    DeepSORT tracker

    Args:
        input_size (list): input feature map size to reid model, [h, w] format,
            [64, 192] as default.
        min_box_area (int): min box area to filter out low quality boxes
        vertical_ratio (float): w/h, the vertical ratio of the bbox to filter
            bad results, set 1.6 default for pedestrian tracking. If set <=0
            means no need to filter bboxes.
        budget (int): If not None, fix samples per class to at most this number.
            Removes the oldest samples when the budget is reached.
        max_age (int): maximum number of missed misses before a track is deleted
        n_init (float): Number of frames that a track remains in initialization
            phase. Number of consecutive detections before the track is confirmed. 
            The track state is set to `Deleted` if a miss occurs within the first 
            `n_init` frames.
        metric_type (str): either "euclidean" or "cosine", the distance metric 
            used for measurement to track association.
        matching_threshold (float): samples with larger distance are 
            considered an invalid match.
        max_iou_distance (float): max iou distance threshold
        motion (object): KalmanFilter instance
    """

    def __init__(self,
                 input_size=[64, 192],
                 min_box_area=0,
                 vertical_ratio=-1,
                 budget=100,
                 max_age=70,
                 n_init=3,
                 metric_type='cosine',
                 matching_threshold=0.2,
                 max_iou_distance=0.9,
                 motion='KalmanFilter'):
        self.input_size = input_size
        self.min_box_area = min_box_area
        self.vertical_ratio = vertical_ratio
        self.max_age = max_age
        self.n_init = n_init
        self.metric = NearestNeighborDistanceMetric(metric_type,
                                                    matching_threshold, budget)
        self.max_iou_distance = max_iou_distance
        if motion == 'KalmanFilter':
            self.motion = KalmanFilter()

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """
        Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.motion)

    def update(self, pred_dets, pred_embs):
        """
        Perform measurement update and track management.
        Args:
            pred_dets (np.array): Detection results of the image, the shape is
                [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
            pred_embs (np.array): Embedding results of the image, the shape is
                [N, 128], usually pred_embs.shape[1] is a multiple of 128.
        """
        pred_cls_ids = pred_dets[:, 0:1]
        pred_scores = pred_dets[:, 1:2]
        pred_xyxys = pred_dets[:, 2:6]
        pred_tlwhs = np.concatenate((pred_xyxys[:, 0:2], pred_xyxys[:, 2:4] - pred_xyxys[:, 0:2] + 1), axis=1)

        detections = [
            Detection(tlwh, score, feat, cls_id)
            for tlwh, score, feat, cls_id in zip(pred_tlwhs, pred_scores,
                                                 pred_embs, pred_cls_ids)
        ]

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.motion,
                                          detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        output_stracks = self.tracks
        return output_stracks

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(self.motion, cost_matrix, tracks,
                                           dets, track_indices,
                                           detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()
        ]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(
                iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.motion.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, covariance, self._next_id, self.n_init, self.max_age,
                  detection.cls_id, detection.score, detection.feature))
        self._next_id += 1
