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
This code is borrow from https://github.com/nwojke/deep_sort/blob/master/deep_sort/tracker.py
"""

import numpy as np

from ..matching.deepsort_matching import NearestNeighborDistanceMetric
from ..matching.deepsort_matching import iou_cost, min_cost_matching, matching_cascade, gate_cost_matrix
from .base_sde_tracker import Track

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['DeepSORTTracker']


@register
@serializable
class DeepSORTTracker(object):
    __inject__ = ['motion']
    """
    DeepSORT tracker

    Args:
        img_size (list): input image size, [h, w]
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
                 img_size=[608, 1088],
                 budget=100,
                 max_age=30,
                 n_init=3,
                 metric_type='cosine',
                 matching_threshold=0.2,
                 max_iou_distance=0.7,
                 motion='KalmanFilter'):
        self.img_size = img_size
        self.max_age = max_age
        self.n_init = n_init
        self.metric = NearestNeighborDistanceMetric(metric_type,
                                                    matching_threshold, budget)
        self.max_iou_distance = max_iou_distance
        self.motion = motion

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """
        Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.motion)

    def update(self, detections):
        """
        Perform measurement update and track management.
        Args:
            detections (list): List[ppdet.modeling.mot.utils.Detection]
            A list of detections at the current time step.
        """
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
                  detection.feature))
        self._next_id += 1
