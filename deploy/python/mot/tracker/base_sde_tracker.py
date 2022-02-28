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
This code is based on https://github.com/nwojke/deep_sort/blob/master/deep_sort/track.py
"""

import datetime

__all__ = ['TrackState', 'Track']


class TrackState(object):
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track(object):
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Args:
        mean (ndarray): Mean vector of the initial state distribution.
        covariance (ndarray): Covariance matrix of the initial state distribution.
        track_id (int): A unique track identifier.
        n_init (int): Number of consecutive detections before the track is confirmed.
            The track state is set to `Deleted` if a miss occurs within the first
            `n_init` frames.
        max_age (int): The maximum number of consecutive misses before the track
            state is set to `Deleted`.
        cls_id (int): The category id of the tracked box.
        score (float): The confidence score of the tracked box.
        feature (Optional[ndarray]): Feature vector of the detection this track
            originates from. If not None, this feature is added to the `features` cache.

    Attributes:
        hits (int): Total number of measurement updates.
        age (int): Total number of frames since first occurance.
        time_since_update (int): Total number of frames since last measurement
            update.
        state (TrackState): The current track state.
        features (List[ndarray]): A cache of features. On each measurement update,
            the associated feature vector is added to this list.
    """

    def __init__(self,
                 mean,
                 covariance,
                 track_id,
                 n_init,
                 max_age,
                 cls_id,
                 score,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.cls_id = cls_id
        self.score = score
        self.start_time = datetime.datetime.now()

        self.state = TrackState.Tentative
        self.features = []
        self.feat = feature
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get position in format `(top left x, top left y, width, height)`."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get position in bounding box format `(min x, miny, max x, max y)`."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kalman_filter):
        """
        Propagate the state distribution to the current time step using a Kalman
        filter prediction step.
        """
        self.mean, self.covariance = kalman_filter.predict(self.mean,
                                                           self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kalman_filter, detection):
        """
        Perform Kalman filter measurement update step and update the associated
        detection feature cache.
        """
        self.mean, self.covariance = kalman_filter.update(self.mean,
                                                          self.covariance,
                                                          detection.to_xyah())
        self.features.append(detection.feature)
        self.feat = detection.feature
        self.cls_id = detection.cls_id
        self.score = detection.score

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
