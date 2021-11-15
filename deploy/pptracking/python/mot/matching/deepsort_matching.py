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
This code is based on https://github.com/nwojke/deep_sort/tree/master/deep_sort
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from ..motion import kalman_filter

INFTY_COST = 1e+5

__all__ = [
    'iou_1toN',
    'iou_cost',
    '_nn_euclidean_distance',
    '_nn_cosine_distance',
    'NearestNeighborDistanceMetric',
    'min_cost_matching',
    'matching_cascade',
    'gate_cost_matrix',
]


def iou_1toN(bbox, candidates):
    """
    Computer intersection over union (IoU) by one box to N candidates.

    Args:
        bbox (ndarray): A bounding box in format `(top left x, top left y, width, height)`.
            candidates (ndarray): A matrix of candidate bounding boxes (one per row) in the
            same format as `bbox`.

    Returns:
        ious (ndarray): The intersection over union in [0, 1] between the `bbox`
            and each candidate. A higher score means a larger fraction of the
            `bbox` is occluded by the candidate.
    """
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    ious = area_intersection / (area_bbox + area_candidates - area_intersection)
    return ious


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    IoU distance metric.

    Args:
        tracks (list[Track]): A list of tracks.
        detections (list[Detection]): A list of detections.
        track_indices (Optional[list[int]]): A list of indices to tracks that
            should be matched. Defaults to all `tracks`.
        detection_indices (Optional[list[int]]): A list of indices to detections
            that should be matched. Defaults to all `detections`.

    Returns:
        cost_matrix (ndarray): A cost matrix of shape len(track_indices), 
            len(detection_indices) where entry (i, j) is 
            `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = 1e+5
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou_1toN(bbox, candidates)
    return cost_matrix


def _nn_euclidean_distance(s, q):
    """
    Compute pair-wise squared (Euclidean) distance between points in `s` and `q`.

    Args:
        s (ndarray): Sample points: an NxM matrix of N samples of dimensionality M.
        q (ndarray): Query points: an LxM matrix of L samples of dimensionality M.

    Returns:
        distances (ndarray): A vector of length M that contains for each entry in `q` the
            smallest Euclidean distance to a sample in `s`.
    """
    s, q = np.asarray(s), np.asarray(q)
    if len(s) == 0 or len(q) == 0:
        return np.zeros((len(s), len(q)))
    s2, q2 = np.square(s).sum(axis=1), np.square(q).sum(axis=1)
    distances = -2. * np.dot(s, q.T) + s2[:, None] + q2[None, :]
    distances = np.clip(distances, 0., float(np.inf))

    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(s, q):
    """
    Compute pair-wise cosine distance between points in `s` and `q`.

    Args:
        s (ndarray): Sample points: an NxM matrix of N samples of dimensionality M.
        q (ndarray): Query points: an LxM matrix of L samples of dimensionality M.

    Returns:
        distances (ndarray): A vector of length M that contains for each entry in `q` the
            smallest Euclidean distance to a sample in `s`.
    """
    s = np.asarray(s) / np.linalg.norm(s, axis=1, keepdims=True)
    q = np.asarray(q) / np.linalg.norm(q, axis=1, keepdims=True)
    distances = 1. - np.dot(s, q.T)

    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Args:
        metric (str): Either "euclidean" or "cosine".
        matching_threshold (float): The matching threshold. Samples with larger
            distance are considered an invalid match.
        budget (Optional[int]): If not None, fix samples per class to at most
            this number. Removes the oldest samples when the budget is reached.

    Attributes: 
        samples (Dict[int -> List[ndarray]]): A dictionary that maps from target
            identities to the list of samples that have been observed so far.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """
        Update the distance metric with new data.

        Args:
            features (ndarray): An NxM matrix of N features of dimensionality M.
            targets (ndarray): An integer array of associated target identities.
            active_targets (List[int]): A list of targets that are currently
                present in the scene.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """
        Compute distance between features and targets.

        Args:
            features (ndarray): An NxM matrix of N features of dimensionality M.
            targets (list[int]): A list of targets to match the given `features` against.

        Returns:
            cost_matrix (ndarray): a cost matrix of shape len(targets), len(features),
                where element (i, j) contains the closest squared distance between
                `targets[i]` and `features[j]`.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


def min_cost_matching(distance_metric,
                      max_distance,
                      tracks,
                      detections,
                      track_indices=None,
                      detection_indices=None):
    """
    Solve linear assignment problem.

    Args:
        distance_metric :
            Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
            The distance metric is given a list of tracks and detections as 
            well as a list of N track indices and M detection indices. The 
            metric should return the NxM dimensional cost matrix, where element
            (i, j) is the association cost between the i-th track in the given
            track indices and the j-th detection in the given detection_indices.
        max_distance (float): Gating threshold. Associations with cost larger
            than this value are disregarded.
        tracks (list[Track]): A list of predicted tracks at the current time
            step.
        detections (list[Detection]): A list of detections at the current time
            step.
        track_indices (list[int]): List of track indices that maps rows in
            `cost_matrix` to tracks in `tracks`.
        detection_indices (List[int]): List of detection indices that maps
            columns in `cost_matrix` to detections in `detections`.

    Returns:
        A tuple (List[(int, int)], List[int], List[int]) with the following
        three entries:
            * A list of matched track and detection indices.
            * A list of unmatched track indices.
            * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices,
                                  detection_indices)

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[0]:
            unmatched_tracks.append(track_idx)
    for row, col in zip(indices[0], indices[1]):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric,
                     max_distance,
                     cascade_depth,
                     tracks,
                     detections,
                     track_indices=None,
                     detection_indices=None):
    """
    Run matching cascade.

    Args:
        distance_metric :
            Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
            The distance metric is given a list of tracks and detections as 
            well as a list of N track indices and M detection indices. The 
            metric should return the NxM dimensional cost matrix, where element
            (i, j) is the association cost between the i-th track in the given
            track indices and the j-th detection in the given detection_indices.
        max_distance (float): Gating threshold. Associations with cost larger
            than this value are disregarded.
        cascade_depth (int): The cascade depth, should be se to the maximum
            track age.
        tracks (list[Track]): A list of predicted tracks at the current time
            step.
        detections (list[Detection]): A list of detections at the current time
            step.
        track_indices (list[int]): List of track indices that maps rows in
            `cost_matrix` to tracks in `tracks`.
        detection_indices (List[int]): List of detection indices that maps
            columns in `cost_matrix` to detections in `detections`.

    Returns:
        A tuple (List[(int, int)], List[int], List[int]) with the following
        three entries:
            * A list of matched track and detection indices.
            * A list of unmatched track indices.
            * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf,
                     cost_matrix,
                     tracks,
                     detections,
                     track_indices,
                     detection_indices,
                     gated_cost=INFTY_COST,
                     only_position=False):
    """
    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Args:
        kf (object): The Kalman filter.
        cost_matrix (ndarray): The NxM dimensional cost matrix, where N is the
            number of track indices and M is the number of detection indices,
            such that entry (i, j) is the association cost between
            `tracks[track_indices[i]]` and `detections[detection_indices[j]]`.
        tracks (list[Track]): A list of predicted tracks at the current time
            step.
        detections (list[Detection]): A list of detections at the current time
            step.
        track_indices (List[int]): List of track indices that maps rows in
            `cost_matrix` to tracks in `tracks`.
        detection_indices (List[int]): List of detection indices that maps
            columns in `cost_matrix` to detections in `detections`.
        gated_cost (Optional[float]): Entries in the cost matrix corresponding
            to infeasible associations are set this value. Defaults to a very
            large value.
        only_position (Optional[bool]): If True, only the x, y position of the
            state distribution is considered during gating. Default False.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance,
                                             measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
