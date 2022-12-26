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
This code is based on https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/matching.py
"""

try:
    import lap
except:
    print(
        'Warning: Unable to use JDE/FairMOT/ByteTrack, please install lap, for example: `pip install lap`, see https://github.com/gatagat/lap'
    )
    pass

import scipy
import numpy as np
from scipy.spatial.distance import cdist
from ..motion import kalman_filter
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'merge_matches',
    'linear_assignment',
    'bbox_ious',
    'iou_distance',
    'embedding_distance',
    'fuse_motion',
]


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix(
        (np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix(
        (np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(cost_matrix, thresh):
    try:
        import lap
    except Exception as e:
        raise RuntimeError(
            'Unable to use JDE/FairMOT/ByteTrack, please install lap, for example: `pip install lap`, see https://github.com/gatagat/lap'
        )
    if cost_matrix.size == 0:
        return np.empty(
            (0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(
                range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def bbox_ious(atlbrs, btlbrs):
    boxes = np.ascontiguousarray(atlbrs, dtype=np.float32)
    query_boxes = np.ascontiguousarray(btlbrs, dtype=np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    ious = np.zeros((N, K), dtype=boxes.dtype)
    if N * K == 0:
        return ious

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[
                        n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    ious[n, k] = iw * ih / ua
    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU between two list[STrack].
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = bbox_ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='euclidean'):
    """
    Compute cost based on features between two list[STrack].
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray(
        [track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features,
                                        metric))  # Nomalized features
    return cost_matrix


def fuse_motion(kf,
                cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
            metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_
                                                         ) * gating_distance
    return cost_matrix
