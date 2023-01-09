# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
This code is based on https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/utils/tracker.py
"""

import copy
import numpy as np
import sklearn

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['CenterTracker']


@register
@serializable
class CenterTracker(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=1,
                 min_box_area=0,
                 vertical_ratio=-1,
                 track_thresh=0.4,
                 pre_thresh=0.5,
                 new_thresh=0.4,
                 out_thresh=0.4,
                 hungarian=False):
        self.num_classes = num_classes
        self.min_box_area = min_box_area
        self.vertical_ratio = vertical_ratio

        self.track_thresh = track_thresh
        self.pre_thresh = max(track_thresh, pre_thresh)
        self.new_thresh = max(track_thresh, new_thresh)
        self.out_thresh = max(track_thresh, out_thresh)
        self.hungarian = hungarian

        self.reset()

    def init_track(self, results):
        print('Initialize tracking!')
        for item in results:
            if item['score'] > self.new_thresh:
                self.id_count += 1
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2,
                                  (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def update(self, results, public_det=None):
        N = len(results)
        M = len(self.tracks)

        dets = np.array([det['ct'] + det['tracking'] for det in results],
                        np.float32)  # N x 2
        track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
            (track['bbox'][3] - track['bbox'][1])) \
            for track in self.tracks], np.float32) # M
        track_cat = np.array([track['class'] for track in self.tracks],
                             np.int32)  # M
        item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
            (item['bbox'][3] - item['bbox'][1])) \
            for item in results], np.float32) # N
        item_cat = np.array([item['class'] for item in results], np.int32)  # N
        tracks = np.array([pre_det['ct'] for pre_det in self.tracks],
                          np.float32)  # M x 2
        dist = (((tracks.reshape(1, -1, 2) - \
            dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M

        invalid = ((dist > track_size.reshape(1, M)) + \
            (dist > item_size.reshape(N, 1)) + \
            (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
        dist = dist + invalid * 1e18

        if self.hungarian:
            item_score = np.array([item['score'] for item in results],
                                  np.float32)
            dist[dist > 1e18] = 1e18
            from sklearn.utils.linear_assignment_ import linear_assignment
            matched_indices = linear_assignment(dist)
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))

        unmatched_dets = [d for d in range(dets.shape[0]) \
            if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) \
            if not (d in matched_indices[:, 1])]

        if self.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ret = []
        for m in matches:
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            ret.append(track)

        # Private detection: create tracks for all un-matched detections
        for i in unmatched_dets:
            track = results[i]
            if track['score'] > self.new_thresh:
                self.id_count += 1
                track['tracking_id'] = self.id_count
                ret.append(track)

        self.tracks = ret
        return ret


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)
