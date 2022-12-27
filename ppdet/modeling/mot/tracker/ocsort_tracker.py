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
This code is based on https://github.com/noahcao/OC_SORT/blob/master/trackers/ocsort_tracker/ocsort.py
"""

import numpy as np
from ..matching.ocsort_matching import associate, linear_assignment, iou_batch, associate_only_iou
from ..motion.ocsort_kalman_filter import OCSORTKalmanFilter
from ppdet.core.workspace import register, serializable


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array(
            [x[0] - w / 2., x[1] - h / 2., x[0] + w / 2.,
             x[1] + h / 2.]).reshape((1, 4))
    else:
        score = np.array([score])
        return np.array([
            x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score
        ]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.

    Args:
        bbox (np.array): bbox in [x1,y1,x2,y2,score] format.
        delta_t (int): delta_t of previous observation
    """
    count = 0

    def __init__(self, bbox, delta_t=3):

        self.kf = OCSORTKalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1., 0, 0, 0, 1., 0, 0], [0, 1., 0, 0, 0, 1., 0],
                              [0, 0, 1., 0, 0, 0, 1], [0, 0, 0, 1., 0, 0, 0],
                              [0, 0, 0, 0, 1., 0, 0], [0, 0, 0, 0, 0, 1., 0],
                              [0, 0, 0, 0, 0, 0, 1.]])
        self.kf.H = np.array([[1., 0, 0, 0, 0, 0, 0], [0, 1., 0, 0, 0, 0, 0],
                              [0, 0, 1., 0, 0, 0, 0], [0, 0, 0, 1., 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.score = bbox[4]
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox, angle_cost=False):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if angle_cost and self.last_observation.sum(
            ) >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x, score=self.score))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x, score=self.score)


@register
@serializable
class OCSORTTracker(object):
    """
    OCSORT tracker, support single class

    Args:
        det_thresh (float): threshold of detection score
        max_age (int): maximum number of missed misses before a track is deleted
        min_hits (int): minimum hits for associate
        iou_threshold (float): iou threshold for associate
        delta_t (int): delta_t of previous observation
        inertia (float): vdc_weight of angle_diff_cost for associate
        vertical_ratio (float): w/h, the vertical ratio of the bbox to filter
            bad results. If set <= 0 means no need to filter bboxes，usually set
            1.6 for pedestrian tracking.
        min_box_area (int): min box area to filter out low quality boxes
        use_byte (bool): Whether use ByteTracker, default False
    """

    def __init__(self,
                 det_thresh=0.6,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 delta_t=3,
                 inertia=0.2,
                 vertical_ratio=-1,
                 min_box_area=0,
                 use_byte=False,
                 use_angle_cost=False):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.vertical_ratio = vertical_ratio
        self.min_box_area = min_box_area
        self.use_byte = use_byte
        self.use_angle_cost = use_angle_cost

        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, pred_dets, pred_embs=None):
        """
        Args:
            pred_dets (np.array): Detection results of the image, the shape is
                [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
            pred_embs (np.array): Embedding results of the image, the shape is
                [N, 128] or [N, 512], default as None.

        Return:
            tracking boxes (np.array): [M, 6], means 'x0, y0, x1, y1, score, id'.
        """
        if pred_dets is None:
            return np.empty((0, 6))

        self.frame_count += 1

        bboxes = pred_dets[:, 2:]
        scores = pred_dets[:, 1:2]
        dets = np.concatenate((bboxes, scores), axis=1)
        scores = scores.squeeze(-1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if self.use_angle_cost:
            velocities = np.array([
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.trackers
            ])

            k_observations = np.array([
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.trackers
            ])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        """
            First round of association
        """
        if self.use_angle_cost:
            matched, unmatched_dets, unmatched_trks = associate(
                dets, trks, self.iou_threshold, velocities, k_observations,
                self.inertia)
        else:
            matched, unmatched_dets, unmatched_trks = associate_only_iou(
                dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(
                dets[m[0], :], angle_cost=self.use_angle_cost)
        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[
                0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = iou_batch(
                dets_second,
                u_trks)  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(
                        dets_second[det_ind, :], angle_cost=self.use_angle_cost)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks,
                                              np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = iou_batch(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[
                        1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(
                        dets[det_ind, :], angle_cost=self.use_angle_cost)
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets,
                                              np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks,
                                              np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation  # tlbr + score
            if (trk.time_since_update < 1) and (
                    trk.hit_streak >= self.min_hits or
                    self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))
