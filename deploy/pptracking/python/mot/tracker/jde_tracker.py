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
This code is based on https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py
"""

import numpy as np
from collections import defaultdict

from ..matching import jde_matching as matching
from ..motion import KalmanFilter
from .base_jde_tracker import TrackState, STrack
from .base_jde_tracker import joint_stracks, sub_stracks, remove_duplicate_stracks

__all__ = ['JDETracker']


class JDETracker(object):
    __shared__ = ['num_classes']
    """
    JDE tracker, support single class and multi classes

    Args:
        use_byte (bool): Whether use ByteTracker, default False
        num_classes (int): the number of classes
        det_thresh (float): threshold of detection score
        track_buffer (int): buffer for tracker
        min_box_area (int): min box area to filter out low quality boxes
        vertical_ratio (float): w/h, the vertical ratio of the bbox to filter
            bad results. If set <= 0 means no need to filter bboxes，usually set
            1.6 for pedestrian tracking.
        tracked_thresh (float): linear assignment threshold of tracked 
            stracks and detections
        r_tracked_thresh (float): linear assignment threshold of 
            tracked stracks and unmatched detections
        unconfirmed_thresh (float): linear assignment threshold of 
            unconfirmed stracks and unmatched detections
        conf_thres (float): confidence threshold for tracking, also used in
            ByteTracker as higher confidence threshold
        match_thres (float): linear assignment threshold of tracked 
            stracks and detections in ByteTracker
        low_conf_thres (float): lower confidence threshold for tracking in
            ByteTracker
        input_size (list): input feature map size to reid model, [h, w] format,
            [64, 192] as default.
        motion (str): motion model, KalmanFilter as default
        metric_type (str): either "euclidean" or "cosine", the distance metric 
            used for measurement to track association.
    """

    def __init__(self,
                 use_byte=False,
                 num_classes=1,
                 det_thresh=0.3,
                 track_buffer=30,
                 min_box_area=0,
                 vertical_ratio=0,
                 tracked_thresh=0.7,
                 r_tracked_thresh=0.5,
                 unconfirmed_thresh=0.7,
                 conf_thres=0,
                 match_thres=0.8,
                 low_conf_thres=0.2,
                 input_size=[64, 192],
                 motion='KalmanFilter',
                 metric_type='euclidean'):
        self.use_byte = use_byte
        self.num_classes = num_classes
        self.det_thresh = det_thresh if not use_byte else conf_thres + 0.1
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.vertical_ratio = vertical_ratio

        self.tracked_thresh = tracked_thresh
        self.r_tracked_thresh = r_tracked_thresh
        self.unconfirmed_thresh = unconfirmed_thresh
        self.conf_thres = conf_thres
        self.match_thres = match_thres
        self.low_conf_thres = low_conf_thres

        self.input_size = input_size
        if motion == 'KalmanFilter':
            self.motion = KalmanFilter()
        self.metric_type = metric_type

        self.frame_id = 0
        self.tracked_tracks_dict = defaultdict(list)  # dict(list[STrack])
        self.lost_tracks_dict = defaultdict(list)  # dict(list[STrack])
        self.removed_tracks_dict = defaultdict(list)  # dict(list[STrack])

        self.max_time_lost = 0
        # max_time_lost will be calculated: int(frame_rate / 30.0 * track_buffer)

    def update(self, pred_dets, pred_embs=None):
        """
        Processes the image frame and finds bounding box(detections).
        Associates the detection with corresponding tracklets and also handles
            lost, removed, refound and active tracklets.

        Args:
            pred_dets (np.array): Detection results of the image, the shape is
                [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
            pred_embs (np.array): Embedding results of the image, the shape is
                [N, 128] or [N, 512].

        Return:
            output_stracks_dict (dict(list)): The list contains information
                regarding the online_tracklets for the received image tensor.
        """
        self.frame_id += 1
        if self.frame_id == 1:
            STrack.init_count(self.num_classes)
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        pred_dets_dict = defaultdict(list)
        pred_embs_dict = defaultdict(list)

        # unify single and multi classes detection and embedding results
        for cls_id in range(self.num_classes):
            cls_idx = (pred_dets[:, 0:1] == cls_id).squeeze(-1)
            pred_dets_dict[cls_id] = pred_dets[cls_idx]
            if pred_embs is not None:
                pred_embs_dict[cls_id] = pred_embs[cls_idx]
            else:
                pred_embs_dict[cls_id] = None

        for cls_id in range(self.num_classes):
            """ Step 1: Get detections by class"""
            pred_dets_cls = pred_dets_dict[cls_id]
            pred_embs_cls = pred_embs_dict[cls_id]
            remain_inds = (pred_dets_cls[:, 1:2] > self.conf_thres).squeeze(-1)
            if remain_inds.sum() > 0:
                pred_dets_cls = pred_dets_cls[remain_inds]
                if pred_embs_cls is None:
                    # in original ByteTrack
                    detections = [
                        STrack(
                            STrack.tlbr_to_tlwh(tlbrs[2:6]),
                            tlbrs[1],
                            cls_id,
                            30,
                            temp_feat=None) for tlbrs in pred_dets_cls
                    ]
                else:
                    pred_embs_cls = pred_embs_cls[remain_inds]
                    detections = [
                        STrack(
                            STrack.tlbr_to_tlwh(tlbrs[2:6]), tlbrs[1], cls_id,
                            30, temp_feat) for (tlbrs, temp_feat) in
                        zip(pred_dets_cls, pred_embs_cls)
                    ]
            else:
                detections = []
            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    # previous tracks which are not active in the current frame are added in unconfirmed list
                    unconfirmed_dict[cls_id].append(track)
                else:
                    # Active tracks are added to the local list 'tracked_stracks'
                    tracked_tracks_dict[cls_id].append(track)
            """ Step 2: First association, with embedding"""
            # building tracking pool for the current frame
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = joint_stracks(
                tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KalmanFilter
            STrack.multi_predict(track_pool_dict[cls_id], self.motion)

            if pred_embs_cls is None:
                # in original ByteTrack
                dists = matching.iou_distance(track_pool_dict[cls_id],
                                              detections)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=self.match_thres)  # not self.tracked_thresh
            else:
                dists = matching.embedding_distance(
                    track_pool_dict[cls_id],
                    detections,
                    metric=self.metric_type)
                dists = matching.fuse_motion(
                    self.motion, dists, track_pool_dict[cls_id], detections)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=self.tracked_thresh)

            for i_tracked, idet in matches:
                # i_tracked is the id of the track and idet is the detection
                track = track_pool_dict[cls_id][i_tracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    # If the track is active, add the detection to the track
                    track.update(detections[idet], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    # We have obtained a detection from a track which is not active,
                    # hence put the track in refind_stracks list
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            # None of the steps below happen if there are no undetected tracks.
            """ Step 3: Second association, with IOU"""
            if self.use_byte:
                inds_low = pred_dets_dict[cls_id][:, 1:2] > self.low_conf_thres
                inds_high = pred_dets_dict[cls_id][:, 1:2] < self.conf_thres
                inds_second = np.logical_and(inds_low, inds_high).squeeze(-1)
                pred_dets_cls_second = pred_dets_dict[cls_id][inds_second]

                # association the untrack to the low score detections
                if len(pred_dets_cls_second) > 0:
                    if pred_embs_dict[cls_id] is None:
                        # in original ByteTrack
                        detections_second = [
                            STrack(
                                STrack.tlbr_to_tlwh(tlbrs[2:6]),
                                tlbrs[1],
                                cls_id,
                                30,
                                temp_feat=None)
                            for tlbrs in pred_dets_cls_second
                        ]
                    else:
                        pred_embs_cls_second = pred_embs_dict[cls_id][
                            inds_second]
                        detections_second = [
                            STrack(
                                STrack.tlbr_to_tlwh(tlbrs[2:6]), tlbrs[1],
                                cls_id, 30, temp_feat) for (tlbrs, temp_feat) in
                            zip(pred_dets_cls_second, pred_embs_cls_second)
                        ]
                else:
                    detections_second = []
                r_tracked_stracks = [
                    track_pool_dict[cls_id][i] for i in u_track
                    if track_pool_dict[cls_id][i].state == TrackState.Tracked
                ]
                dists = matching.iou_distance(r_tracked_stracks,
                                              detections_second)
                matches, u_track, u_detection_second = matching.linear_assignment(
                    dists, thresh=0.4)  # not r_tracked_thresh
            else:
                detections = [detections[i] for i in u_detection]
                r_tracked_stracks = []
                for i in u_track:
                    if track_pool_dict[cls_id][i].state == TrackState.Tracked:
                        r_tracked_stracks.append(track_pool_dict[cls_id][i])
                dists = matching.iou_distance(r_tracked_stracks, detections)

                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=self.r_tracked_thresh)

            for i_tracked, idet in matches:
                track = r_tracked_stracks[i_tracked]
                det = detections[
                    idet] if not self.use_byte else detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)
            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=self.unconfirmed_thresh)
            for i_tracked, idet in matches:
                unconfirmed_dict[cls_id][i_tracked].update(detections[idet],
                                                           self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][
                    i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)
            """ Step 4: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.motion, self.frame_id)
                activated_tracks_dict[cls_id].append(track)
            """ Step 5: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            self.tracked_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id]
                if t.state == TrackState.Tracked
            ]
            self.tracked_tracks_dict[cls_id] = joint_stracks(
                self.tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = joint_stracks(
                self.tracked_tracks_dict[cls_id], refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(
                self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(
                self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[
                cls_id] = remove_duplicate_stracks(
                    self.tracked_tracks_dict[cls_id],
                    self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [
                track for track in self.tracked_tracks_dict[cls_id]
                if track.is_activated
            ]

        return output_tracks_dict
