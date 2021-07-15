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
This code is borrow from https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py
"""

import paddle

from ..matching import jde_matching as matching
from .base_jde_tracker import TrackState, BaseTrack, STrack
from .base_jde_tracker import joint_stracks, sub_stracks, remove_duplicate_stracks

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['JDETracker']


@register
@serializable
class JDETracker(object):
    __inject__ = ['motion']
    """
    JDE tracker

    Args:
        det_thresh (float): threshold of detection score
        track_buffer (int): buffer for tracker
        min_box_area (int): min box area to filter out low quality boxes
        tracked_thresh (float): linear assignment threshold of tracked 
            stracks and detections
        r_tracked_thresh (float): linear assignment threshold of 
            tracked stracks and unmatched detections
        unconfirmed_thresh (float): linear assignment threshold of 
            unconfirmed stracks and unmatched detections
        motion (object): KalmanFilter instance
        conf_thres (float): confidence threshold for tracking
        metric_type (str): either "euclidean" or "cosine", the distance metric 
            used for measurement to track association.
    """

    def __init__(self,
                 det_thresh=0.3,
                 track_buffer=30,
                 min_box_area=200,
                 tracked_thresh=0.7,
                 r_tracked_thresh=0.5,
                 unconfirmed_thresh=0.7,
                 motion='KalmanFilter',
                 conf_thres=0,
                 metric_type='euclidean'):
        self.det_thresh = det_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.tracked_thresh = tracked_thresh
        self.r_tracked_thresh = r_tracked_thresh
        self.unconfirmed_thresh = unconfirmed_thresh
        self.motion = motion
        self.conf_thres = conf_thres
        self.metric_type = metric_type

        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.max_time_lost = 0
        # max_time_lost will be calculated: int(frame_rate / 30.0 * track_buffer)

    def update(self, pred_dets, pred_embs):
        """
        Processes the image frame and finds bounding box(detections).
        Associates the detection with corresponding tracklets and also handles
            lost, removed, refound and active tracklets.

        Args:
            pred_dets (Tensor): Detection results of the image, shape is [N, 5].
            pred_embs (Tensor): Embedding results of the image, shape is [N, 512].

        Return:
            output_stracks (list): The list contains information regarding the
                online_tracklets for the recieved image tensor.
        """
        self.frame_id += 1
        activated_starcks = []
        # for storing active tracks, for the current frame
        refind_stracks = []
        # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []
        # The tracks which are not obtained in the current frame but are not 
        # removed. (Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        remain_inds = paddle.nonzero(pred_dets[:, 4] > self.conf_thres)
        if remain_inds.shape[0] == 0:
            pred_dets = paddle.zeros([0, 1])
            pred_embs = paddle.zeros([0, 1])
        else:
            pred_dets = paddle.gather(pred_dets, remain_inds)
            pred_embs = paddle.gather(pred_embs, remain_inds)

        # Filter out the image with box_num = 0. pred_dets = [[0.0, 0.0, 0.0 ,0.0]]
        empty_pred = True if len(pred_dets) == 1 and paddle.sum(
            pred_dets) == 0.0 else False
        """ Step 1: Network forward, get detections & embeddings"""
        if len(pred_dets) > 0 and not empty_pred:
            pred_dets = pred_dets.numpy()
            pred_embs = pred_embs.numpy()
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30)
                for (tlbrs, f) in zip(pred_dets, pred_embs)
            ]
        else:
            detections = []
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)
        """ Step 2: First association, with embedding"""
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.motion)

        dists = matching.embedding_distance(
            strack_pool, detections, metric=self.metric_type)
        dists = matching.fuse_motion(self.motion, dists, strack_pool,
                                     detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.tracked_thresh)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active,
                # hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        """ Step 3: Second association, with IOU"""
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = []
        # This is container for stracks which were tracked till the previous
        # frame but no detection was found for it in the current frame.

        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.r_tracked_thresh)
        # matches is the list of detections which matched with corresponding
        # tracks by IOU distance method.

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.unconfirmed_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.motion, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks,
                                             activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks,
                                             refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [
            track for track in self.tracked_stracks if track.is_activated
        ]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format(
            [track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format(
            [track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format(
            [track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format(
            [track.track_id for track in removed_stracks]))

        return output_stracks
