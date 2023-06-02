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

import numpy as np
import paddle
import paddle.nn as nn
from scipy.special import softmax
from ppdet.modeling.lane_utils import Lane
from ppdet.modeling.losses import line_iou


class CLRNetPostProcess(object):
    """
    Args:
        input_shape (int): network input image size
        ori_shape (int): ori image shape of before padding
        scale_factor (float): scale factor of ori image
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self, img_w, ori_img_h, cut_height, conf_threshold, nms_thres,
                 max_lanes, num_points):
        self.img_w = img_w
        self.conf_threshold = conf_threshold
        self.nms_thres = nms_thres
        self.max_lanes = max_lanes
        self.num_points = num_points
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.ori_img_h = ori_img_h
        self.cut_height = cut_height

        self.prior_ys = paddle.linspace(
            start=1, stop=0, num=self.n_offsets).astype('float64')

    def predictions_to_pred(self, predictions):
        """
        Convert predictions to internal Lane structure for evaluation.
        """
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:].clone()
            start = min(
                max(0, int(round(lane[2].item() * self.n_strips))),
                self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            if start > 0:
                mask = ((lane_xs[:start] >= 0.) &
                        (lane_xs[:start] <= 1.)).cpu().detach().numpy()[::-1]
                mask = ~((mask.cumprod()[::-1]).astype(np.bool))
                lane_xs[:start][mask] = -2
            if end < len(self.prior_ys) - 1:
                lane_xs[end + 1:] = -2

            lane_ys = self.prior_ys[lane_xs >= 0].clone()
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(axis=0).astype('float64')
            lane_ys = lane_ys.flip(axis=0)

            lane_ys = (lane_ys *
                       (self.ori_img_h - self.cut_height) + self.cut_height
                       ) / self.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = paddle.stack(
                x=(lane_xs.reshape([-1, 1]), lane_ys.reshape([-1, 1])),
                axis=1).squeeze(axis=2)
            lane = Lane(
                points=points.cpu().numpy(),
                metadata={
                    'start_x': lane[3],
                    'start_y': lane[2],
                    'conf': lane[1]
                })
            lanes.append(lane)
        return lanes

    def lane_nms(self, predictions, scores, nms_overlap_thresh, top_k):
        """
        NMS for lane detection.
        predictions: paddle.Tensor [num_lanes,conf,y,x,lenght,72offsets] [12,77]
        scores: paddle.Tensor [num_lanes]
        nms_overlap_thresh: float
        top_k: int
        """
        # sort by scores to get idx
        idx = scores.argsort(descending=True)
        keep = []

        condidates = predictions.clone()
        condidates = condidates.index_select(idx)

        while len(condidates) > 0:
            keep.append(idx[0])
            if len(keep) >= top_k or len(condidates) == 1:
                break

            ious = []
            for i in range(1, len(condidates)):
                ious.append(1 - line_iou(
                    condidates[i].unsqueeze(0),
                    condidates[0].unsqueeze(0),
                    img_w=self.img_w,
                    length=15))
            ious = paddle.to_tensor(ious)

            mask = ious <= nms_overlap_thresh
            id = paddle.where(mask == False)[0]

            if id.shape[0] == 0:
                break
            condidates = condidates[1:].index_select(id)
            idx = idx[1:].index_select(id)
        keep = paddle.stack(keep)

        return keep

    def get_lanes(self, output, as_lanes=True):
        """
        Convert model output to lanes.
        """
        softmax = nn.Softmax(axis=1)
        decoded = []

        for predictions in output:
            if len(predictions) == 0:
                decoded.append([])
                continue
            threshold = self.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = paddle.concat(
                x=[nms_predictions[..., :4], nms_predictions[..., 5:]], axis=-1)

            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (
                self.img_w - 1)

            keep = self.lane_nms(
                nms_predictions[..., 5:],
                scores,
                nms_overlap_thresh=self.nms_thres,
                top_k=self.max_lanes)

            predictions = predictions.index_select(keep)

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            predictions[:, 5] = paddle.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)
        return decoded

    def __call__(self, lanes_list):
        lanes = self.get_lanes(lanes_list)
        return lanes
