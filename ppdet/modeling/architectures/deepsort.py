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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ppdet.mot.mot_utils import Detection
from ppdet.mot.mot_utils import get_crops, scale_coords, clip_box

__all__ = ['DeepSORT']


@register
class DeepSORT(BaseArch):
    __category__ = 'architecture'

    def __init__(self,
                 detector='YOLOv3',
                 reid='PCB_plus_dropout_pyramid',
                 tracker='DeepSORTTracker'):
        super(DeepSORT, self).__init__()
        self.detector = detector
        self.reid = reid
        self.tracker = tracker

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        if cfg['detector'] != 'None':
            detector = create(cfg['detector'])
        else:
            detector = None
        reid = create(cfg['reid'])
        tracker = create(cfg['tracker'])

        return {
            "detector": detector,
            "reid": reid,
            "tracker": tracker,
        }

    def _forward(self):
        load_dets = 'bbox_xyxy' in self.inputs and 'pred_scores' in self.inputs

        img0 = self.inputs['img0'].numpy()
        img0_shape = self.inputs['img0_shape'].numpy()[0]
        img_size = self.tracker.img_size

        if self.detector and not load_dets:
            outs = self.detector(self.inputs)
            bbox_num = outs['bbox_num']
            if bbox_num > 0:
                bbox_xyxy = outs['bbox'][:, 2:].numpy()
                bbox_xyxy = scale_coords(img_size, bbox_xyxy,
                                         img0_shape).round()
                pred_scores = outs['bbox'][:, 1:2].numpy()
            else:
                bbox_xyxy = []
                pred_scores = []
        else:
            bbox_xyxy = self.inputs['bbox_xyxy']
            pred_scores = self.inputs['pred_scores']

        if len(bbox_xyxy) > 0:
            bbox_xyxy = clip_box(bbox_xyxy, img0_shape)
            bbox_tlwh = np.hstack(
                (bbox_xyxy[:, 0:2], bbox_xyxy[:, 2:4] - bbox_xyxy[:, 0:2] + 1))
            crops, pred_scores = get_crops(
                bbox_xyxy, img0, pred_scores, w=64, h=192)
            if len(crops) > 0:
                features = self.reid(paddle.to_tensor(crops))
                detections = [Detection(bbox_tlwh[i], conf, features[i])\
                                        for i,conf in enumerate(pred_scores)]
            else:
                detections = []
        else:
            detections = []

        self.tracker.predict()
        online_targets = self.tracker.update(detections)

        return online_targets

    def get_pred(self):
        return self._forward()
