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

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ppdet.modeling.mot.utils import Detection, get_crops, scale_coords, clip_box

__all__ = ['DeepSORT']


@register
class DeepSORT(BaseArch):
    """
    DeepSORT network, see https://arxiv.org/abs/1703.07402

    Args:
        detector (object): detector model instance
        reid (object): reid model instance
        tracker (object): tracker instance
    """
    __category__ = 'architecture'

    def __init__(self,
                 detector='YOLOv3',
                 reid='PCBPyramid',
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
        assert 'ori_image' in self.inputs
        load_dets = 'pred_bboxes' in self.inputs and 'pred_scores' in self.inputs

        ori_image = self.inputs['ori_image']
        input_shape = self.inputs['image'].shape[2:]
        im_shape = self.inputs['im_shape']
        scale_factor = self.inputs['scale_factor']

        if self.detector and not load_dets:
            outs = self.detector(self.inputs)
            if outs['bbox_num'] > 0:
                pred_bboxes = scale_coords(outs['bbox'][:, 2:], input_shape,
                                           im_shape, scale_factor)
                pred_scores = outs['bbox'][:, 1:2]
            else:
                pred_bboxes = []
                pred_scores = []
        else:
            pred_bboxes = self.inputs['pred_bboxes']
            pred_scores = self.inputs['pred_scores']

        if len(pred_bboxes) > 0:
            pred_bboxes = clip_box(pred_bboxes, input_shape, im_shape,
                                   scale_factor)
            bbox_tlwh = paddle.concat(
                (pred_bboxes[:, 0:2],
                 pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1),
                axis=1)

            crops, pred_scores = get_crops(
                pred_bboxes, ori_image, pred_scores, w=64, h=192)

            if len(crops) > 0:
                features = self.reid(paddle.to_tensor(crops))
                detections = [Detection(bbox_tlwh[i], conf, features[i])\
                                        for i, conf in enumerate(pred_scores)]
            else:
                detections = []
        else:
            detections = []

        self.tracker.predict()
        online_targets = self.tracker.update(detections)

        return online_targets

    def get_pred(self):
        return self._forward()
