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
        crops = self.inputs['crops']
        outs = {}
        outs['embeddings'] = self.reid(crops)
        return outs

    def get_pred(self):
        return self._forward()
