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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['ByteTrack']


@register
class ByteTrack(BaseArch):
    """
    ByteTrack network, see https://arxiv.org/abs/2110.06864

    Args:
        detector (object): detector model instance
        reid (object): reid model instance, default None
        tracker (object): tracker instance
    """
    __category__ = 'architecture'

    def __init__(self,
                 detector='YOLOX',
                 reid=None,
                 tracker='JDETracker'):
        super(ByteTrack, self).__init__()
        self.detector = detector
        self.reid = reid
        self.tracker = tracker

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        detector = create(cfg['detector'])

        if cfg['reid'] != 'None':
            reid = create(cfg['reid'])
        else:
            reid = None

        tracker = create(cfg['tracker'])

        return {
            "detector": detector,
            "reid": reid,
            "tracker": tracker,
        }

    def _forward(self):
        det_outs = self.detector(self.inputs)

        if self.training:
            return det_outs
        else:
            if self.reid is not None:
                assert 'crops' in self.inputs
                crops = self.inputs['crops']
                pred_embs = self.reid(crops)
            else:
                pred_embs = None
            det_outs['embeddings'] = pred_embs
            return det_outs

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

