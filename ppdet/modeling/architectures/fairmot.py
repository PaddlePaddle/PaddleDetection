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

__all__ = ['FairMOT']


@register
class FairMOT(BaseArch):
    __category__ = 'architecture'

    def __init__(self, detector, reid, loss):
        super(FairMOT, self).__init__()
        self.detector = detector
        self.reid = reid
        self.loss = loss

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        detector = create(cfg['detector'])

        kwargs = {'input_shape': detector.neck.out_shape}
        reid = create(cfg['reid'], **kwargs)
        loss = create(cfg['loss'])

        return {'detector': detector, 'reid': reid, 'loss': loss}

    def _forward(self):
        loss = dict()
        print('image in faitmot', self.inputs['image'].numpy().mean())
        det_outs = self.detector(self.inputs)
        neck_feat = det_outs['neck_feat']
        reid_outs = self.reid(neck_feat, self.inputs)
        if self.training:
            det_loss = det_outs['det_loss']
            reid_loss = reid_outs['reid_loss']
            loss = self.loss(det_loss, reid_loss)
            loss.update({
                'heatmap_loss': det_outs['heatmap_loss'],
                'size_loss': det_outs['size_loss'],
                'offset_loss': det_outs['offset_loss'],
                'reid_loss': reid_outs['reid_loss']
            })
            return loss
        else:
            heatmap = det_outs['heatmap']
            size = det_outs['size']
            offset = det_outs['offset']
            embedding = reid_outs['embedding']
            output = {
                'heatmap': heatmap,
                'size': size,
                'offset': offset,
                'embedding': embedding
            }
            return output

    def get_pred(self):
        output = self._forward()
        return output

    def get_loss(self):
        loss = self._forward()
        return loss
