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
    """
    CenterNet network, see http://arxiv.org/abs/1904.07850

    Args:
        backbone (object): backbone instance
        neck (object): 'CenterDLAFPN' instance
        head (object): 'CenterHead' instance
        post_process (object): 'BBoxPostProcess' instance

    """

    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self, detector, reid, tracker, loss):
        super(FairMOT, self).__init__()
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.loss = loss

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        detector = create(cfg['detector'])

        kwargs = {'input_shape': detector.neck.out_shape}
        reid = create(cfg['reid'], **kwargs)
        loss = create(cfg['loss'])
        tracker = create(cfg['tracker'])

        return {
            'detector': detector,
            'reid': reid,
            'loss': loss,
            'tracker': tracker
        }

    def _forward(self):
        loss = dict()
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
            embedding = reid_outs['embedding']
            bbox_inds = det_outs['bbox_inds']
            embedding = paddle.transpose(embedding, [0, 2, 3, 1])
            embedding = paddle.reshape(embedding,
                                       [-1, paddle.shape(embedding)[-1]])
            id_feature = paddle.gather(embedding, bbox_inds)
            #output = {'dets': det_outs['bbox'], 'id_feature': id_feature}
            dets = det_outs['bbox']
            id_feature = id_feature
            # Note: the tracker only considers batch_size=1 and num_classses=1
            online_targets = self.tracker.update(dets, id_feature)
            return online_targets

    def get_pred(self):
        output = self._forward()
        return output

    def get_loss(self):
        loss = self._forward()
        return loss
