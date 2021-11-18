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
    FairMOT network, see http://arxiv.org/abs/2004.01888

    Args:
        detector (object): 'CenterNet' instance
        reid (object): 'FairMOTEmbeddingHead' instance
        tracker (object): 'JDETracker' instance
        loss (object): 'FairMOTLoss' instance

    """

    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self,
                 detector='CenterNet',
                 reid='FairMOTEmbeddingHead',
                 tracker='JDETracker',
                 loss='FairMOTLoss'):
        super(FairMOT, self).__init__()
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.loss = loss

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        detector = create(cfg['detector'])
        detector_out_shape = detector.neck and detector.neck.out_shape or detector.backbone.out_shape

        kwargs = {'input_shape': detector_out_shape}
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
        # det_outs keys:
        # train: neck_feat, det_loss, heatmap_loss, size_loss, offset_loss (optional: iou_loss)
        # eval/infer: neck_feat, bbox, bbox_inds
        det_outs = self.detector(self.inputs)
        neck_feat = det_outs['neck_feat']
        if self.training:
            reid_loss = self.reid(neck_feat, self.inputs)

            det_loss = det_outs['det_loss']
            loss = self.loss(det_loss, reid_loss)
            for k, v in det_outs.items():
                if 'loss' not in k:
                    continue
                loss.update({k: v})
            loss.update({'reid_loss': reid_loss})
            return loss
        else:
            pred_dets, pred_embs = self.reid(
                neck_feat, self.inputs, det_outs['bbox'], det_outs['bbox_inds'],
                det_outs['topk_clses'])
            return pred_dets, pred_embs

    def get_pred(self):
        output = self._forward()
        return output

    def get_loss(self):
        loss = self._forward()
        return loss
