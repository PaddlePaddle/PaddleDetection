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

__all__ = ['CenterTrack']


@register
class CenterTrack(BaseArch):
    """
    CenterTrack network, see http://arxiv.org/abs/2004.01177

    Args:
        detector (object): 'CenterNet' instance
        plugin_head (object): 'CenterTrackHead' instance
        tracker (object): 'CenterTracker' instance
    """
    __category__ = 'architecture'
    __shared__ = ['mot_metric']

    def __init__(self,
                 detector='CenterNet',
                 plugin_head='CenterTrackHead',
                 tracker='CenterTracker',
                 mot_metric=False):
        super(CenterTrack, self).__init__()
        self.detector = detector
        self.plugin_head = plugin_head
        self.tracker = tracker
        self.mot_metric = mot_metric

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        detector = create(cfg['detector'])
        detector_out_shape = detector.neck and detector.neck.out_shape or detector.backbone.out_shape

        kwargs = {'input_shape': detector_out_shape}
        plugin_head = create(cfg['plugin_head'], **kwargs)
        tracker = create(cfg['tracker'])

        return {
            'detector': detector,
            'plugin_head': plugin_head,
            'tracker': tracker,
        }

    def _forward(self):
        det_outs = self.detector(self.inputs)
        neck_feat = det_outs['neck_feat']
        if self.training:
            losses = {}
            for k, v in det_outs.items():
                if 'loss' not in k: continue
                losses.update({k: v})

            plugin_outs = self.plugin_head(neck_feat, self.inputs)
            for k, v in plugin_outs.items():
                if 'loss' not in k: continue
                losses.update({k: v})

            losses['loss'] = det_outs['det_loss'] + plugin_outs['plugin_loss']
            return losses
        else:
            if self.mot_metric:
                pred_dets, pred_embs = self.plugin_head(
                    neck_feat, self.inputs, det_outs['bbox'],
                    det_outs['bbox_inds'], det_outs['topk_clses'],
                    det_outs['topk_ys'], det_outs['topk_xs'])
                return pred_dets, pred_embs
            else:
                bbox = det_outs['bbox']
                bbox_num = det_outs['bbox_num']
                return {'bbox': bbox, 'bbox_num': bbox_num}

    def get_pred(self):
        return self._forward()

    def get_loss(self):
        return self._forward()
