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

__all__ = ['JDE']


@register
class JDE(BaseArch):
    """
    JDE network, see https://arxiv.org/abs/1909.12605v1

    Args:
        detector (object): detector model instance
        reid (object): reid model instance
        tracker (object): tracker instance
    """
    __category__ = 'architecture'

    def __init__(self,
                 detector='YOLOv3',
                 reid='JEDEmbeddingHead',
                 tracker='JDETracker',
                 test_emb=False,
                 test_track=False):
        super(JDE, self).__init__()
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.test_emb = test_emb
        self.test_track = test_track

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        detector = create(cfg['detector'])
        kwargs = {'input_shape': detector.neck.out_shape}

        reid = create(cfg['reid'], **kwargs)

        tracker = create(cfg['tracker'])

        return {
            "detector": detector,
            "reid": reid,
            "tracker": tracker,
        }

    def _forward(self):
        det_outs = self.detector(self.inputs)

        if self.training:
            emb_feats = det_outs['emb_feats']
            loss_confs = det_outs['det_losses']['loss_confs']
            loss_boxes = det_outs['det_losses']['loss_boxes']
            jde_losses = self.reid(emb_feats, self.inputs, loss_confs,
                                   loss_boxes)
            return jde_losses
        else:
            if self.test_emb:
                emb_feats = det_outs['emb_feats']
                embs_and_gts = self.reid(emb_feats, self.inputs, test_emb=True)
                return embs_and_gts

            elif self.test_track:
                emb_feats = det_outs['emb_feats']
                emb_outs = self.reid(emb_feats, self.inputs)

                boxes_idx = det_outs['boxes_idx']
                bbox = det_outs['bbox']
                nms_keep_idx = det_outs['nms_keep_idx']

                pred_dets = paddle.concat((bbox[:, 2:], bbox[:, 1:2]), axis=1)

                emb_valid = paddle.gather_nd(emb_outs, boxes_idx)
                pred_embs = paddle.gather_nd(emb_valid, nms_keep_idx)
                scale_factor = self.inputs['scale_factor']

                online_targets = self.tracker.update(pred_dets, pred_embs,
                                                     scale_factor)
                return online_targets

            else:
                det_results = {
                    'bbox': det_outs['bbox'],
                    'bbox_num': det_outs['bbox_num'],
                }
                return det_results

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
