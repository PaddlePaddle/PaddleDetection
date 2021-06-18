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
from ppdet.modeling.mot.utils import scale_coords
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['JDE']


@register
class JDE(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['metric']
    """
    JDE network, see https://arxiv.org/abs/1909.12605v1

    Args:
        detector (object): detector model instance
        reid (object): reid model instance
        tracker (object): tracker instance
        metric (str): 'MOTDet' for training and detection evaluation, 'ReID'
            for ReID embedding evaluation, or 'MOT' for multi object tracking
            evaluation.
    """

    def __init__(self,
                 detector='YOLOv3',
                 reid='JDEEmbeddingHead',
                 tracker='JDETracker',
                 metric='MOT'):
        super(JDE, self).__init__()
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.metric = metric

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
            if self.metric == 'MOTDet':
                det_results = {
                    'bbox': det_outs['bbox'],
                    'bbox_num': det_outs['bbox_num'],
                }
                return det_results

            elif self.metric == 'ReID':
                emb_feats = det_outs['emb_feats']
                embs_and_gts = self.reid(emb_feats, self.inputs, test_emb=True)
                return embs_and_gts

            elif self.metric == 'MOT':
                emb_feats = det_outs['emb_feats']
                emb_outs = self.reid(emb_feats, self.inputs)

                boxes_idx = det_outs['boxes_idx']
                bbox = det_outs['bbox']

                input_shape = self.inputs['image'].shape[2:]
                im_shape = self.inputs['im_shape']
                scale_factor = self.inputs['scale_factor']

                bbox[:, 2:] = scale_coords(bbox[:, 2:], input_shape, im_shape,
                                           scale_factor)

                nms_keep_idx = det_outs['nms_keep_idx']

                pred_dets = paddle.concat((bbox[:, 2:], bbox[:, 1:2]), axis=1)

                emb_valid = paddle.gather_nd(emb_outs, boxes_idx)
                pred_embs = paddle.gather_nd(emb_valid, nms_keep_idx)

                return pred_dets, pred_embs

            else:
                raise ValueError("Unknown metric {} for multi object tracking.".
                                 format(self.metric))

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
