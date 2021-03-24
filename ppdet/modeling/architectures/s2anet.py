# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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
import numpy as np

__all__ = ['S2ANet']


@register
class S2ANet(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        's2anet_head',
        's2anet_bbox_post_process',
        's2anet_anchor_assigner',
    ]

    def __init__(self,
                 backbone,
                 neck,
                 s2anet_head,
                 s2anet_bbox_post_process,
                 s2anet_anchor_assigner=None):
        """
        """
        super(S2ANet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.s2anet_head = s2anet_head
        self.s2anet_bbox_post_process = s2anet_bbox_post_process
        self.s2anet_anchor_assigner = s2anet_anchor_assigner

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        s2anet_head = create(cfg['s2anet_head'], **kwargs)
        s2anet_bbox_post_process = create(cfg['s2anet_bbox_post_process'], **kwargs)
        s2anet_anchor_assigner = create(cfg['s2anet_anchor_assigner'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "s2anet_head": s2anet_head,
            "s2anet_bbox_post_process": s2anet_bbox_post_process,
            "s2anet_anchor_assigner": s2anet_anchor_assigner,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # fam_list  odm_list
        s2anet_head_out = self.s2anet_head(body_feats)

        return s2anet_head_out

    def get_loss(self, ):
        s2anet_head_out = self._forward()

        loss = self.s2anet_head.get_loss(self.inputs, s2anet_head_out, self.s2anet_anchor_assigner)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        s2anet_head_out = self._forward()

        im_shape = self.inputs['im_shape']
        scale_factor = self.inputs['scale_factor']
        nms_pre = self.s2anet_bbox_post_process.nms_pre
        pred_scores, pred_bboxes = self.s2anet_head.get_prediction(s2anet_head_out, nms_pre)

        # post_process
        bbox_pred, bbox_num, index = self.s2anet_bbox_post_process.get_nms_result(pred_scores, pred_bboxes)
        # pred_result = self.s2anet_bbox_post_process.get_pred(bbox_pred, bbox_num, im_shape, scale_factor)

        output = {
            'bbox': bbox_pred,
            'bbox_num': bbox_num
        }
        return output
