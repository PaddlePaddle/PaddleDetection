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

__all__ = ['S2ANet']


@register
class S2ANet(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        's2anet_head',
        's2anet_bbox_post_process',
    ]

    def __init__(self, backbone, neck, s2anet_head, s2anet_bbox_post_process):
        """
        S2ANet, see https://arxiv.org/pdf/2008.09397.pdf

        Args:
            backbone (object): backbone instance
            neck (object): `FPN` instance
            s2anet_head (object): `S2ANetHead` instance
            s2anet_bbox_post_process (object): `S2ANetBBoxPostProcess` instance
        """
        super(S2ANet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.s2anet_head = s2anet_head
        self.s2anet_bbox_post_process = s2anet_bbox_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        s2anet_head = create(cfg['s2anet_head'], **kwargs)
        s2anet_bbox_post_process = create(cfg['s2anet_bbox_post_process'],
                                          **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "s2anet_head": s2anet_head,
            "s2anet_bbox_post_process": s2anet_bbox_post_process,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        self.s2anet_head(body_feats)
        if self.training:
            loss = self.s2anet_head.get_loss(self.inputs)
            total_loss = paddle.add_n(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            nms_pre = self.s2anet_bbox_post_process.nms_pre
            pred_scores, pred_bboxes = self.s2anet_head.get_prediction(nms_pre)

            # post_process
            pred_bboxes, bbox_num = self.s2anet_bbox_post_process(pred_scores,
                                                                  pred_bboxes)
            # rescale the prediction back to origin image
            pred_bboxes = self.s2anet_bbox_post_process.get_pred(
                pred_bboxes, bbox_num, im_shape, scale_factor)

            # output
            output = {'bbox': pred_bboxes, 'bbox_num': bbox_num}
            return output

    def get_loss(self, ):
        loss = self._forward()
        return loss

    def get_pred(self):
        output = self._forward()
        return output
