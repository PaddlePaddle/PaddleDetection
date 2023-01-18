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

__all__ = ['PPYOLOE']


@register
class PPYOLOE(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='CSPResNet',
                 neck='CustomCSPPAN',
                 yolo_head='PPYOLOEHead',
                 post_process='BBoxPostProcess',
                 for_mot=False):
        """
        PPYOLOE network, see https://arxiv.org/abs/2203.16250

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            post_process (object): `BBoxPostProcess` instance
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(PPYOLOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)
            return yolo_losses
        else:
            yolo_head_outs = self.yolo_head(neck_feats)
            if self.post_process is not None:
                bbox, bbox_num = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])
            else:
                bbox, bbox_num = self.yolo_head.post_process(
                    yolo_head_outs, self.inputs['scale_factor'])
            output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
