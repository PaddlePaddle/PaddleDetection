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

__all__ = ['TTFNet']


@register
class TTFNet(BaseArch):
    """
    TTFNet network, see https://arxiv.org/abs/1909.00700

    Args:
        backbone (object): backbone instance
        neck (object): 'TTFFPN' instance
        ttf_head (object): 'TTFHead' instance
        post_process (object): 'BBoxPostProcess' instance
    """

    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DarkNet',
                 neck='TTFFPN',
                 ttf_head='TTFHead',
                 post_process='BBoxPostProcess'):
        super(TTFNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.ttf_head = ttf_head
        self.post_process = post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        ttf_head = create(cfg['ttf_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "ttf_head": ttf_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        body_feats = self.neck(body_feats)
        hm, wh = self.ttf_head(body_feats)
        if self.training:
            return hm, wh
        else:
            bbox, bbox_num = self.post_process(hm, wh, self.inputs['im_shape'],
                                               self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self, ):
        loss = {}
        heatmap = self.inputs['ttf_heatmap']
        box_target = self.inputs['ttf_box_target']
        reg_weight = self.inputs['ttf_reg_weight']
        hm, wh = self._forward()
        head_loss = self.ttf_head.get_loss(hm, wh, heatmap, box_target,
                                           reg_weight)
        loss.update(head_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output
