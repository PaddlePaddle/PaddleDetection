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
    __inject__ = ['head']

    def __init__(self, backbone, neck, head):
        """
        S2ANet, see https://arxiv.org/pdf/2008.09397.pdf

        Args:
            backbone (object): backbone instance
            neck (object): `FPN` instance
            head (object): `Head` instance
        """
        super(S2ANet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.s2anet_head = head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        head = create(cfg['head'], **kwargs)

        return {'backbone': backbone, 'neck': neck, "head": head}

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        if self.training:
            loss = self.s2anet_head(body_feats, self.inputs)
            return loss
        else:
            head_outs = self.s2anet_head(body_feats)
            # post_process
            bboxes, bbox_num = self.s2anet_head.get_bboxes(head_outs)
            # rescale the prediction back to origin image
            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bboxes = self.s2anet_head.get_pred(bboxes, bbox_num, im_shape,
                                               scale_factor)
            # output
            output = {'bbox': bboxes, 'bbox_num': bbox_num}
            return output

    def get_loss(self, ):
        loss = self._forward()
        return loss

    def get_pred(self):
        output = self._forward()
        return output
