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

__all__ = ['CenterNet']


@register
class CenterNet(BaseArch):
    __category__ = 'architecture'

    def __init__(self, backbone='DLA', neck='FairDLAFPN', head='CenterHead'):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        #import numpy as np
        #print('image in centernet', self.inputs['image'].numpy().mean())
        body_feats = self.backbone(self.inputs)
        #for i in range(len(body_feats)):
        #    print('-----------------base {}'.format(i), np.mean(body_feats[i].numpy()))
        neck_feat = self.neck(body_feats)
        head_out = self.head(neck_feat[-1], self.inputs)
        head_out.update({'neck_feat': neck_feat[-1]})
        return head_out

    def get_pred(self):
        return self._forward()

    def get_loss(self):
        return self._forward()
