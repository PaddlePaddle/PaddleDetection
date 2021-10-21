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
    """
    CenterNet network, see http://arxiv.org/abs/1904.07850

    Args:
        backbone (object): backbone instance
        neck (object): 'CenterDLAFPN' instance
        head (object): 'CenterHead' instance
        post_process (object): 'CenterNetPostProcess' instance
        for_mot (bool): whether return other features used in tracking model

    """
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DLA',
                 neck='CenterDLAFPN',
                 head='CenterHead',
                 post_process='CenterNetPostProcess',
                 for_mot=False):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.post_process = post_process
        self.for_mot = for_mot

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {'backbone': backbone, 'neck': neck, "head": head}

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feat = self.neck(body_feats)
        head_out = self.head(neck_feat, self.inputs)
        if self.for_mot:
            head_out.update({'neck_feat': neck_feat})
        return head_out

    def get_pred(self):
        head_out = self._forward()
        if self.for_mot:
            bbox, bbox_inds = self.post_process(
                head_out['heatmap'],
                head_out['size'],
                head_out['offset'],
                im_shape=self.inputs['im_shape'],
                scale_factor=self.inputs['scale_factor'])
            output = {
                "bbox": bbox,
                "bbox_inds": bbox_inds,
                "neck_feat": head_out['neck_feat']
            }
        else:
            bbox, bbox_num = self.post_process(
                head_out['heatmap'],
                head_out['size'],
                head_out['offset'],
                im_shape=self.inputs['im_shape'],
                scale_factor=self.inputs['scale_factor'])
            output = {
                "bbox": bbox,
                "bbox_num": bbox_num,
            }
        return output

    def get_loss(self):
        return self._forward()
