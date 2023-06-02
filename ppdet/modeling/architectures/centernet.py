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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['CenterNet']


@register
class CenterNet(BaseArch):
    """
    CenterNet network, see http://arxiv.org/abs/1904.07850

    Args:
        backbone (object): backbone instance
        neck (object): FPN instance, default use 'CenterNetDLAFPN'
        head (object): 'CenterNetHead' instance
        post_process (object): 'CenterNetPostProcess' instance
        for_mot (bool): whether return other features used in tracking model

    """
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['for_mot']

    def __init__(self,
                 backbone,
                 neck='CenterNetDLAFPN',
                 head='CenterNetHead',
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
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        head = create(cfg['head'], **kwargs)

        return {'backbone': backbone, 'neck': neck, "head": head}

    def _forward(self):
        neck_feat = self.backbone(self.inputs)
        if self.neck is not None:
            neck_feat = self.neck(neck_feat)
        head_out = self.head(neck_feat, self.inputs)
        if self.for_mot:
            head_out.update({'neck_feat': neck_feat})
        elif self.training:
            head_out['loss'] = head_out.pop('det_loss')
        return head_out

    def get_pred(self):
        head_out = self._forward()
        bbox, bbox_num, bbox_inds, topk_clses, topk_ys, topk_xs = self.post_process(
            head_out['heatmap'],
            head_out['size'],
            head_out['offset'],
            im_shape=self.inputs['im_shape'],
            scale_factor=self.inputs['scale_factor'])

        if self.for_mot:
            output = {
                "bbox": bbox,
                "bbox_num": bbox_num,
                "bbox_inds": bbox_inds,
                "topk_clses": topk_clses,
                "topk_ys": topk_ys,
                "topk_xs": topk_xs,
                "neck_feat": head_out['neck_feat']
            }
        else:
            output = {"bbox": bbox, "bbox_num": bbox_num}
        return output

    def get_loss(self):
        return self._forward()
