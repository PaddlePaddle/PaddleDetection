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
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['FCOS']


@register
class FCOS(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'backbone',
        'neck',
        'fcos_head',
        'fcos_post_process',
    ]

    def __init__(self,
                 backbone,
                 neck,
                 fcos_head='FCOSHead',
                 fcos_post_process='FCOSPostProcess'):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fcos_head = fcos_head
        self.fcos_post_process = fcos_post_process

    def model_arch(self, ):
        body_feats = self.backbone(self.inputs)

        fpn_feats, spatial_scale = self.neck(body_feats)

        self.fcos_head_outs = self.fcos_head(fpn_feats, self.training)

        if not self.training:
            self.bboxes = self.fcos_post_process(self.fcos_head_outs,
                                                 self.inputs['scale_factor'])

    def get_loss(self, ):
        loss = {}
        tag_labels, tag_bboxes, tag_centerness = [], [], []
        for i in range(len(self.fcos_head.fpn_stride)):
            # reg_target, labels, scores, centerness
            k_lbl = 'labels{}'.format(i)
            if k_lbl in self.inputs:
                tag_labels.append(self.inputs[k_lbl])
            k_box = 'reg_target{}'.format(i)
            if k_box in self.inputs:
                tag_bboxes.append(self.inputs[k_box])
            k_ctn = 'centerness{}'.format(i)
            if k_ctn in self.inputs:
                tag_centerness.append(self.inputs[k_ctn])

        loss_fcos = self.fcos_head.get_loss(self.fcos_head_outs, tag_labels,
                                            tag_bboxes, tag_centerness)
        loss.update(loss_fcos)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox, bbox_num = self.bboxes
        output = {
            'bbox': bbox,
            'bbox_num': bbox_num,
        }
        return output
