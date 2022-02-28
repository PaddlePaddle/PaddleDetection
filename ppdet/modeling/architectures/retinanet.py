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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle

__all__ = ['RetinaNet']

@register
class RetinaNet(BaseArch):
    __category__ = 'architecture'

    def __init__(self,
                 backbone,
                 neck,
                 head):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)
        head = create(cfg['head'])
        return {
            'backbone': backbone,
            'neck': neck,
            'head': head}

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats)
        head_outs = self.head(neck_feats)
        if not self.training:
            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bboxes, bbox_num = self.head.post_process(head_outs, im_shape, scale_factor)
            return bboxes, bbox_num
        return head_outs

    def get_loss(self):
        loss = dict()
        head_outs = self._forward()
        loss_retina = self.head.get_loss(head_outs, self.inputs)
        loss.update(loss_retina)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update(loss=total_loss)
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = dict(bbox=bbox_pred, bbox_num=bbox_num)
        return output
