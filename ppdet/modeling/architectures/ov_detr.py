# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create
import numpy as np
__all__ = ['OVDETR']


@register
class OVDETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['backbone', 'head', 'post_process']

    def __init__(self,
                 backbone='ResNet',
                 head='OVDETRHead',
                 post_process='OVDETRPostProcess',
                 text_embedding=''):
        super(OVDETR, self).__init__()
        self.backbone = backbone
        self.head = head

        self.post_process = post_process
        self.text_embedding = paddle.to_tensor(
            np.load(text_embedding), dtype='float32')

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'head': head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        pad_mask = self.inputs.get('pad_mask', None)

        # DETR Head
        if self.training:

            ov_detr_losses = self.head(
                body_feats, pad_mask, self.text_embedding,
                self.inputs['gt_class'], self.inputs['gt_bbox'])
            ov_detr_losses.update({
                'loss': paddle.add_n([v for k, v in ov_detr_losses.items()])
            })
            return ov_detr_losses

        else:
            outputs = self.head(body_feats, pad_mask, self.text_embedding)
            results = self.post_process(
                outputs['pred_logits'], outputs['pred_boxes'],
                outputs['select_id'], self.inputs['im_shape'],
                self.inputs['scale_factor'])

            return results

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
