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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['QueryInst']


@register
class QueryInst(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head,
                 post_process='SparsePostProcess'):
        super(QueryInst, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.post_process = post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        roi_head = create(cfg['roi_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            'rpn_head': rpn_head,
            "roi_head": roi_head
        }

    def _forward(self, targets=None):
        features = self.backbone(self.inputs)
        features = self.neck(features)

        proposal_bboxes, proposal_features = self.rpn_head(self.inputs[
            'img_whwh'])
        outputs = self.roi_head(features, proposal_bboxes, proposal_features,
                                targets)

        if self.training:
            return outputs
        else:
            bbox_pred, bbox_num, mask_pred = self.post_process(
                outputs['class_logits'], outputs['bbox_pred'],
                self.inputs['scale_factor_whwh'], self.inputs['ori_shape'],
                outputs['mask_logits'])
            return bbox_pred, bbox_num, mask_pred

    def get_loss(self):
        targets = []
        for i in range(len(self.inputs['img_whwh'])):
            boxes = self.inputs['gt_bbox'][i]
            labels = self.inputs['gt_class'][i].squeeze(-1)
            img_whwh = self.inputs['img_whwh'][i]
            if boxes.shape[0] != 0:
                img_whwh_tgt = img_whwh.unsqueeze(0).tile([boxes.shape[0], 1])
            else:
                img_whwh_tgt = paddle.zeros_like(boxes)
            gt_segm = self.inputs['gt_segm'][i].astype('float32')
            targets.append({
                'boxes': boxes,
                'labels': labels,
                'img_whwh': img_whwh,
                'img_whwh_tgt': img_whwh_tgt,
                'gt_segm': gt_segm
            })
        losses = self._forward(targets)
        losses.update({'loss': sum(losses.values())})
        return losses

    def get_pred(self):
        bbox_pred, bbox_num, mask_pred = self._forward()
        return {'bbox': bbox_pred, 'bbox_num': bbox_num, 'mask': mask_pred}
