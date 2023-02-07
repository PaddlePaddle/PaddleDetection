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
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..ssod_utils import permute_to_N_HWA_K, QFLv2
from ..losses import GIoULoss

__all__ = ['FCOS']


@register
class FCOS(BaseArch):
    """
    FCOS network, see https://arxiv.org/abs/1904.01355

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        fcos_head (object): 'FCOSHead' instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, neck='FPN', fcos_head='FCOSHead'):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fcos_head = fcos_head
        self.is_teacher = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        fcos_head = create(cfg['fcos_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "fcos_head": fcos_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)

        self.is_teacher = self.inputs.get('is_teacher', False)
        if self.training or self.is_teacher:
            losses = self.fcos_head(fpn_feats, self.inputs)
            return losses
        else:
            fcos_head_outs = self.fcos_head(fpn_feats)
            bbox_pred, bbox_num = self.fcos_head.post_process(
                fcos_head_outs, self.inputs['scale_factor'])
            return {'bbox': bbox_pred, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_box', 'loss_quality']

    def get_ssod_distill_loss(self, student_head_outs, teacher_head_outs,
                              train_cfg):
        student_logits, student_deltas, student_quality = student_head_outs
        teacher_logits, teacher_deltas, teacher_quality = teacher_head_outs
        nc = student_logits[0].shape[1]

        student_logits = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, nc])
                for _ in student_logits
            ],
            axis=0)
        teacher_logits = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, nc])
                for _ in teacher_logits
            ],
            axis=0)

        student_deltas = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 4])
                for _ in student_deltas
            ],
            axis=0)
        teacher_deltas = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 4])
                for _ in teacher_deltas
            ],
            axis=0)

        student_quality = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 1])
                for _ in student_quality
            ],
            axis=0)
        teacher_quality = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 1])
                for _ in teacher_quality
            ],
            axis=0)

        ratio = train_cfg.get('ratio', 0.01)
        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_logits.shape[0] * ratio)
            teacher_probs = F.sigmoid(teacher_logits)
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals,
                                                   teacher_logits.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0

        # distill_loss_cls
        loss_logits = QFLv2(
            F.sigmoid(student_logits),
            teacher_probs,
            weight=mask,
            reduction="sum") / fg_num

        # distill_loss_box
        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            axis=-1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            axis=-1)
        iou_loss = GIoULoss(reduction='mean')
        loss_deltas = iou_loss(inputs, targets)

        # distill_loss_quality
        loss_quality = F.binary_cross_entropy(
            F.sigmoid(student_quality[b_mask]),
            F.sigmoid(teacher_quality[b_mask]),
            reduction='mean')

        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_box": loss_deltas,
            "distill_loss_quality": loss_quality,
            "fg_sum": fg_num,
        }
