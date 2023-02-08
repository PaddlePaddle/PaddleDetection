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
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.losses.iou_loss import GIoULoss
from .utils import QFLv2

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'SSODFCOSLoss',
    'SSODPPYOLOELoss',
]


@register
class SSODFCOSLoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(SSODFCOSLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, student_head_outs, teacher_head_outs, train_cfg):
        # for semi-det distill
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


@register
class SSODPPYOLOELoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(SSODPPYOLOELoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, student_head_outs, teacher_head_outs, train_cfg):
        # for semi-det distill
        # student_probs: already sigmoid
        student_probs, student_deltas, student_dfl = student_head_outs
        teacher_probs, teacher_deltas, teacher_dfl = teacher_head_outs
        bs, l, nc = student_probs.shape[:]  # bs, l, num_classes
        bs, l, _, reg_ch = student_dfl.shape[:]  # bs, l, 4, reg_ch
        student_probs = student_probs.reshape([-1, nc])
        teacher_probs = teacher_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_deltas = teacher_deltas.reshape([-1, 4])
        student_dfl = student_dfl.reshape([-1, 4, reg_ch])
        teacher_dfl = teacher_dfl.reshape([-1, 4, reg_ch])

        ratio = train_cfg.get('ratio', 0.01)

        # for contrast loss
        curr_iter = train_cfg['curr_iter']
        st_iter = train_cfg['st_iter']
        if curr_iter == st_iter + 1:
            # start semi-det training
            self.queue_ptr = 0
            self.queue_size = int(bs * l * ratio)
            self.queue_feats = paddle.zeros([self.queue_size, nc])
            self.queue_probs = paddle.zeros([self.queue_size, nc])
        contrast_loss_cfg = train_cfg['contrast_loss']
        temperature = contrast_loss_cfg.get('temperature', 0.2)
        alpha = contrast_loss_cfg.get('alpha', 0.9)
        smooth_iter = contrast_loss_cfg.get('smooth_iter', 100) + st_iter

        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_probs.shape[0] * ratio)
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals,
                                                   teacher_probs.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

            # for contrast loss
            probs = teacher_probs[b_mask].detach()
            if curr_iter > smooth_iter:  # memory-smoothing
                A = paddle.exp(
                    paddle.mm(teacher_probs[b_mask], self.queue_probs.t()) /
                    temperature)
                A = A / A.sum(1, keepdim=True)
                probs = alpha * probs + (1 - alpha) * paddle.mm(
                    A, self.queue_probs)
            n = student_probs[b_mask].shape[0]
            # update memory bank
            self.queue_feats[self.queue_ptr:self.queue_ptr +
                             n, :] = teacher_probs[b_mask].detach()
            self.queue_probs[self.queue_ptr:self.queue_ptr +
                             n, :] = teacher_probs[b_mask].detach()
            self.queue_ptr = (self.queue_ptr + n) % self.queue_size

        # embedding similarity
        sim = paddle.exp(
            paddle.mm(student_probs[b_mask], teacher_probs[b_mask].t()) / 0.2)
        sim_probs = sim / sim.sum(1, keepdim=True)
        # pseudo-label graph with self-loop
        Q = paddle.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= 0.5).astype('float32')
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        # contrastive loss
        loss_contrast = -(paddle.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()

        # distill_loss_cls
        loss_cls = QFLv2(
            student_probs, teacher_probs, weight=mask, reduction="sum") / fg_num

        # distill_loss_iou
        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            -1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            -1)
        iou_loss = GIoULoss(reduction='mean')
        loss_iou = iou_loss(inputs, targets)

        # distill_loss_dfl
        loss_dfl = F.cross_entropy(
            student_dfl[b_mask].reshape([-1, reg_ch]),
            teacher_dfl[b_mask].reshape([-1, reg_ch]),
            soft_label=True,
            reduction='mean')

        return {
            "distill_loss_cls": loss_cls,
            "distill_loss_iou": loss_iou,
            "distill_loss_dfl": loss_dfl,
            "distill_loss_contrast": loss_contrast,
            "fg_sum": fg_num,
        }
