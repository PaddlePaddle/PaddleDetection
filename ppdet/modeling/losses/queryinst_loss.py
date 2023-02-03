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
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.losses.iou_loss import GIoULoss
from .sparsercnn_loss import HungarianMatcher

__all__ = ['QueryInstLoss']


@register
class QueryInstLoss(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 class_weight=2.0,
                 l1_weight=5.0,
                 giou_weight=2.0,
                 mask_weight=8.0):
        super(QueryInstLoss, self).__init__()

        self.num_classes = num_classes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.loss_weights = {
            "loss_cls": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
            "loss_mask": mask_weight
        }
        self.giou_loss = GIoULoss(eps=1e-6, reduction='sum')

        self.matcher = HungarianMatcher(focal_loss_alpha, focal_loss_gamma,
                                        class_weight, l1_weight, giou_weight)

    def loss_classes(self, class_logits, targets, indices, avg_factor):
        tgt_labels = paddle.full(
            class_logits.shape[:2], self.num_classes, dtype='int32')

        if sum(len(v['labels']) for v in targets) > 0:
            tgt_classes = paddle.concat([
                paddle.gather(
                    tgt['labels'], tgt_idx, axis=0)
                for tgt, (_, tgt_idx) in zip(targets, indices)
            ])
            batch_idx, src_idx = self._get_src_permutation_idx(indices)
            for i, (batch_i, src_i) in enumerate(zip(batch_idx, src_idx)):
                tgt_labels[int(batch_i), int(src_i)] = tgt_classes[i]

        tgt_labels = tgt_labels.flatten(0, 1).unsqueeze(-1)

        tgt_labels_onehot = paddle.cast(
            tgt_labels == paddle.arange(0, self.num_classes), dtype='float32')
        tgt_labels_onehot.stop_gradient = True

        src_logits = class_logits.flatten(0, 1)

        loss_cls = F.sigmoid_focal_loss(
            src_logits,
            tgt_labels_onehot,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction='sum') / avg_factor
        losses = {'loss_cls': loss_cls * self.loss_weights['loss_cls']}
        return losses

    def loss_bboxes(self, bbox_pred, targets, indices, avg_factor):
        bboxes = paddle.concat([
            paddle.gather(
                src, src_idx, axis=0)
            for src, (src_idx, _) in zip(bbox_pred, indices)
        ])

        tgt_bboxes = paddle.concat([
            paddle.gather(
                tgt['boxes'], tgt_idx, axis=0)
            for tgt, (_, tgt_idx) in zip(targets, indices)
        ])
        tgt_bboxes.stop_gradient = True

        im_shapes = paddle.concat([tgt['img_whwh_tgt'] for tgt in targets])
        bboxes_norm = bboxes / im_shapes
        tgt_bboxes_norm = tgt_bboxes / im_shapes

        loss_giou = self.giou_loss(bboxes, tgt_bboxes) / avg_factor
        loss_bbox = F.l1_loss(
            bboxes_norm, tgt_bboxes_norm, reduction='sum') / avg_factor
        losses = {
            'loss_bbox': loss_bbox * self.loss_weights['loss_bbox'],
            'loss_giou': loss_giou * self.loss_weights['loss_giou']
        }
        return losses

    def loss_masks(self, pos_bbox_pred, mask_logits, targets, indices,
                   avg_factor):
        tgt_segm = [
            paddle.gather(
                tgt['gt_segm'], tgt_idx, axis=0)
            for tgt, (_, tgt_idx) in zip(targets, indices)
        ]

        tgt_masks = []
        for i in range(len(indices)):
            gt_segm = tgt_segm[i].unsqueeze(1)
            if len(gt_segm) == 0:
                continue
            boxes = pos_bbox_pred[i]
            boxes[:, 0::2] = paddle.clip(
                boxes[:, 0::2], min=0, max=gt_segm.shape[3])
            boxes[:, 1::2] = paddle.clip(
                boxes[:, 1::2], min=0, max=gt_segm.shape[2])
            boxes_num = paddle.to_tensor([1] * len(boxes), dtype='int32')
            gt_mask = paddle.vision.ops.roi_align(
                gt_segm,
                boxes,
                boxes_num,
                output_size=mask_logits.shape[-2:],
                aligned=True)
            tgt_masks.append(gt_mask)
        tgt_masks = paddle.concat(tgt_masks).squeeze(1)
        tgt_masks = paddle.cast(tgt_masks >= 0.5, dtype='float32')
        tgt_masks.stop_gradient = True

        tgt_labels = paddle.concat([
            paddle.gather(
                tgt['labels'], tgt_idx, axis=0)
            for tgt, (_, tgt_idx) in zip(targets, indices)
        ])

        mask_label = F.one_hot(tgt_labels, self.num_classes).unsqueeze([2, 3])
        mask_label = paddle.expand_as(mask_label, mask_logits)
        mask_label.stop_gradient = True

        src_masks = paddle.gather_nd(mask_logits, paddle.nonzero(mask_label))
        shape = mask_logits.shape
        src_masks = paddle.reshape(src_masks, [shape[0], shape[2], shape[3]])
        src_masks = F.sigmoid(src_masks)

        X = src_masks.flatten(1)
        Y = tgt_masks.flatten(1)
        inter = paddle.sum(X * Y, 1)
        union = paddle.sum(X * X, 1) + paddle.sum(Y * Y, 1)
        dice = (2 * inter) / (union + 2e-5)

        loss_mask = (1 - dice).sum() / avg_factor
        losses = {'loss_mask': loss_mask * self.loss_weights['loss_mask']}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = paddle.concat(
            [paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx
