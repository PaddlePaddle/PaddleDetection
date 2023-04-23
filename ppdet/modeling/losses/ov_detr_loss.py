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

import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .iou_loss import GIoULoss
from ..transformers import bbox_cxcywh_to_xyxy, sigmoid_focal_loss

__all__ = ['OVDETRLoss']


@register
class OVDETRLoss(nn.Layer):
    __inject__ = ['matcher']

    def __init__(self,
                 aux_loss=True,
                 num_classes=91,
                 dec_layers=6,
                 loss_coeff={
                     'class': 3,
                     'bbox': 5,
                     'giou': 2,
                     'embed': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 losses=["class", "boxes", "embed"],
                 use_focal_loss=True,
                 focal_alpha=0.25,
                 matcher='OVHungarianMatcher'):
        super(OVDETRLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.losses = losses
        self.use_focal_loss = use_focal_loss
        self.giou_loss = GIoULoss()
        self.focal_alpha = focal_alpha

    def _get_num_gts_means(self, targets, mask, dtype="float32"):
        num_gts = sum(len(a[m]) for a, m in zip(targets, mask))
        num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.)
        return num_gts

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, dst, axis=0) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_permutation_idx(self, indices):
        batch_idx = paddle.concat(x=[
            paddle.full_like(
                x=src, fill_value=i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = paddle.concat(x=[src for src, _ in indices])

        return batch_idx, src_idx

    def _get_loss_class(
            self,
            logits,
            gt_class,
            match_indices,
            num_gts, ):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class"
        if sum(len(a) for a in gt_class) > 0:
            idx = self._get_src_permutation_idx(match_indices)
            gt_class = [a.squeeze(axis=1) for a in gt_class]
            target_classes_o = paddle.concat(x=[
                t[J] if len(t) > 0 else paddle.to_tensor([])
                for t, (_, J) in zip(gt_class, match_indices)
            ]).astype('int64')

            target_classes_o = paddle.zeros_like(target_classes_o)

            target_classes = paddle.full(
                shape=logits.shape[:2],
                fill_value=logits.shape[2]).astype('int64')

            target_classes[idx] = target_classes_o

            target_classes_onehot = paddle.zeros(
                shape=[logits.shape[0], logits.shape[1], logits.shape[2] + 1],
                dtype=paddle.int64)
            bs, num_query_objects, cls = target_classes_onehot.shape
            index = paddle.arange(num_query_objects * bs, dtype=paddle.int64)
            target_classes_onehot = target_classes_onehot.reshape([-1, cls])
            target_classes_onehot[index, target_classes.reshape([-1])] = 1

            target_classes_onehot = target_classes_onehot.reshape(
                [bs, num_query_objects, cls])[:, :, :-1].astype(logits.dtype)

            return {
                name_class: self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits,
                    target_classes_onehot,
                    num_gts / num_query_objects,
                    alpha=self.focal_alpha)
                if self.use_focal_loss else F.cross_entropy(
                    logits,
                    target_classes_onehot,
                    weight=self.loss_coeff['class'])
            }
        else:
            return {name_class: paddle.to_tensor(0.0)}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox"
        name_giou = "loss_giou"

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            return loss

        idx = self._get_src_permutation_idx(match_indices)
        src_boxes = boxes[idx]

        target_boxes = []
        for t, (_, J) in zip(gt_bbox, match_indices):

            if len(t) == 0:
                continue
            target_boxes.append(t[J] if len(t[J].shape) > 1 else t[J].unsqueeze(
                axis=0))

        target_boxes = paddle.concat(x=target_boxes)

        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_boxes, target_boxes, reduction='sum') / num_gts

        loss[name_giou] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_boxes), bbox_cxcywh_to_xyxy(target_boxes))
        loss[name_giou] = loss[name_giou].sum() / num_gts

        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_embed(self, pred_embed, select_id, clip_query, gt_class,
                        indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)

        src_feature = pred_embed[idx]
        select_id = paddle.to_tensor(select_id)

        if sum(len(a) for a in gt_class) == 0:
            losses = {"loss_embed": paddle.to_tensor([0.])}
            return losses

        target_feature = []
        for t, (_, i) in zip(gt_class, indices):
            if len(t) < 1:
                continue
            for c in t[i]:
                index = (select_id == c).nonzero(as_tuple=False)[0]
                target_feature.append(clip_query[index])

        target_feature = paddle.stack(x=target_feature, axis=0)
        if len(src_feature.shape) == 1:
            src_feature = src_feature.unsqueeze(axis=0)

        src_feature = paddle.nn.functional.normalize(x=src_feature, axis=1)

        loss_feature = paddle.nn.functional.mse_loss(
            input=src_feature, label=target_feature, reduction='none')

        losses = {
            "loss_embed":
            loss_feature.sum() / num_boxes * self.loss_coeff['embed']
        }
        return losses

    def _get_loss(self,
                  loss,
                  outputs,
                  match_indices,
                  num_gts,
                  gt_class,
                  gt_bbox,
                  gt_mask=None):

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_embed = outputs["pred_embed"] if 'pred_embed' in outputs.keys(
        ) else None
        select_id = outputs['select_id'] if 'select_id' in outputs.keys(
        ) else None
        clip_query = outputs['clip_query'] if 'clip_query' in outputs.keys(
        ) else None

        loss_map = {
            "class": self._get_loss_class,
            "boxes": self._get_loss_bbox,
            "embed": self._get_loss_embed,
        }
        if loss == 'class':
            return loss_map[loss](pred_logits, gt_class, match_indices, num_gts)
        elif loss == 'boxes':
            return loss_map[loss](pred_boxes, gt_bbox, match_indices, num_gts)
        elif loss == 'embed':
            return loss_map[loss](pred_embed, select_id, clip_query, gt_class,
                                  match_indices, num_gts)

    def forward(self, outputs, gt_bbox, gt_class, masks=None, gt_mask=None):

        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        masks = []
        for t in gt_class:
            mask = t == -2
            for ind, v in enumerate(t):
                if v in outputs["select_id"]:
                    mask[ind] = True

            masks.append(mask)
        num_gts = self._get_num_gts_means(gt_class, masks)

        # Retrieve the matching between the outputs of the last layer and the targets
        select_id = outputs["select_id"]
        indices = self.matcher(outputs_without_aux['pred_boxes'],
                               outputs_without_aux['pred_logits'], gt_bbox,
                               gt_class, select_id)

        # # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self._get_loss(loss, outputs, indices, num_gts, gt_class,
                               gt_bbox, gt_mask))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs['pred_boxes'],
                                       aux_outputs['pred_logits'], gt_bbox,
                                       gt_class, select_id)

                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue

                    l_dict = self._get_loss(loss, aux_outputs, indices, num_gts,
                                            gt_class, gt_bbox, gt_mask)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_gt_class = []
            for bt in gt_class:
                bt = paddle.zeros_like(bt)
                bin_gt_class.append(bt)

            indices = self.matcher(enc_outputs['pred_boxes'],
                                   enc_outputs['pred_logits'], gt_bbox,
                                   bin_gt_class)

            for loss in self.losses:
                if loss == "masks" or loss == "embed":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                l_dict = self._get_loss(loss, enc_outputs, indices, num_gts,
                                        bin_gt_class, gt_bbox, gt_mask)

                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
