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
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .iou_loss import GIoULoss
from ..transformers import bbox_cxcywh_to_xyxy, sigmoid_focal_loss

__all__ = ['DETRLoss', 'DINOLoss']


@register
class DETRLoss(nn.Layer):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes

        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss

        if not self.use_focal_loss:
            self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                                   loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix=""):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix
        if logits is None:
            return {name_class: paddle.zeros([1])}
        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        if sum(len(a) for a in gt_class) > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
        return {
            name_class: self.loss_coeff['class'] * sigmoid_focal_loss(
                logits, target_label, num_gts / num_query_objects)
            if self.use_focal_loss else F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        }

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix
        if boxes is None:
            return {name_bbox: paddle.zeros([1]), name_giou: paddle.zeros([1])}
        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix
        if masks is None:
            return {name_mask: paddle.zeros([1]), name_dice: paddle.zeros([1])}
        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix=""):
        if boxes is None or logits is None:
            return {
                "loss_class_aux" + postfix: paddle.paddle.zeros([1]),
                "loss_bbox_aux" + postfix: paddle.paddle.zeros([1]),
                "loss_giou_aux" + postfix: paddle.paddle.zeros([1])
            }
        loss_class = []
        loss_bbox = []
        loss_giou = []
        for aux_boxes, aux_logits in zip(boxes, logits):
            if dn_match_indices is None:
                match_indices = self.matcher(aux_boxes, aux_logits, gt_bbox,
                                             gt_class)
            else:
                match_indices = dn_match_indices
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts, postfix)['loss_class' +
                                                                 postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_giou_aux" + postfix: paddle.add_n(loss_giou)
        }
        return loss

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

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor|None): [l, b, query, 4]
            logits (Tensor|None): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """
        dn_match_indices = kwargs.get("dn_match_indices", None)
        if dn_match_indices is None and (boxes is not None and
                                         logits is not None):
            match_indices = self.matcher(boxes[-1].detach(),
                                         logits[-1].detach(), gt_bbox, gt_class)
        else:
            match_indices = dn_match_indices

        num_gts = sum(len(a) for a in gt_bbox)
        num_gts = paddle.to_tensor([num_gts], dtype="float32")
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.) * kwargs.get("dn_num_group", 1.)

        total_loss = dict()
        total_loss.update(
            self._get_loss_class(logits[
                -1] if logits is not None else None, gt_class, match_indices,
                                 self.num_classes, num_gts, postfix))
        total_loss.update(
            self._get_loss_bbox(boxes[-1] if boxes is not None else None,
                                gt_bbox, match_indices, num_gts, postfix))
        if masks is not None and gt_mask is not None:
            total_loss.update(
                self._get_loss_mask(masks if masks is not None else None,
                                    gt_mask, match_indices, num_gts, postfix))

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1] if boxes is not None else None, logits[:-1]
                    if logits is not None else None, gt_bbox, gt_class,
                    self.num_classes, num_gts, dn_match_indices, postfix))

        return total_loss


@register
class DINOLoss(DETRLoss):
    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_meta=None,
                **kwargs):
        total_loss = super(DINOLoss, self).forward(boxes, logits, gt_bbox,
                                                   gt_class)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = []
            for i in range(len(gt_class)):
                num_gt = len(gt_class[i])
                if num_gt > 0:
                    gt_idx = paddle.arange(end=num_gt, dtype="int64")
                    gt_idx = gt_idx.unsqueeze(0).tile(
                        [dn_num_group, 1]).flatten()
                    assert len(gt_idx) == len(dn_positive_idx[i])
                    dn_match_indices.append((dn_positive_idx[i], gt_idx))
                else:
                    dn_match_indices.append((paddle.zeros(
                        [0], dtype="int64"), paddle.zeros(
                            [0], dtype="int64")))
        else:
            dn_match_indices, dn_num_group = None, 1.

        # compute denoising training loss
        dn_loss = super(DINOLoss, self).forward(
            dn_out_bboxes,
            dn_out_logits,
            gt_bbox,
            gt_class,
            postfix="_dn",
            dn_match_indices=dn_match_indices,
            dn_num_group=dn_num_group)
        total_loss.update(dn_loss)

        return total_loss
