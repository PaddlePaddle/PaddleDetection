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
from ..transformers import bbox_cxcywh_to_xyxy, sigmoid_focal_loss, varifocal_loss_with_logits, mal_loss_with_logits
from ..bbox_utils import bbox_iou

__all__ = ['DETRLoss', 'DINOLoss', 'RTDETRv3Loss', 'MaskDINOLoss',
           'DocLayoutV3Loss']


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
                 use_focal_loss=False,
                 use_mal=False,
                 use_vfl=False,
                 vfl_iou_type='bbox',
                 use_uni_match=False,
                 uni_match_ind=0):
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
        self.use_mal = use_mal
        self.use_vfl = use_vfl
        self.vfl_iou_type = vfl_iou_type
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

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
                        postfix="",
                        iou_score=None,
                        gt_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
            if iou_score is not None and (self.use_vfl or self.use_mal):
                if gt_score is not None:
                    target_score = paddle.zeros([bs, num_query_objects])
                    target_score = paddle.scatter(
                        target_score.reshape([-1, 1]), index, gt_score)
                    target_score = target_score.reshape(
                        [bs, num_query_objects, 1]) * target_label

                    target_score_iou = paddle.zeros([bs, num_query_objects])
                    target_score_iou = paddle.scatter(
                        target_score_iou.reshape([-1, 1]), index, iou_score)
                    target_score_iou = target_score_iou.reshape(
                        [bs, num_query_objects, 1]) * target_label
                    target_score = paddle.multiply(target_score,
                                                   target_score_iou)
                    if self.use_mal:                    
                        loss_ = self.loss_coeff[
                            'class'] * mal_loss_with_logits(
                                logits, target_score, target_label,
                                num_gts / num_query_objects)
                    else:
                        loss_ = self.loss_coeff[
                            'class'] * varifocal_loss_with_logits(
                                logits, target_score, target_label,
                                num_gts / num_query_objects)
                else:
                    target_score = paddle.zeros([bs, num_query_objects])
                    if num_gt > 0:
                        target_score = paddle.scatter(
                            target_score.reshape([-1, 1]), index, iou_score)
                    target_score = target_score.reshape(
                        [bs, num_query_objects, 1]) * target_label
                    if self.use_mal:
                        loss_ = self.loss_coeff[
                            'class'] * mal_loss_with_logits(
                                logits, target_score, target_label,
                                num_gts / num_query_objects)
                    else:
                        loss_ = self.loss_coeff[
                            'class'] * varifocal_loss_with_logits(
                                logits, target_score, target_label,
                                num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

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
                      postfix="",
                      masks=None,
                      gt_mask=None,
                      gt_score=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_mask, loss_dice = [], []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if self.use_vfl or self.use_mal:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                else:
                    iou_score = None
                if gt_score is not None:
                    _, target_score = self._get_src_target_assign(
                        logits[-1].detach(), gt_score, match_indices)
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(
                    aux_logits,
                    gt_class,
                    match_indices,
                    bg_index,
                    num_gts,
                    postfix,
                    iou_score,
                    gt_score=target_score
                    if gt_score is not None else None)['loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_giou_aux" + postfix: paddle.add_n(loss_giou)
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = paddle.add_n(loss_mask)
            loss["loss_dice_aux" + postfix] = paddle.add_n(loss_dice)
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        if 'npu' in paddle.device.get_device():
            target_assign = paddle.concat([
                paddle.gather(
                    t.to(paddle.int32), dst.to(paddle.int32), axis=0) for t, (_, dst) in zip(target, match_indices)
            ])
        else:
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

    def _get_num_gts(self, targets, dtype="float32"):
        num_gts = sum(len(a) for a in targets)
        num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             gt_bbox,
                             gt_class,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1,
                             gt_score=None):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl or self.use_mal:
            if gt_score is not None:  #ssod
                _, target_score = self._get_src_target_assign(
                    logits[-1].detach(), gt_score, match_indices)
            elif sum(len(a) for a in gt_bbox) > 0:
                if self.vfl_iou_type == 'bbox':
                    src_bbox, target_bbox = self._get_src_target_assign(
                        boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                elif self.vfl_iou_type == 'mask':
                    assert (masks is not None and gt_mask is not None,
                            'Make sure the input has `mask` and `gt_mask`')
                    assert sum(len(a) for a in gt_mask) > 0
                    src_mask, target_mask = self._get_src_target_assign(
                        masks.detach(), gt_mask, match_indices)
                    src_mask = F.interpolate(
                        src_mask.unsqueeze(0),
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=False).squeeze(0)
                    target_mask = F.interpolate(
                        target_mask.unsqueeze(0),
                        size=src_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False).squeeze(0)
                    src_mask = src_mask.flatten(1)
                    src_mask = F.sigmoid(src_mask)
                    src_mask = paddle.where(
                        src_mask > 0.5, 1., 0.).astype(masks.dtype)
                    target_mask = target_mask.flatten(1)
                    target_mask = paddle.where(
                        target_mask > 0.5, 1., 0.).astype(masks.dtype)
                    inter = (src_mask * target_mask).sum(1)
                    union = src_mask.sum(1) + target_mask.sum(1) - inter
                    iou_score = (inter + 1e-2) / (union + 1e-2)
                    iou_score = iou_score.unsqueeze(-1)
                else:
                    iou_score = None
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(
                logits,
                gt_class,
                match_indices,
                self.num_classes,
                num_gts,
                postfix,
                iou_score,
                gt_score=target_score if gt_score is not None else None))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts,
                                postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts,
                                    postfix))
        return loss

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                gt_score=None,
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts,
            gt_score=gt_score if gt_score is not None else None)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask,
                    gt_score=gt_score if gt_score is not None else None))

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
                gt_score=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(DINOLoss, self).forward(
            boxes,
            logits,
            gt_bbox,
            gt_class,
            num_gts=num_gts,
            gt_score=gt_score)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts,
                gt_score=gt_score)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = paddle.arange(end=num_gt, dtype="int64")
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((paddle.zeros(
                    [0], dtype="int64"), paddle.zeros(
                        [0], dtype="int64")))
        return dn_match_indices


@register
class RTDETRv3Loss(DETRLoss):
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
                gt_score=None,
                o2m=1,
                **kwargs):
        if o2m != 1:
            gt_boxes_copy = [box.tile([o2m, 1]) for box in gt_bbox]
            gt_class_copy = [label.tile([o2m, 1]) for label in gt_class]
        else:
            gt_boxes_copy = gt_bbox
            gt_class_copy = gt_class
        num_gts_copy = self._get_num_gts(gt_class_copy)
        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_boxes_copy,
            gt_class_copy,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=None,
            num_gts=num_gts_copy,
            gt_score=gt_score if gt_score is not None else None)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_boxes_copy,
                    gt_class_copy,
                    self.num_classes,
                    num_gts_copy,
                    dn_match_indices=None,
                    postfix=postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask,
                    gt_score=gt_score if gt_score is not None else None))

        if dn_meta is not None:
            num_gts = self._get_num_gts(gt_class)
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(RTDETRv3Loss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts,
                gt_score=gt_score)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = paddle.arange(end=num_gt, dtype="int64")
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((paddle.zeros(
                    [0], dtype="int64"), paddle.zeros(
                        [0], dtype="int64")))
        return dn_match_indices


@register
class MaskDINOLoss(DETRLoss):
    __shared__ = ['num_classes', 'use_focal_loss', 'num_sample_points']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 4,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 5,
                     'dice': 5
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 vfl_iou_type='bbox',
                 num_sample_points=12544,
                 oversample_ratio=3.0,
                 important_sample_ratio=0.75):
        super(MaskDINOLoss, self).__init__(num_classes, matcher, loss_coeff,
                                           aux_loss, use_focal_loss, use_vfl, vfl_iou_type)
        assert oversample_ratio >= 1
        assert important_sample_ratio <= 1 and important_sample_ratio >= 0

        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.important_sample_ratio = important_sample_ratio
        self.num_oversample_points = int(num_sample_points * oversample_ratio)
        self.num_important_points = int(num_sample_points *
                                        important_sample_ratio)
        self.num_random_points = num_sample_points - self.num_important_points

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
                dn_out_masks=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(MaskDINOLoss, self).forward(
            boxes,
            logits,
            gt_bbox,
            gt_class,
            masks=masks,
            gt_mask=gt_mask,
            num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = DINOLoss.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(MaskDINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                masks=dn_out_masks,
                gt_mask=gt_mask,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        # sample points
        sample_points = self._get_point_coords_by_uncertainty(src_masks)
        sample_points = 2.0 * sample_points.unsqueeze(1) - 1.0

        src_masks = F.grid_sample(
            src_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2])

        target_masks = F.grid_sample(
            target_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2]).detach()

        loss[name_mask] = self.loss_coeff[
            'mask'] * F.binary_cross_entropy_with_logits(
                src_masks, target_masks,
                reduction='none').mean(1).sum() / num_gts
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _get_point_coords_by_uncertainty(self, masks):
        # Sample points based on their uncertainty.
        masks = masks.detach()
        num_masks = masks.shape[0]
        sample_points = paddle.rand(
            [num_masks, 1, self.num_oversample_points, 2])

        out_mask = F.grid_sample(
            masks.unsqueeze(1), 2.0 * sample_points - 1.0,
            align_corners=False).squeeze([1, 2])
        out_mask = -paddle.abs(out_mask)

        _, topk_ind = paddle.topk(out_mask, self.num_important_points, axis=1)
        batch_ind = paddle.arange(end=num_masks, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_important_points])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        sample_points = paddle.gather_nd(sample_points.squeeze(1), topk_ind)
        if self.num_random_points > 0:
            sample_points = paddle.concat(
                [
                    sample_points,
                    paddle.rand([num_masks, self.num_random_points, 2])
                ],
                axis=1)
        return sample_points

class RelativeReadingOrderLoss(nn.Layer):
    """
    Relative Reading Order Loss for PP-DocLayoutV3.

    This loss models pairwise relative reading order relationships between document
    elements using a global pointer mechanism. It supervises the model to predict
    whether element i comes before element j in the reading order.

    Key features:
        1. Pairwise modeling: Predicts relative order for all element pairs (i, j).
        2. Locality-aware: Emphasizes correct ordering of neighboring elements.
        3. Robust training: Uses GCE loss to handle annotation noise.
        4. Antisymmetric constraint: Leverages logits[i,j] = -logits[j,i] property.

    The loss function computes:
        Loss = (1/|P|) * Σ_{(i,j)∈P} w_{ij} * L(logits[i,j], target[i,j])
    where P is the set of valid element pairs, w_{ij} is the pair weight (higher
    for neighbors), and L is the per-pair loss (GCE or BCE).

    Args:
        use_upper_only (bool): If True, only supervise the upper triangle of the
            order matrix to avoid redundancy (since logits are antisymmetric).
            Default: True.
        k_local (int): Fixed local neighbor window size. Elements within distance
            k_local in the ground truth order are considered neighbors.
            Default: 5.
        locality (bool): If True, apply higher weights to neighboring element pairs
            to emphasize local ordering correctness. Default: True.
        k_local_ratio (float): Dynamic neighbor window as a ratio of total elements.
            The actual window is max(k_local, k_local_ratio * N). This helps adapt
            to varying numbers of elements. Default: 0.3.
        w_gt (float): Weight multiplier for neighboring pairs (within k_local).
            Neighboring pairs have weight w_gt, while distant pairs have weight 1.0.
            Default: 2.0.
        label_smooth (float): Label smoothing epsilon. Smooths binary targets:
            target = target * (1 - eps) + 0.5 * eps. Helps prevent overconfidence.
            Default: 0.01.
        robust (str): Robust loss type to handle annotation noise.
            - 'gce': Generalized Cross Entropy, more robust to label noise.
            - 'bce': Standard Binary Cross Entropy.
            Default: 'gce'.
        q (float): GCE loss parameter q in range [0, 1]. Lower q = more robust
            to noise. Formula: (1 - (p_correct)^q) / q. Default: 0.7.

    Note:
        Elements with order < 0 are treated as invalid and excluded from supervision.
        This allows marking elements whose order is unknown or ambiguous.

    Examples:
        .. code-block:: python

            loss_fn = RelativeReadingOrderLoss(
                use_upper_only=True,
                k_local=5,
                locality=True,
                w_gt=2.0
            )
            # order_logits: [B, N, N] pairwise order predictions
            # gt_order: list of [N_i] ground truth orders
            loss = loss_fn(order_logits, gt_order, match_indices)
    """

    def __init__(self,
                 use_upper_only=True,
                 k_local=5,
                 locality=True,
                 k_local_ratio=0.3,
                 w_gt=2.0,
                 label_smooth=0.01,
                 robust='gce',
                 q=0.7):
        super().__init__()
        self.use_upper_only = use_upper_only
        self.k_local = k_local
        self.locality = locality
        self.k_local_ratio = k_local_ratio
        self.w_gt = w_gt
        self.label_smooth = label_smooth
        self.robust = robust
        self.q = q

    @staticmethod
    def _pair_mask(N: int, use_upper_only: bool = False):
        """
        Generate supervision pair mask to determine which pairs to supervise.

        Args:
            N (int): Number of elements.
            use_upper_only (bool): If True, only include upper triangle pairs (i < j).
                This avoids redundancy when logits have antisymmetric property.
                Default: False.

        Returns:
            Tensor: Boolean mask of shape [N, N]. True for pairs to supervise.
                If use_upper_only=True: returns upper triangle (excluding diagonal).
                If use_upper_only=False: returns all non-diagonal pairs.
        """
        if use_upper_only:
            # Upper triangle only: logits[i,j] for i < j
            return paddle.triu(paddle.ones([N, N], dtype='bool'), 1)
        else:
            # All pairs except diagonal
            eye = paddle.eye(N)
            return ~eye.astype('bool')

    @staticmethod
    def _valid_pair_mask(order):
        """
        Generate valid pair mask by filtering out elements with invalid order.

        Elements with order < 0 are considered invalid (e.g., no ground truth
        order annotation). Pairs involving invalid elements are excluded.

        Args:
            order (Tensor): Ground truth reading order sequence.
                Shape: [N] where N is the number of elements.
                Values: order[i] >= 0 for valid elements, < 0 for invalid.

        Returns:
            Tensor: Boolean mask of shape [N, N]. True for pairs where both
                elements have valid order (order >= 0).
        """
        # v[i] = True if element i has valid order (order[i] >= 0)
        v = (order >= 0).astype('bool')
        # Outer product: valid_pair[i,j] = v[i] AND v[j]
        return v.unsqueeze(0) & v.unsqueeze(1)

    def _gce_loss(self, logits, target, q):
        """
        Compute Generalized Cross Entropy (GCE) loss for robust training.

        GCE is more robust to label noise than standard BCE. The formula is:
            GCE = (1 - (p_correct)^q) / q
        where p_correct = p if target=1, else (1-p).

        Lower q values make the loss more robust but slower to converge.
        q=1 reduces to standard cross entropy, q→0 approaches MAE.

        Args:
            logits (Tensor): Predicted logits. Shape: [num_pairs]
            target (Tensor): Binary targets (0 or 1). Shape: [num_pairs]
            q (float): GCE parameter in (0, 1]. Lower q = more robust.

        Returns:
            Tensor: Per-sample GCE loss. Shape: [num_pairs]
        """
        p = F.sigmoid(logits)
        # Probability of the correct class
        p_y = p * target + (1 - p) * (1 - target)
        # GCE formula: (1 - p_y^q) / q
        return (1.0 - paddle.pow(paddle.clip(p_y, 1e-6, 1.0), q)) / q

    def forward(self, relative_logits, gt_read_order, match_indices, gt_bboxes=None):
        """
        Compute relative reading order loss over all images in a batch.

        This method:
        1. Extracts matched predictions and ground truths using match_indices
        2. Builds target matrix where target[i,j]=1 if order[i] < order[j]
        3. Applies locality-aware weighting to emphasize neighbor pairs
        4. Computes weighted loss using GCE or BCE

        Args:
            relative_logits (Tensor): Pairwise reading order logits from model.
                Shape: [batch_size, num_queries, num_queries]
                where logits[b, i, j] > 0 indicates query i comes before query j.
            gt_read_order (list[Tensor]): Ground truth reading order for each image.
                List of length batch_size, each element is Tensor of shape [N_i]
                containing integer order values. Elements with order < 0 are invalid.
            match_indices (list[tuple]): Matching between predictions and ground truths.
                List of length batch_size, each element is (pred_idx, gt_idx) where:
                - pred_idx (Tensor): Indices of matched predictions. Shape: [M_i]
                - gt_idx (Tensor): Indices of matched ground truths. Shape: [M_i]
                Computed by Hungarian matcher.
            gt_bboxes (list[Tensor]|None): Ground truth bounding boxes (optional, not used).
                Reserved for potential spatial distance-based weighting. Default: None.

        Returns:
            Tensor: Scalar loss value averaged over all valid pairs in the batch.
                If no valid pairs exist, returns 0.

        Note:
            The loss is computed only for matched queries to ensure alignment between
            predictions and ground truths. Unmatched predictions are not supervised.
        """
        total_loss_num = paddle.to_tensor(0.0, dtype='float32')
        total_pairs = paddle.to_tensor(0.0, dtype='float32')

        B = len(gt_read_order)
        for i in range(B):
            # Get matched prediction and ground truth indices for this image
            pred_idx, gt_idx = match_indices[i]
            if pred_idx.numel() == 0 or gt_idx.numel() == 0:
                continue
            N = pred_idx.shape[0]
            if N <= 1:
                # Skip images with 0 or 1 element (no pairs to supervise)
                continue

            # Extract order logits for matched predictions: [N, N] submatrix
            logits = relative_logits[i][pred_idx][:, pred_idx]
            # Extract ground truth order for matched elements: [N]
            order = gt_read_order[i][gt_idx]

            # Generate masks for valid and supervised pairs
            valid_pair = self._valid_pair_mask(order)  # [N, N], filter order<0
            pair_mask = self._pair_mask(N, self.use_upper_only)  # [N, N], upper triangle
            base_mask = valid_pair & pair_mask  # [N, N], intersection

            pair_sum = paddle.sum(base_mask.astype('float32'))
            if pair_sum.item() == 0:
                # No valid pairs in this image
                continue

            # Build target matrix: target[i,j] = 1 if order[i] < order[j]
            o1, o2 = order.unsqueeze(1), order.unsqueeze(0)  # [N, 1] and [1, N]
            target_full = (o1 < o2).astype('float32')  # [N, N]

            # Compute pairwise distance in reading order
            order_dist = paddle.abs(o1 - o2)  # [N, N]
            # Dynamic local window: max of fixed k_local and ratio-based window
            k_local_i = min(max(self.k_local, int(N * self.k_local_ratio)), N - 1)

            # Build weight matrix W: higher weight for neighboring pairs
            W = paddle.ones([N, N], dtype='float32')
            if self.locality:
                # gt_local[i,j] = True if elements i and j are neighbors
                # (0 < |order[i] - order[j]| <= k_local_i)
                gt_local = (order_dist > 0) & (order_dist <= k_local_i)
                # Increase weight for neighboring pairs by factor (w_gt - 1.0)
                W = W + (self.w_gt - 1.0) * gt_local.astype('float32')

            # Extract supervised pairs using base_mask
            z = logits[base_mask]  # [num_pairs] predicted logits
            t = target_full[base_mask]  # [num_pairs] binary targets
            Wm = W[base_mask]  # [num_pairs] weights

            # Normalize weights to have mean 1.0 for stable gradient scaling
            Wm = Wm / (paddle.mean(Wm) + 1e-6)

            # Apply label smoothing: target = target * (1-eps) + 0.5 * eps
            # This softens hard 0/1 labels to prevent overconfidence
            if self.label_smooth > 0:
                eps = self.label_smooth
                t = t * (1 - eps) + 0.5 * eps

            # Compute per-pair loss using robust loss function
            if self.robust == 'gce':
                per_pair_loss = self._gce_loss(z, t, self.q)
            else:
                per_pair_loss = F.binary_cross_entropy_with_logits(z, t, reduction='none')

            # Weighted mean loss for this image
            loss_main = (per_pair_loss * Wm).mean()
            # Accumulate weighted loss across images
            total_loss_num = total_loss_num + loss_main * pair_sum
            total_pairs = total_pairs + pair_sum

        # Return average loss over all pairs in the batch
        return total_loss_num / (total_pairs + 1e-12)


@register
class DocLayoutV3Loss(MaskDINOLoss):
    """
    PP-DocLayoutV3 Loss Function with reading order prediction support.

    This loss extends MaskDINOLoss by adding relative reading order loss computation
    for document layout analysis. It computes a weighted combination of:
        1. Classification loss (focal loss or cross entropy)
        2. Bounding box regression loss (L1 + GIoU)
        3. Mask segmentation loss (BCE + Dice)
        4. Reading order loss (pairwise GCE) - NEW

    The total loss is:
        L_total = λ_cls*L_cls + λ_bbox*L_bbox + λ_giou*L_giou +
                  λ_mask*L_mask + λ_dice*L_dice + λ_order*L_order

    Key enhancements over MaskDINOLoss:
        - Adds RelativeReadingOrderLoss module for pairwise order supervision
        - Includes 'order' coefficient in loss_coeff (default: 50)
        - Handles order_logits in forward pass
        - Maintains full backward compatibility with MaskDINOLoss

    Args:
        num_classes (int): Number of object categories. Default: 80.
        matcher (str): Hungarian matcher for prediction-GT matching. Default: 'HungarianMatcher'.
        loss_coeff (dict): Loss weight coefficients for each loss term.
            Default: {
                'class': 4,   # Classification loss weight
                'bbox': 5,    # BBox L1 loss weight
                'giou': 2,    # GIoU loss weight
                'mask': 5,    # Mask BCE loss weight
                'dice': 5,    # Mask Dice loss weight
                'order': 50   # Reading order loss weight (NEW)
            }
            The 'order' coefficient is set to 50 to balance with other losses,
            as order loss operates on O(N²) pairs while others operate on O(N) objects.
        aux_loss (bool): Whether to compute auxiliary losses for intermediate decoder layers.
            Enables deep supervision for better training. Default: True.
        use_focal_loss (bool): Whether to use focal loss for classification. Default: False.
        use_vfl (bool): Whether to use varifocal loss. Default: False.
        vfl_iou_type (str): IoU type for VFL ('bbox' or 'mask'). Default: 'bbox'.
        num_sample_points (int): Number of points sampled for mask loss computation. Default: 12544.
        oversample_ratio (float): Oversampling ratio for hard negative mining. Default: 3.0.
        important_sample_ratio (float): Ratio of important points to sample. Default: 0.75.

    Inheritance:
        Inherits from MaskDINOLoss, which provides classification, bbox, and mask losses.
        Only extends the forward method to add reading order loss computation.

    Examples:
        .. code-block:: python

            loss_fn = DocLayoutV3Loss(
                num_classes=25,
                loss_coeff={'class': 4, 'bbox': 5, 'giou': 2,
                           'mask': 5, 'dice': 5, 'order': 50}
            )
            # Model outputs
            boxes, logits, masks, order_logits = model_outputs
            # Compute loss
            loss_dict = loss_fn(boxes, logits, order_logits,
                               gt_bbox, gt_class, gt_read_order,
                               masks=masks, gt_mask=gt_mask)
    """

    __shared__ = ['num_classes', 'use_focal_loss', 'num_sample_points']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 4,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 5,
                     'dice': 5,
                     'order': 50
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 vfl_iou_type='bbox',
                 num_sample_points=12544,
                 oversample_ratio=3.0,
                 important_sample_ratio=0.75,
                 order_loss_config=None):
        super(DocLayoutV3Loss, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            loss_coeff=loss_coeff,
            aux_loss=aux_loss,
            use_focal_loss=use_focal_loss,
            use_vfl=use_vfl,
            vfl_iou_type=vfl_iou_type,
            num_sample_points=num_sample_points,
            oversample_ratio=oversample_ratio,
            important_sample_ratio=important_sample_ratio
        )
        if order_loss_config is None:
            order_loss_config = {}
        self.read_order_loss = RelativeReadingOrderLoss(**order_loss_config)

    def forward(self,
                boxes,
                logits,
                order_logits,
                gt_bbox,
                gt_class,
                gt_read_order,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_out_masks=None,
                dn_meta=None,
                **kwargs):
        """
        Forward pass to compute all losses including reading order loss.

        Args:
            boxes: Predicted bounding boxes [num_layers, batch_size, num_queries, 4].
            logits: Classification logits [num_layers, batch_size, num_queries, num_classes].
            order_logits: Pairwise order logits [num_layers, batch_size, num_queries, num_queries].
            gt_bbox: Ground truth bboxes, list of [num_gts_i, 4].
            gt_class: Ground truth classes, list of [num_gts_i].
            gt_read_order: Ground truth reading order, list of [num_gts_i]. Values < 0 = invalid.
            masks: Predicted masks [num_layers, batch_size, num_queries, H, W]. Default: None.
            gt_mask: Ground truth masks, list of [num_gts_i, H, W]. Default: None.
            postfix: Suffix for loss keys. Default: "".
            dn_out_bboxes/dn_out_logits/dn_out_masks: Denoising outputs. Default: None.
            dn_meta: Denoising metadata. Default: None.

        Returns:
            dict: Loss values including 'order_loss' and base losses from MaskDINOLoss.
        """
        # Call MaskDINOLoss.forward for all base losses (class/bbox/giou/mask/dice + denoising)
        total_loss = super().forward(
            boxes, logits, gt_bbox, gt_class,
            masks=masks, gt_mask=gt_mask, postfix=postfix,
            dn_out_bboxes=dn_out_bboxes, dn_out_logits=dn_out_logits,
            dn_out_masks=dn_out_masks, dn_meta=dn_meta, **kwargs)

        # Compute reading order loss (last decoder layer only)
        if order_logits is not None and gt_read_order is not None:
            match_indices = self.matcher(
                boxes[-1], logits[-1], gt_bbox, gt_class,
                masks=masks[-1] if masks is not None else None,
                gt_mask=gt_mask)
            total_loss["order_loss"] = (
                self.read_order_loss(order_logits[-1], gt_read_order, match_indices)
                * self.loss_coeff['order'])

        return total_loss

