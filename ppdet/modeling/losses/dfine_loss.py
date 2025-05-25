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
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .detr_loss import DINOLoss
from ..transformers.dfine_transformer import bbox2distance
from ..transformers import bbox_cxcywh_to_xyxy
from ..bbox_utils import bbox_iou

__all__ = ['DFINELoss']


@register
class DFINELoss(DINOLoss):

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1,
                     'fgl': 0.15,
                     'ddf': 1.5
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_mal=False,
                 use_vfl=False,
                 vfl_iou_type='bbox',
                 uni_match_ind=0,
                 reg_max=32,
                 reg_scale=4.0):
        assert use_focal_loss
        self.reg_max = reg_max
        self.reg_scale = reg_scale

        super(DFINELoss, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            loss_coeff=loss_coeff,
            aux_loss=aux_loss,
            use_focal_loss=use_focal_loss,
            use_mal=use_mal,
            use_vfl=use_vfl,
            vfl_iou_type=vfl_iou_type,
            use_uni_match=False,
            uni_match_ind=uni_match_ind)

    def _get_loss_local(self, boxes, corners, refs, gt_bbox, match_indices, num_gts,
                        teacher_logits=None, teacher_corners=None, postfix="", T=5, is_dn=False):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_fgl = "loss_fgl" + postfix
        name_ddf = "loss_ddf" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0 or corners is None or refs is None:
            loss[name_fgl] = paddle.to_tensor([0.])
            loss[name_ddf] = paddle.to_tensor([0.])
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes.detach(), gt_bbox,
                                                            match_indices)

        corners_pos = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(corners, match_indices)
        ])
        refs_pos = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(refs, match_indices)
        ])

        target = bbox_cxcywh_to_xyxy(target_bbox)

        # cache
        if self.fgl_targets_dn is None and is_dn:
            self.fgl_targets_dn = bbox2distance(
                refs_pos, target, self.reg_max, self.reg_scale, 0.5)
        if self.fgl_targets is None and not is_dn:
            self.fgl_targets = bbox2distance(
                refs_pos, target, self.reg_max, self.reg_scale, 0.5)

        target_corners, weight_right, weight_left = self.fgl_targets_dn if is_dn else self.fgl_targets

        pos_ious = bbox_iou(
            bbox_cxcywh_to_xyxy(src_bbox).split(4, -1), target.split(4, -1))
        weight_targets = pos_ious.unsqueeze(-1).tile([1, 1, 4]).reshape([-1])

        loss[name_fgl] = self.loss_coeff['fgl'] * self.unimodal_distribution_focal_loss(
            corners_pos.reshape([-1, self.reg_max + 1]),
            target_corners,
            weight_right=weight_right,
            weight_left=weight_left,
            weight=weight_targets,
            avg_factor=num_gts)

        if teacher_logits is not None and teacher_corners is not None:
            assert self.use_focal_loss
            teacher_logits = teacher_logits.reshape([-1, self.num_classes])
            teacher_corners = teacher_corners.reshape([-1, self.reg_max + 1])
            corners = corners.reshape([-1, self.reg_max + 1])

            weight_targets_local = teacher_logits.sigmoid().max(axis=-1)

            idx = paddle.concat([src for (src, _) in match_indices])
            mask = paddle.zeros_like(weight_targets_local, dtype="bool")
            mask[idx] = True
            mask = mask.unsqueeze(-1).tile([1, 4]).reshape([-1])

            weight_targets_local[idx] = \
                pos_ious.reshape([-1]).to(weight_targets_local.dtype)
            weight_targets_local = \
                weight_targets_local.unsqueeze(-1).tile([1, 4]).reshape([-1]).detach()

            loss_match_local = weight_targets_local * (T ** 2) * (paddle.nn.KLDivLoss(reduction='none')
            (F.log_softmax(corners / T, axis=1), F.softmax(teacher_corners.detach() / T, axis=1))).sum(-1)
            if self.num_pos is None:
                batch_scale = 8 / boxes.shape[0]  # Avoid the influence of batch size per GPU
                self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
            loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
            loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
            loss[name_ddf] = (loss_match_local1 * self.num_pos +
                              loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)
            loss[name_ddf] = self.loss_coeff['ddf'] * loss[name_ddf]
        return loss

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             corners,
                             refs,
                             gt_bbox,
                             gt_class,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             teacher_logits=None,
                             teacher_corners=None,
                             match_indices=None,
                             match_indices_go=None,
                             dn_match_indices=None,
                             num_gts=1,
                             num_gts_go=None,
                             gt_score=None):
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)

        if match_indices_go is None:
            match_indices_go = match_indices

        if num_gts_go is None:
            num_gts_go = num_gts

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
            self._get_loss_bbox(boxes, gt_bbox, match_indices_go, num_gts_go,
                                postfix))
        loss.update(
            self._get_loss_local(boxes, corners, refs, gt_bbox, match_indices_go, num_gts_go,
                                 teacher_logits, teacher_corners, postfix,
                                 is_dn=dn_match_indices is not None))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices_go, num_gts_go,
                                    postfix))
        return loss

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      corners,
                      refs,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      num_gts_go=None,
                      teacher_logits=None,
                      teacher_corners=None,
                      match_indices_list=None,
                      match_indices_go=None,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None,
                      gt_score=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_fgl, loss_ddf = [], []
        loss_mask, loss_dice = [], []
        if num_gts_go is None:
            num_gts_go = num_gts
        for i, (aux_boxes, aux_logits, aux_corners, aux_refs) in \
                enumerate(zip(boxes, logits, corners, refs)):
            aux_masks = masks[i] if masks is not None else None

            if dn_match_indices is not None:
                match_indices = dn_match_indices
            elif match_indices_list is not None:
                match_indices = match_indices_list[i]

            if match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if match_indices_go is None:
                match_indices_go = match_indices
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
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices_go,
                                        num_gts_go, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            loss_ = self._get_loss_local(aux_boxes, aux_corners, aux_refs, gt_bbox,
                                         match_indices_go, num_gts_go,
                                         teacher_logits, teacher_corners, postfix,
                                         is_dn=dn_match_indices is not None)
            loss_fgl.append(loss_['loss_fgl' + postfix])
            if teacher_logits is not None and teacher_corners is not None:
                loss_ddf.append(loss_['loss_ddf' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices_go,
                                            num_gts_go, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_giou_aux" + postfix: paddle.add_n(loss_giou)
        }
        if teacher_logits is not None and teacher_corners is not None:
            loss["loss_fgl_aux" + postfix] = paddle.add_n(loss_fgl)
            loss["loss_ddf_aux" + postfix] = paddle.add_n(loss_ddf)
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = paddle.add_n(loss_mask)
            loss["loss_dice_aux" + postfix] = paddle.add_n(loss_dice)
        return loss

    def _get_go_indices(self, *indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_list in zip(*indices_aux_list):
            idx0_list, idx1_list = list(zip(*indices_list))
            if sum(map(len, idx0_list)) == 0:
                results.append((paddle.to_tensor([], dtype='int64'),
                                paddle.to_tensor([], dtype='int64')))
                continue

            idx0 = paddle.concat(idx0_list, axis=0)
            idx1 = paddle.concat(idx1_list, axis=0)
            ind = paddle.stack([idx0, idx1], axis=1)

            unique, counts = paddle.unique(ind, return_counts=True, axis=0)
            count_sort_indices = paddle.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = paddle.to_tensor(list(column_to_row.keys()), dtype='int64')
            final_cols = paddle.to_tensor(list(column_to_row.values()), dtype='int64')
            results.append((final_rows, final_cols))
        return results

    def forward(self,
                boxes,
                logits,
                corners,
                refs,
                pre_bboxes,
                pre_logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_bboxes=None,
                dn_logits=None,
                dn_corners=None,
                dn_refs=None,
                dn_pre_bboxes=None,
                dn_pre_logits=None,
                dn_meta=None,
                gt_score=None,
                **kwargs):
        assert masks is None and gt_mask is None, "Mask loss is not supported in DFINE"
        assert gt_score is None, "GT score is not supported in DFINE"

        # clear cache
        self.num_pos, self.num_neg = None, None
        self.fgl_targets_dn, self.fgl_targets = None, None

        indices_list = []
        for box, logit in zip(boxes, logits):
            indices = self.matcher(box, logit, gt_bbox, gt_class)
            indices_list.append(indices)
        pre_indices = self.matcher(pre_bboxes, pre_logits, gt_bbox, gt_class)

        enc_indices, *aux_indices_list, out_indices = indices_list
        indices_go = self._get_go_indices(
            out_indices, *aux_indices_list, pre_indices, enc_indices)

        enc_boxes, *aux_boxes_list, out_boxes = boxes.unbind()
        enc_logits, *aux_logits_list, out_logits = logits.unbind()
        *aux_corners_list, out_corners = corners.unbind()
        *aux_refs_list, out_refs = refs.unbind()

        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = paddle.to_tensor([num_boxes_go], dtype="float32")
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_boxes_go)
            num_boxes_go /= paddle.distributed.get_world_size()
        num_boxes_go = paddle.clip(num_boxes_go, min=1.)

        num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            out_boxes,
            out_logits,
            out_corners,
            out_refs,
            gt_bbox,
            gt_class,
            teacher_logits=None,
            teacher_corners=None,
            match_indices=out_indices,
            match_indices_go=indices_go,
            dn_match_indices=None,
            num_gts=num_gts,
            num_gts_go=num_boxes_go,
            gt_score=gt_score if gt_score is not None else None)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    aux_boxes_list,
                    aux_logits_list,
                    aux_corners_list,
                    aux_refs_list,
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    num_gts_go=num_boxes_go,
                    teacher_logits=out_logits.detach(),
                    teacher_corners=out_corners.detach(),
                    match_indices_list=aux_indices_list,
                    match_indices_go=indices_go,
                    dn_match_indices=None,
                    postfix=postfix,
                    gt_score=gt_score if gt_score is not None else None))
            total_loss.update(
                self._get_loss_aux(
                    [pre_bboxes],
                    [pre_logits],
                    [None],
                    [None],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    num_gts_go=num_boxes_go,
                    teacher_logits=None,
                    teacher_corners=None,
                    match_indices_list=[pre_indices],
                    match_indices_go=indices_go,
                    dn_match_indices=None,
                    postfix=postfix + "_pre",
                    gt_score=gt_score if gt_score is not None else None))
            total_loss.update(
                self._get_loss_aux(
                    [enc_boxes],
                    [enc_logits],
                    [None],
                    [None],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    num_gts_go=num_boxes_go,
                    teacher_logits=None,
                    teacher_corners=None,
                    match_indices_list=[enc_indices],
                    match_indices_go=indices_go,
                    dn_match_indices=None,
                    postfix=postfix + "_enc",
                    gt_score=gt_score if gt_score is not None else None))

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group

            *dn_aux_boxes_list, dn_out_boxes = dn_bboxes.unbind()
            *dn_aux_logits_list, dn_out_logits = dn_logits.unbind()
            *dn_aux_corners_list, dn_out_corners = dn_corners.unbind()
            *dn_aux_refs_list, dn_out_refs = dn_refs.unbind()

            dn_loss = self._get_prediction_loss(
                dn_out_boxes,
                dn_out_logits,
                dn_out_corners,
                dn_out_refs,
                gt_bbox,
                gt_class,
                postfix=postfix + '_dn',
                teacher_logits=None,
                teacher_corners=None,
                match_indices=None,
                match_indices_go=None,
                dn_match_indices=dn_match_indices,
                num_gts=num_gts,
                num_gts_go=None,
                gt_score=gt_score if gt_score is not None else None)

            if self.aux_loss:
                dn_loss.update(
                    self._get_loss_aux(
                        dn_aux_boxes_list,
                        dn_aux_logits_list,
                        dn_aux_corners_list,
                        dn_aux_refs_list,
                        gt_bbox,
                        gt_class,
                        self.num_classes,
                        num_gts,
                        num_gts_go=None,
                        teacher_logits=dn_out_logits.detach(),
                        teacher_corners=dn_out_corners.detach(),
                        match_indices_list=None,
                        match_indices_go=None,
                        dn_match_indices=dn_match_indices,
                        postfix=postfix + '_dn',
                        gt_score=gt_score if gt_score is not None else None))
                dn_loss.update(
                    self._get_loss_aux(
                        [dn_pre_bboxes],
                        [dn_pre_logits],
                        [None],
                        [None],
                        gt_bbox,
                        gt_class,
                        self.num_classes,
                        num_gts,
                        num_gts_go=None,
                        teacher_logits=None,
                        teacher_corners=None,
                        match_indices_list=None,
                        match_indices_go=None,
                        dn_match_indices=dn_match_indices,
                        postfix=postfix + '_dn' + '_pre',
                        gt_score=gt_score if gt_score is not None else None))

            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.to("int64")
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape([-1]) \
                + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape([-1])

        if weight is not None:
            weight = weight.to("float32")
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss
