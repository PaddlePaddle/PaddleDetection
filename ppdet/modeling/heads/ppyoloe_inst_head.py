# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.cspresnet import ConvBNLayer
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['PPYOLOEInstHead']


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


@register
class PPYOLOEInstHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms', 'exclude_post_process'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 num_prototypes=32,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                     'mask': 2.5,
                     'dice': 2.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False,
                 num_sample_points=12544,
                 oversample_ratio=3.0,
                 important_sample_ratio=0.75):
        super(PPYOLOEInstHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.num_prototypes = num_prototypes
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.important_sample_ratio = important_sample_ratio
        self.num_oversample_points = int(num_sample_points * oversample_ratio)
        self.num_important_points = int(num_sample_points *
                                        important_sample_ratio)
        self.num_random_points = num_sample_points - self.num_important_points

        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        self.stem_coeff = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
            self.stem_coeff.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        self.pred_coeff = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2D(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
            self.pred_coeff.append(
                nn.Conv2D(
                    in_c, self.num_prototypes, 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        # semantic seg conv
        self.conv_seg = nn.Conv2D(
            self.num_prototypes, self.num_classes, 1, bias_attr=False)

        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        proj = paddle.linspace(0, self.reg_max, self.reg_max + 1).reshape(
            [1, self.reg_max + 1, 1, 1])
        self.proj_conv.weight.set_value(proj)
        self.proj_conv.weight.stop_gradient = True
        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward_train(self, feats, targets):
        feats, mask_feat = feats
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list, mask_coeff_list = [], [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            mask_coeff = self.pred_coeff[i](self.stem_coeff[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
            mask_coeff_list.append(mask_coeff.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, mask_coeff_list, mask_feat,
            anchors, anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        feats, mask_feat = feats
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list, mask_coeff_list  = [], [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l]).transpose(
                [0, 2, 3, 1])
            reg_dist = self.proj_conv(F.softmax(reg_dist, axis=1)).squeeze(1)
            mask_coeff = self.pred_coeff[i](self.stem_coeff[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)
            mask_coeff_list.append(mask_coeff.reshape([-1, self.num_prototypes, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=-1)

        return cls_score_list, reg_dist_list, mask_coeff_list, mask_feat, \
            anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        # assert len(feats) == len(self.fpn_strides), \
        #     "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_max + 1]))
        pred_dist = self.proj_conv(pred_dist.transpose([0, 3, 1, 2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = paddle.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def _mask_loss(self, pred_coeffs, mask_feat, gt_masks,
                   assigned_labels, assigned_gt_index):
        mask_positive = (assigned_labels != self.num_classes)
        batch_size, num_max_boxes = gt_masks.shape[:2]
        mask_h, mask_w = mask_feat.shape[-2:]
        # subtract extra offset
        batch_ind = paddle.arange(end=batch_size).unsqueeze(-1)
        assigned_gt_index -= batch_ind * num_max_boxes
        num_pos = mask_positive.sum()
        pos_pred_masks = []
        pos_assigned_masks = []
        loss_mask = paddle.zeros([1])
        loss_dice = paddle.zeros([1])
        if num_pos > 0:
            for i in range(batch_size):
                num_inst = mask_positive[i].sum()
                if num_inst > 0:
                    coeff_mask = mask_positive[i].unsqueeze(-1).tile(
                        [1, self.num_prototypes])
                    pos_pred_coeff = paddle.masked_select(
                        pred_coeffs[i], coeff_mask).reshape(
                        [-1, self.num_prototypes])
                    pos_pred_mask = (pos_pred_coeff @ mask_feat[i].flatten(1)).reshape(
                        [num_inst, mask_h, mask_w])
                    pos_assigned_gt_index = paddle.masked_select(
                        assigned_gt_index[i], mask_positive[i])
                    pos_assigned_mask = paddle.gather(
                        gt_masks[i], pos_assigned_gt_index, axis=0)
                    pos_pred_masks.append(pos_pred_mask)
                    pos_assigned_masks.append(pos_assigned_mask)
            pos_pred_masks = paddle.concat(pos_pred_masks, axis=0)
            pos_assigned_masks = paddle.concat(pos_assigned_masks, axis=0)
            # sample points
            sample_points = self._get_point_coords_by_uncertainty(pos_pred_masks)
            sample_points = 2.0 * sample_points.unsqueeze(1) - 1.0

            pos_pred_masks = F.grid_sample(
                pos_pred_masks.unsqueeze(1), sample_points,
                align_corners=False).squeeze([1, 2])
            pos_assigned_masks = F.grid_sample(
                pos_assigned_masks.unsqueeze(1), sample_points,
                align_corners=False).squeeze([1, 2]).detach()

            loss_mask = F.binary_cross_entropy_with_logits(
                pos_pred_masks, pos_assigned_masks,
                reduction='none').mean(1).sum() / num_pos
            loss_dice = self._dice_loss(
                pos_pred_masks, pos_assigned_masks, num_pos)
        return loss_mask, loss_dice

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

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_coeffs, mask_feat, anchors, \
            anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        gt_masks = gt_meta['gt_segm']

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, \
                assigned_scores, assigned_gt_index = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    gt_masks,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor,
                    pred_coeffs=pred_coeffs.detach(),
                    mask_feat=mask_feat.detach())
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, \
                assigned_scores, assigned_gt_index = \
                self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    gt_masks,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_coeffs=pred_coeffs.detach(),
                    mask_feat=mask_feat.detach())
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss_mask, loss_dice = self._mask_loss(pred_coeffs, mask_feat, gt_masks,
                                               assigned_labels, assigned_gt_index)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['mask'] * loss_mask + \
               self.loss_weight['dice'] * loss_dice
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
        }
        return out_dict

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_dist, pred_coeffs, mask_feat, \
            anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes,
                 pred_scores.transpose([0, 2, 1]),
                 pred_coeffs.transpose([0, 2, 1])], axis=-1), mask_feat
        else:
            # scale bbox to origin
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores, pred_coeffs, mask_feat
            else:
                self.nms.return_index = True
                bbox_pred, bbox_num, keep_index = self.nms(pred_bboxes, pred_scores)
                assert keep_index is not None
                batch_size = mask_feat.shape[0]
                assert batch_size == 1
                keep_index = keep_index.reshape([batch_size, -1])
                pred_coeffs = pred_coeffs.transpose([0, 2, 1])
                mask_h, mask_w = mask_feat.shape[-2:]
                mask_pred = []
                for i in range(batch_size):
                    pos_pred_coeff = paddle.gather(pred_coeffs[i], keep_index[i], axis=0)
                    pos_pred_mask = pos_pred_coeff @ mask_feat[i].flatten(1)
                    pos_pred_mask = pos_pred_mask.reshape([-1, mask_h, mask_w])
                    mask_pred.append(pos_pred_mask)

                mask_pred = paddle.concat(mask_pred, axis=0)
                mask_pred = F.interpolate(
                    mask_pred.unsqueeze(0),
                    scale_factor=[4 / float(scale_y), 4 / float(scale_x)],
                    mode='bilinear',
                    align_corners=False).squeeze(0)
                mask_pred = (F.sigmoid(mask_pred) > 0.5).astype('uint8')

                return bbox_pred, bbox_num, mask_pred
