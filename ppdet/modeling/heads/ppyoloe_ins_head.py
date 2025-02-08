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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.backbones.csp_darknet import BaseConv
from ppdet.modeling.layers import MultiClassNMS
from ppdet.modeling.ops import get_static_shape, get_act_fn
from .ppyoloe_head import ESEAttn
from ..assigners.utils import generate_anchors_for_grid_cell
from ..bbox_utils import batch_distance2bbox
from ..initializer import bias_init_with_prob, constant_
from ..losses import GIoULoss

__all__ = ['PPYOLOEInsHead']


def custom_binary_cross_entropy_with_logits(x, y):
    max_val = paddle.maximum(-x, paddle.to_tensor(0.0))
    loss = (1 - y) * x + max_val + paddle.log(
        paddle.exp(-max_val) + paddle.exp(-x - max_val))
    return loss


class MaskProto(nn.Layer):
    # YOLOv8 mask Proto module for instance segmentation models
    def __init__(self, ch_in, num_protos=256, num_masks=32, act='silu'):
        super().__init__()
        self.conv1 = BaseConv(ch_in, num_protos, 3, 1, act=act)
        self.upsample = nn.Conv2DTranspose(num_protos,
                                           num_protos,
                                           2,
                                           2,
                                           0,
                                           bias_attr=True)
        self.conv2 = BaseConv(num_protos, num_protos, 3, 1, act=act)
        self.conv3 = BaseConv(num_protos, num_masks, 1, 1, act=act)

    def forward(self, x):
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.
    """
    assert x.shape[
        -1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = paddle.empty_like(x) if isinstance(
        x, paddle.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks (paddle.Tensor): [h, w, n] tensor of masks
      boxes (paddle.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
      (paddle.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = paddle.chunk(boxes[:, :, None], 4, axis=1)
    r = paddle.arange(w, dtype=x1.dtype)[None, None, :]
    c = paddle.arange(h, dtype=y1.dtype)[None, :, None]
    if "npu" in paddle.device.get_all_custom_device_type():
        # bool tensor broadcast multiply is extreamly slow on npu, so we cast it to float32.
        m_dtype = masks.dtype
        return masks * ((r >= x1).cast(m_dtype) * (r < x2).cast(m_dtype) *
                        (c >= y1).cast(m_dtype) * (c < y2).cast(m_dtype))
    else:
        return masks * ((r >= x1) * (r < x2) * (c >= y1) *
                        (c < y2)).astype(masks.dtype)


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
    quality but is slower.

    Args:
      protos (paddle.Tensor): [mask_dim, mask_h, mask_w]
      masks_in (paddle.Tensor): [n, mask_dim], n is number of masks after nms
      bboxes (paddle.Tensor): [n, 4], n is number of masks after nms
      shape (tuple): the size of the input image (h,w)

    Returns:
      (paddle.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = F.sigmoid(masks_in @ protos.reshape([c, -1])).reshape([-1, mh, mw])
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks


@register
class PPYOLOEInsHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill', 'width_mult'
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
                 reg_range=None,
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
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 mask_thr_binary=0.5,
                 num_masks=32,
                 num_protos=256,
                 width_mult=1.0,
                 for_distill=False):
        super(PPYOLOEInsHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"

        self.mask_thr_binary = mask_thr_binary
        self.num_masks = num_masks
        self.num_protos = int(num_protos * width_mult)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.sm_use = True
            self.reg_range = reg_range
        else:
            self.sm_use = False
            self.reg_range = (0, reg_max + 1)
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
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
        self.use_shared_conv = use_shared_conv
        self.for_distill = for_distill
        self.is_teacher = False

        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        self.stem_ins = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
            self.stem_reg.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
            self.stem_ins.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        self.pred_ins = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2D(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2D(in_c, 4 * self.reg_channels, 3, padding=1))
            self.pred_ins.append(nn.Conv2D(in_c, self.num_masks, 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2D(self.reg_channels, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True
        self._init_weights()

        self.proto = MaskProto(in_channels[-1],
                               self.num_protos,
                               self.num_masks,
                               act=act)

        if self.for_distill:
            self.distill_pairs = {}

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
        }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        proj = paddle.linspace(self.reg_range[0], self.reg_range[1] - 1,
                               self.reg_channels).reshape(
                                   [1, self.reg_channels, 1, 1])
        self.proj_conv.weight.set_value(proj)
        self.proj_conv.weight.stop_gradient = True
        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        mask_feat = self.proto(feats[-1])
        mask_coeff_list = []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            if "npu" in paddle.device.get_all_custom_device_type(
            ):  # backward in avgpool is extremely slow in npu kernel, replace it with mean
                avg_feat = feat.mean(axis=[2, 3], keepdim=True)
            else:
                avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            msk_coeff = self.pred_ins[i](self.stem_ins[i](feat, avg_feat) +
                                         feat)

            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            mask_coeff_list.append(msk_coeff.flatten(2).transpose([0, 2,
                                                                   1]))  ###
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

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
            anchor_point = paddle.cast(paddle.stack([shift_x, shift_y],
                                                    axis=-1),
                                       dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        mask_proto = self.proto(feats[-1])

        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list, pred_mask_list = [], [], []
        feats_shapes = []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            feats_shapes.append(l)

            if "npu" in paddle.device.get_all_custom_device_type():
                # backward in avgpool is extremely slow in npu kernel, replace it with mean
                avg_feat = feat.mean(axis=[2, 3], keepdim=True)
            else:
                avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            mask_coeff = self.pred_ins[i](self.stem_ins[i](feat, avg_feat) +
                                          feat)
            pred_mask_list.append(mask_coeff.reshape([-1, self.num_masks, l]))

            reg_dist = reg_dist.reshape([-1, 4, self.reg_channels,
                                         l]).transpose([0, 2, 3, 1])

            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(reg_dist,
                                                    axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        pred_mask_list = paddle.concat(pred_mask_list, axis=-1)

        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, pred_mask_list, mask_proto, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
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
        loss = F.binary_cross_entropy(score,
                                      label,
                                      weight=weight,
                                      reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(pred_score,
                                      gt_score,
                                      weight=weight,
                                      reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist.transpose([0, 3, 1,
                                                        2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox_decode_fake(self, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        pred_dist_dfl = F.softmax(
            pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist_dfl.transpose([0, 3, 1,
                                                            2])).squeeze(1)
        return pred_dist, pred_dist_dfl

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        if "npu" in paddle.device.get_all_custom_device_type(
        ):  # npu clip kernel causes nan grad, replace it with maximum & minimum.
            out = paddle.concat([lt, rb], -1)
            out = paddle.maximum(
                out, paddle.to_tensor(self.reg_range[0], dtype=out.dtype))
            out = paddle.minimum(
                out,
                paddle.to_tensor(self.reg_range[1] - 1 - 0.01,
                                 dtype=out.dtype))
            return out
        else:
            return paddle.concat([lt, rb],
                                 -1).clip(self.reg_range[0],
                                          self.reg_range[1] - 1 - 0.01)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = paddle.cast(target.floor(), 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist,
                                    target_left - lower_bound,
                                    reduction='none') * weight_left
        loss_right = F.cross_entropy(pred_dist,
                                     target_right - lower_bound,
                                     reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def get_loss(self, head_outs, gt_meta):
        assert 'gt_bbox' in gt_meta and 'gt_class' in gt_meta
        assert 'gt_segm' in gt_meta

        pred_scores, pred_distri, pred_mask_coeffs, mask_proto, anchors, \
            anchor_points, num_anchors_list, stride_tensor = head_outs

        bs = pred_scores.shape[0]
        imgsz = paddle.to_tensor(
            [640, 640]
        )  # paddle.to_tensor(pred_scores[0].shape[2:]) * self.fpn_strides[0]  # image size (h,w)
        mask_h, mask_w = mask_proto.shape[-2:]

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = paddle.stack(gt_meta['gt_class'])
        gt_bboxes = paddle.stack(gt_meta['gt_bbox'])
        pad_gt_mask = paddle.stack(gt_meta['pad_gt_mask'])
        gt_segms = paddle.stack(gt_meta['gt_segm']).cast('float32')
        if tuple(gt_segms.shape[-2:]) != (mask_h, mask_w):  # downsample
            gt_segms = F.interpolate(gt_segms, (mask_h, mask_w),
                                     mode='nearest').reshape(
                                         [bs, -1, mask_h * mask_w])

        # label assignment
        assigned_labels, assigned_bboxes, assigned_scores, assigned_gt_index = \
            self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_segms=gt_segms)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # assign segms for masks
        assigned_masks = paddle.gather(gt_segms.reshape([-1, mask_h * mask_w]),
                                       assigned_gt_index.flatten(),
                                       axis=0)
        assigned_masks = assigned_masks.reshape(
            [bs, assigned_gt_index.shape[1], mask_h * mask_w])

        assign_out_dict = self.get_loss_from_assign(
            pred_scores, pred_distri, pred_bboxes, anchor_points_s,
            assigned_labels, assigned_bboxes, assigned_scores, assigned_masks,
            pred_mask_coeffs, mask_proto, stride_tensor, imgsz)

        loss = assign_out_dict
        return loss

    def get_loss_from_assign(self, pred_scores, pred_distri, pred_bboxes,
                             anchor_points_s, assigned_labels, assigned_bboxes,
                             assigned_scores, assigned_masks, pred_mask_coeffs,
                             mask_proto, stride_tensor, imgsz):
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores,
                                        assigned_scores,
                                        alpha_l=-1)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        if "npu" in paddle.device.get_all_custom_device_type():
            # npu clip kernel causes nan grad, replace it with maximum & minimum.
            assigned_scores_sum = paddle.maximum(
                assigned_scores_sum,
                paddle.to_tensor(1., dtype=assigned_scores_sum.dtype))
        else:
            assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)

        loss_cls /= assigned_scores_sum

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.astype('int32').unsqueeze(-1).tile(
                [1, 1, 4]).astype('bool')
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(assigned_scores.sum(-1),
                                               mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # dfl loss
            dist_mask = mask_positive.unsqueeze(-1).astype('int32').tile(
                [1, 1, self.reg_channels * 4]).astype('bool')
            pred_dist_pos = paddle.masked_select(pred_distri,
                                                 dist_mask).reshape([
                                                     -1, 4, self.reg_channels
                                                 ])  # pred_dist in funs
            assigned_ltrb = self._bbox2distance(
                anchor_points_s, assigned_bboxes)  # anchor_points in func
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos,
                                     self.reg_range[0]) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum

            # mask loss
            loss_mask = self.calculate_segmentation_loss(
                mask_positive, assigned_masks, assigned_bboxes * stride_tensor,
                mask_proto, pred_mask_coeffs, imgsz)
            # [bs, 8400] [bs, 8400, 160 * 160] [bs, 8400, 4] [bs, 32, 160, 160] [bs, 8400, 32]
            loss_mask /= assigned_scores_sum
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_mask = paddle.zeros([1])
            loss_dfl = paddle.zeros([1])

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['iou'] * loss_mask

        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_mask': loss_mask,
            'loss_l1': loss_l1,
        }
        return out_dict

    def calculate_segmentation_loss(self,
                                    fg_mask,
                                    masks,
                                    target_bboxes,
                                    proto,
                                    pred_masks,
                                    imgsz,
                                    overlap=True):
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (paddle.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (paddle.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (paddle.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (paddle.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (paddle.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (paddle.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (paddle.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (paddle.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (paddle.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = paddle.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = paddle.to_tensor([0.])

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]].cast(
            target_bboxes.dtype)
        # [8, 8400, 4]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[...,
                                                    2:].prod(2).unsqueeze(-1)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * paddle.to_tensor(
            [mask_w, mask_h, mask_w, mask_h],
            dtype=target_bboxes_normalized.dtype)

        for i, single_i in enumerate(
                zip(fg_mask, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            #  [8400] [8400, 32] [32, 160, 160] [8400, 4]  [8400, 1]  [8400, 25600]
            if fg_mask_i.any():
                loss += self.single_mask_loss(masks_i[fg_mask_i],
                                              pred_masks_i[fg_mask_i], proto_i,
                                              mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])
                # [10, 25600]  [10, 32]  [32, 160, 160]  [10, 4]  [10, 1]
            else:
                loss += (proto * 0).sum() + (
                    pred_masks * 0).sum()  # inf sums may lead to nan loss
        return loss

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        """
        Compute the instance segmentation loss for a single image.
        Args:
            gt_mask (paddle.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (paddle.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (paddle.Tensor): Prototype masks of shape (32, H, W).
            xyxy (paddle.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (paddle.Tensor): Area of each ground truth bounding box of shape (n,).
        Returns:
            (paddle.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = paddle.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        nt = pred.shape[0]
        gt_mask = gt_mask.reshape([nt, *proto.shape[1:]])
        nmasks = 32
        pred_mask = (pred @ proto.reshape([nmasks, -1])).reshape(
            [-1, *proto.shape[1:]])  # (n,32) @ (32,80,80) -> (n,80,80)

        if "npu" in paddle.device.get_all_custom_device_type():
            # bce npu kernel causes nan grad, replace it with numeric stable custom implementation.
            loss = custom_binary_cross_entropy_with_logits(pred_mask, gt_mask)
        else:
            loss = F.binary_cross_entropy_with_logits(pred_mask,
                                                      gt_mask,
                                                      reduction='none')
        return (crop_mask(loss, xyxy).mean(axis=(1, 2)) /
                area.squeeze(-1)).sum()

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640],
                     rescale=True):

        if getattr(self, "export_mode", False):
            return self.export_post_process(head_outs, im_shape, scale_factor, infer_shape, rescale)

        pred_scores, pred_dist, pred_mask_coeffs, mask_feat, anchor_points, stride_tensor = head_outs

        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat([
                pred_bboxes,
                pred_scores.transpose([0, 2, 1]),
                pred_mask_coeffs.transpose([0, 2, 1])
            ],
                                 axis=-1), mask_feat, None
            # [1, 8400, 4+80+32], [1, 32, 160, 160]

        bbox_pred, bbox_num, keep_idxs = self.nms(pred_bboxes, pred_scores)

        if bbox_num.sum() > 0:
            pred_mask_coeffs = pred_mask_coeffs.transpose([0, 2, 1])
            mask_coeffs = paddle.gather(
                pred_mask_coeffs.reshape([-1, self.num_masks]), keep_idxs)

            mask_logits = process_mask_upsample(mask_feat[0], mask_coeffs,
                                                bbox_pred[:, 2:6], infer_shape)
            if rescale:
                ori_h, ori_w = im_shape[0] / scale_factor[0]
                mask_logits = F.interpolate(
                    mask_logits.unsqueeze(0),
                    size=[
                        int(paddle.round(mask_logits.shape[-2] /
                              scale_factor[0][0])),
                        int(paddle.round(mask_logits.shape[-1] /
                              scale_factor[0][1]))
                    ],
                    mode='bilinear',
                    align_corners=False)
                if "npu" in paddle.device.get_all_custom_device_type():
                    # due to npu numeric error, we need to take round of img size.
                    mask_logits = mask_logits[
                        ..., :round(ori_h.item()), :round(ori_w.item())]
                else:
                    mask_logits = mask_logits[..., :int(ori_h), :int(ori_w)]

            masks = mask_logits.squeeze(0)
            mask_pred = paddle.to_tensor(masks > self.mask_thr_binary).cast("float32")

            # scale bbox to origin
            scale_factor = scale_factor.flip(-1).tile([1, 2])
            bbox_pred[:, 2:6] /= scale_factor
        else:
            ori_h, ori_w = im_shape[0] / scale_factor[0]
            bbox_num = paddle.to_tensor([1]).cast("int32")
            bbox_pred = paddle.zeros([bbox_num, 6])
            mask_pred = paddle.zeros([bbox_num, int(ori_h), int(ori_w)])

        return bbox_pred, bbox_num, mask_pred, keep_idxs
    
    def export_post_process(self,
                            head_outs,
                            im_shape,
                            scale_factor,
                            infer_shape=[640, 640],
                            rescale=True):
        pred_scores, pred_dist, pred_mask_coeffs, mask_feat, anchor_points, stride_tensor = head_outs

        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat([
                pred_bboxes,
                pred_scores.transpose([0, 2, 1]),
                pred_mask_coeffs.transpose([0, 2, 1])
            ],
                                 axis=-1), mask_feat, None
            # [1, 8400, 4+80+32], [1, 32, 160, 160]

        bbox_pred, bbox_num, keep_idxs = self.nms(pred_bboxes, pred_scores)

        pred_mask_coeffs = pred_mask_coeffs.transpose([0, 2, 1])
        mask_coeffs = paddle.gather(
            pred_mask_coeffs.reshape([-1, self.num_masks]), keep_idxs)

        mask_logits = process_mask_upsample(mask_feat[0], mask_coeffs,
                                            bbox_pred[:, 2:6], infer_shape)
        if rescale:
            ori_h, ori_w = im_shape[0] / scale_factor[0]
            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0),
                size=[
                    int(paddle.round(mask_logits.shape[-2] /
                            scale_factor[0][0])),
                    int(paddle.round(mask_logits.shape[-1] /
                            scale_factor[0][1]))
                ],
                mode='bilinear',
                align_corners=False)
            mask_logits = mask_logits[..., :int(ori_h), :int(ori_w)]

        masks = mask_logits.squeeze(0)
        mask_pred = paddle.to_tensor(masks > self.mask_thr_binary).cast("float32")

        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2])
        bbox_pred[:, 2:6] /= scale_factor

        return bbox_pred, bbox_num, mask_pred, keep_idxs