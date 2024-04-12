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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant

from ppdet.modeling.layers import MultiClassNMS
from ppdet.core.workspace import register
from ppdet.modeling.bbox_utils import delta2bbox_v2

__all__ = ['YOLOFHead']

INF = 1e8


def reduce_mean(tensor):
    world_size = paddle.distributed.get_world_size()
    if world_size == 1:
        return tensor
    paddle.distributed.all_reduce(tensor)
    return tensor / world_size


def find_inside_anchor(feat_size, stride, num_anchors, im_shape):
    feat_h, feat_w = feat_size[:2]
    im_h, im_w = im_shape[:2]
    inside_h = min(int(np.ceil(im_h / stride)), feat_h)
    inside_w = min(int(np.ceil(im_w / stride)), feat_w)
    inside_mask = paddle.zeros([feat_h, feat_w], dtype=paddle.bool)
    inside_mask[:inside_h, :inside_w] = True
    inside_mask = inside_mask.unsqueeze(-1).expand(
        [feat_h, feat_w, num_anchors])
    return inside_mask.reshape([-1])


@register
class YOLOFFeat(nn.Layer):
    def __init__(self,
                 feat_in=256,
                 feat_out=256,
                 num_cls_convs=2,
                 num_reg_convs=4,
                 norm_type='bn'):
        super(YOLOFFeat, self).__init__()
        assert norm_type == 'bn', "YOLOFFeat only support BN now."
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_type = norm_type

        cls_subnet, reg_subnet = [], []
        for i in range(self.num_cls_convs):
            feat_in = self.feat_in if i == 0 else self.feat_out
            cls_subnet.append(
                nn.Conv2D(
                    feat_in,
                    self.feat_out,
                    3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0.0, std=0.01)),
                    bias_attr=ParamAttr(initializer=Constant(value=0.0))))
            cls_subnet.append(
                nn.BatchNorm2D(
                    self.feat_out,
                    weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            cls_subnet.append(nn.ReLU())

        for i in range(self.num_reg_convs):
            feat_in = self.feat_in if i == 0 else self.feat_out
            reg_subnet.append(
                nn.Conv2D(
                    feat_in,
                    self.feat_out,
                    3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0.0, std=0.01)),
                    bias_attr=ParamAttr(initializer=Constant(value=0.0))))
            reg_subnet.append(
                nn.BatchNorm2D(
                    self.feat_out,
                    weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            reg_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.reg_subnet = nn.Sequential(*reg_subnet)

    def forward(self, fpn_feat):
        cls_feat = self.cls_subnet(fpn_feat)
        reg_feat = self.reg_subnet(fpn_feat)
        return cls_feat, reg_feat


@register
class YOLOFHead(nn.Layer):
    __shared__ = ['num_classes', 'trt', 'exclude_nms']
    __inject__ = [
        'conv_feat', 'anchor_generator', 'bbox_assigner', 'loss_class',
        'loss_bbox', 'nms'
    ]

    def __init__(self,
                 num_classes=80,
                 conv_feat='YOLOFFeat',
                 anchor_generator='AnchorGenerator',
                 bbox_assigner='UniformAssigner',
                 loss_class='FocalLoss',
                 loss_bbox='GIoULoss',
                 ctr_clip=32.0,
                 delta_mean=[0.0, 0.0, 0.0, 0.0],
                 delta_std=[1.0, 1.0, 1.0, 1.0],
                 nms='MultiClassNMS',
                 prior_prob=0.01,
                 nms_pre=1000,
                 use_inside_anchor=False,
                 trt=False,
                 exclude_nms=False):
        super(YOLOFHead, self).__init__()
        self.num_classes = num_classes
        self.conv_feat = conv_feat
        self.anchor_generator = anchor_generator
        self.na = self.anchor_generator.num_anchors
        self.bbox_assigner = bbox_assigner
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.ctr_clip = ctr_clip
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        self.nms = nms
        self.nms_pre = nms_pre
        self.use_inside_anchor = use_inside_anchor
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms

        bias_init_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_score = self.add_sublayer(
            'cls_score',
            nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=self.num_classes * self.na,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(
                    value=bias_init_value))))

        self.bbox_pred = self.add_sublayer(
            'bbox_pred',
            nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=4 * self.na,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

        self.object_pred = self.add_sublayer(
            'object_pred',
            nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=self.na,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

    def forward(self, feats, targets=None):
        assert len(feats) == 1, "YOLOF only has one level feature."
        conv_cls_feat, conv_reg_feat = self.conv_feat(feats[0])
        cls_logits = self.cls_score(conv_cls_feat)
        objectness = self.object_pred(conv_reg_feat)
        bboxes_reg = self.bbox_pred(conv_reg_feat)

        N, C, H, W = cls_logits.shape[:]
        cls_logits = cls_logits.reshape((N, self.na, self.num_classes, H, W))
        objectness = objectness.reshape((N, self.na, 1, H, W))
        norm_cls_logits = cls_logits + objectness - paddle.log(
            1.0 + paddle.clip(
                cls_logits.exp(), max=INF) + paddle.clip(
                    objectness.exp(), max=INF))
        norm_cls_logits = norm_cls_logits.reshape((N, C, H, W))

        anchors = self.anchor_generator([norm_cls_logits])

        if self.training:
            yolof_losses = self.get_loss(
                [anchors[0], norm_cls_logits, bboxes_reg], targets)
            return yolof_losses
        else:
            return anchors[0], norm_cls_logits, bboxes_reg

    def get_loss(self, head_outs, targets):
        anchors, cls_logits, bbox_preds = head_outs

        feat_size = cls_logits.shape[-2:]
        cls_logits = cls_logits.transpose([0, 2, 3, 1])
        cls_logits = cls_logits.reshape([0, -1, self.num_classes])
        bbox_preds = bbox_preds.transpose([0, 2, 3, 1])
        bbox_preds = bbox_preds.reshape([0, -1, 4])

        num_pos_list = []
        cls_pred_list, cls_tar_list = [], []
        reg_pred_list, reg_tar_list = [], []
        # find and gather preds and targets in each image
        for cls_logit, bbox_pred, gt_bbox, gt_class, im_shape in zip(
                cls_logits, bbox_preds, targets['gt_bbox'], targets['gt_class'],
                targets['im_shape']):
            if self.use_inside_anchor:
                inside_mask = find_inside_anchor(
                    feat_size, self.anchor_generator.strides[0], self.na,
                    im_shape.tolist())
                cls_logit = cls_logit[inside_mask]
                bbox_pred = bbox_pred[inside_mask]
                anchors = anchors[inside_mask]

            bbox_pred = delta2bbox_v2(
                bbox_pred,
                anchors,
                self.delta_mean,
                self.delta_std,
                ctr_clip=self.ctr_clip)
            bbox_pred = bbox_pred.reshape([-1, bbox_pred.shape[-1]])

            # -2:ignore, -1:neg, >=0:pos
            match_labels, pos_bbox_pred, pos_bbox_tar = self.bbox_assigner(
                bbox_pred, anchors, gt_bbox)
            pos_mask = (match_labels >= 0)
            neg_mask = (match_labels == -1)
            chosen_mask = paddle.logical_or(pos_mask, neg_mask)

            gt_class = gt_class.reshape([-1])
            bg_class = paddle.to_tensor(
                [self.num_classes], dtype=gt_class.dtype)
            # a trick to assign num_classes to negative targets
            gt_class = paddle.concat([gt_class, bg_class], axis=-1)
            match_labels = paddle.where(
                neg_mask,
                paddle.full_like(match_labels, gt_class.size - 1), match_labels)
            num_pos_list.append(max(1.0, pos_mask.sum().item()))

            cls_pred_list.append(cls_logit[chosen_mask])
            cls_tar_list.append(gt_class[match_labels[chosen_mask]])
            reg_pred_list.append(pos_bbox_pred)
            reg_tar_list.append(pos_bbox_tar)

        num_tot_pos = paddle.to_tensor(sum(num_pos_list))
        num_tot_pos = reduce_mean(num_tot_pos).item()
        num_tot_pos = max(1.0, num_tot_pos)

        cls_pred = paddle.concat(cls_pred_list)
        cls_tar = paddle.concat(cls_tar_list)
        cls_loss = self.loss_class(
            cls_pred, cls_tar, reduction='sum') / num_tot_pos

        reg_pred_list = [_ for _ in reg_pred_list if _ is not None]
        reg_tar_list = [_ for _ in reg_tar_list if _ is not None]
        if len(reg_pred_list) == 0:
            reg_loss = bbox_preds.sum() * 0.0
        else:
            reg_pred = paddle.concat(reg_pred_list)
            reg_tar = paddle.concat(reg_tar_list)
            reg_loss = self.loss_bbox(reg_pred, reg_tar).sum() / num_tot_pos

        yolof_losses = {
            'loss': cls_loss + reg_loss,
            'loss_cls': cls_loss,
            'loss_reg': reg_loss,
        }
        return yolof_losses

    def get_bboxes_single(self,
                          anchors,
                          cls_scores,
                          bbox_preds,
                          im_shape,
                          scale_factor,
                          rescale=True):
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for anchor, cls_score, bbox_pred in zip(anchors, cls_scores,
                                                bbox_preds):
            cls_score = cls_score.reshape([-1, self.num_classes])
            bbox_pred = bbox_pred.reshape([-1, 4])
            if self.nms_pre is not None and cls_score.shape[0] > self.nms_pre:
                max_score = cls_score.max(axis=1)
                _, topk_inds = max_score.topk(self.nms_pre)
                bbox_pred = bbox_pred.gather(topk_inds)
                anchor = anchor.gather(topk_inds)
                cls_score = cls_score.gather(topk_inds)

            bbox_pred = delta2bbox_v2(
                bbox_pred,
                anchor,
                self.delta_mean,
                self.delta_std,
                max_shape=im_shape,
                ctr_clip=self.ctr_clip).squeeze()
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(F.sigmoid(cls_score))
        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        mlvl_bboxes = paddle.squeeze(mlvl_bboxes)
        if rescale:
            mlvl_bboxes = mlvl_bboxes / paddle.concat(
                [scale_factor[::-1], scale_factor[::-1]])
        mlvl_scores = paddle.concat(mlvl_scores)
        mlvl_scores = mlvl_scores.transpose([1, 0])
        return mlvl_bboxes, mlvl_scores

    def decode(self, anchors, cls_scores, bbox_preds, im_shape, scale_factor):
        batch_bboxes = []
        batch_scores = []
        for img_id in range(cls_scores[0].shape[0]):
            num_lvls = len(cls_scores)
            cls_score_list = [cls_scores[i][img_id] for i in range(num_lvls)]
            bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_lvls)]
            bboxes, scores = self.get_bboxes_single(
                anchors, cls_score_list, bbox_pred_list, im_shape[img_id],
                scale_factor[img_id])
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = paddle.stack(batch_bboxes, 0)
        batch_scores = paddle.stack(batch_scores, 0)
        return batch_bboxes, batch_scores

    def post_process(self, head_outs, im_shape, scale_factor):
        anchors, cls_scores, bbox_preds = head_outs
        cls_scores = cls_scores.transpose([0, 2, 3, 1])
        bbox_preds = bbox_preds.transpose([0, 2, 3, 1])
        pred_bboxes, pred_scores = self.decode(
            [anchors], [cls_scores], [bbox_preds], im_shape, scale_factor)

        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
