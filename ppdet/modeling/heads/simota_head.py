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

# The code is based on:
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/yolox_head.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from functools import partial
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register

from ppdet.modeling.bbox_utils import distance2bbox, bbox2distance
from ppdet.data.transform.atss_assigner import bbox_overlaps

from .gfl_head import GFLHead


@register
class OTAHead(GFLHead):
    """
    OTAHead
    Args:
        conv_feat (object): Instance of 'FCOSFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_qfl (object): Instance of QualityFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'assigner', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 loss_class='QualityFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 assigner='SimOTAAssigner',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(OTAHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_qfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_qfl.use_sigmoid

        self.assigner = assigner

    def _get_target_single(self, flatten_cls_pred, flatten_center_and_stride,
                           flatten_bbox, gt_bboxes, gt_labels):
        """Compute targets for priors in a single image.
        """
        pos_num, label, label_weight, bbox_target = self.assigner(
            F.sigmoid(flatten_cls_pred), flatten_center_and_stride,
            flatten_bbox, gt_bboxes, gt_labels)

        return (pos_num, label, label_weight, bbox_target)

    def get_loss(self, head_outs, gt_meta):
        cls_scores, bbox_preds = head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_scores
        ]
        num_imgs = gt_meta['im_id'].shape[0]
        featmap_sizes = [[featmap.shape[-2], featmap.shape[-1]]
                         for featmap in cls_scores]

        decode_bbox_preds = []
        center_and_strides = []
        for featmap_size, stride, bbox_pred in zip(featmap_sizes,
                                                   self.fpn_stride, bbox_preds):

            # center in origin image
            yy, xx = self.get_single_level_center_point(featmap_size, stride,
                                                        self.cell_offset)

            center_and_stride = paddle.stack([xx, yy, stride, stride], -1).tile(
                [num_imgs, 1, 1])
            center_and_strides.append(center_and_stride)
            center_in_feature = center_and_stride.reshape(
                [-1, 4])[:, :-2] / stride
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, 4 * (self.reg_max + 1)])
            pred_distances = self.distribution_project(bbox_pred)
            decode_bbox_pred_wo_stride = distance2bbox(
                center_in_feature, pred_distances).reshape([num_imgs, -1, 4])
            decode_bbox_preds.append(decode_bbox_pred_wo_stride * stride)

        flatten_cls_preds = [
            cls_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_scores
        ]
        flatten_cls_preds = paddle.concat(flatten_cls_preds, axis=1)
        flatten_bboxes = paddle.concat(decode_bbox_preds, axis=1)
        flatten_center_and_strides = paddle.concat(center_and_strides, axis=1)

        gt_boxes, gt_labels = gt_meta['gt_bbox'], gt_meta['gt_class']
        pos_num_l, label_l, label_weight_l, bbox_target_l = [], [], [], []
        for flatten_cls_pred,flatten_center_and_stride,flatten_bbox,gt_box, gt_label \
            in zip(flatten_cls_preds.detach(),flatten_center_and_strides.detach(), \
                   flatten_bboxes.detach(),gt_boxes, gt_labels):
            pos_num, label, label_weight, bbox_target = self._get_target_single(
                flatten_cls_pred, flatten_center_and_stride, flatten_bbox,
                gt_box, gt_label)
            pos_num_l.append(pos_num)
            label_l.append(label)
            label_weight_l.append(label_weight)
            bbox_target_l.append(bbox_target)

        labels = paddle.to_tensor(np.stack(label_l, axis=0))
        label_weights = paddle.to_tensor(np.stack(label_weight_l, axis=0))
        bbox_targets = paddle.to_tensor(np.stack(bbox_target_l, axis=0))

        center_and_strides_list = self._images_to_levels(
            flatten_center_and_strides, num_level_anchors)
        labels_list = self._images_to_levels(labels, num_level_anchors)
        label_weights_list = self._images_to_levels(label_weights,
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(bbox_targets,
                                                   num_level_anchors)
        num_total_pos = sum(pos_num_l)
        try:
            paddle.distributed.all_reduce(paddle.to_tensor(num_total_pos))
            num_total_pos = paddle.clip(
                num_total_pos / paddle.distributed.get_world_size(), min=1.)
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_qfl_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, center_and_strides, labels, label_weights, bbox_targets, stride in zip(
                cls_scores, bbox_preds, center_and_strides_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride):
            center_and_strides = center_and_strides.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])
            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])
            label_weights = label_weights.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = paddle.nonzero(
                paddle.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            score = np.zeros(labels.shape)

            if len(pos_inds) > 0:
                pos_bbox_targets = paddle.gather(bbox_targets, pos_inds, axis=0)
                pos_bbox_pred = paddle.gather(bbox_pred, pos_inds, axis=0)
                pos_centers = paddle.gather(
                    center_and_strides[:, :-2], pos_inds, axis=0) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = paddle.gather(
                    weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)
                score[pos_inds.numpy()] = bbox_iou

                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
                target_corners = bbox2distance(pos_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape([-1])
                # regression loss
                loss_bbox = paddle.sum(
                    self.loss_bbox(pos_decode_bbox_pred,
                                   pos_decode_bbox_targets) * weight_targets)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = paddle.to_tensor([0], dtype='float32')

            # qfl loss
            score = paddle.to_tensor(score)
            loss_qfl = self.loss_qfl(
                cls_score, (labels, score),
                weight=label_weights,
                avg_factor=num_total_pos)
            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_qfl_list.append(loss_qfl)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        try:
            paddle.distributed.all_reduce(paddle.to_tensor(avg_factor))
            avg_factor = paddle.clip(
                avg_factor / paddle.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)
        if avg_factor <= 0:
            loss_qfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = paddle.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_qfl = sum(loss_qfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss_states = dict(
            loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states


@register
class OTAVFLHead(OTAHead):
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'assigner', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 loss_class='VarifocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 assigner='SimOTAAssigner',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(OTAVFLHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_vfl.use_sigmoid

        self.assigner = assigner

    def get_loss(self, head_outs, gt_meta):
        cls_scores, bbox_preds = head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_scores
        ]
        num_imgs = gt_meta['im_id'].shape[0]
        featmap_sizes = [[featmap.shape[-2], featmap.shape[-1]]
                         for featmap in cls_scores]

        decode_bbox_preds = []
        center_and_strides = []
        for featmap_size, stride, bbox_pred in zip(featmap_sizes,
                                                   self.fpn_stride, bbox_preds):
            # center in origin image
            yy, xx = self.get_single_level_center_point(featmap_size, stride,
                                                        self.cell_offset)
            strides = paddle.full((len(xx), ), stride)
            center_and_stride = paddle.stack([xx, yy, strides, strides],
                                             -1).tile([num_imgs, 1, 1])
            center_and_strides.append(center_and_stride)
            center_in_feature = center_and_stride.reshape(
                [-1, 4])[:, :-2] / stride
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, 4 * (self.reg_max + 1)])
            pred_distances = self.distribution_project(bbox_pred)
            decode_bbox_pred_wo_stride = distance2bbox(
                center_in_feature, pred_distances).reshape([num_imgs, -1, 4])
            decode_bbox_preds.append(decode_bbox_pred_wo_stride * stride)

        flatten_cls_preds = [
            cls_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_scores
        ]
        flatten_cls_preds = paddle.concat(flatten_cls_preds, axis=1)
        flatten_bboxes = paddle.concat(decode_bbox_preds, axis=1)
        flatten_center_and_strides = paddle.concat(center_and_strides, axis=1)

        gt_boxes, gt_labels = gt_meta['gt_bbox'], gt_meta['gt_class']
        pos_num_l, label_l, label_weight_l, bbox_target_l = [], [], [], []
        for flatten_cls_pred, flatten_center_and_stride, flatten_bbox,gt_box,gt_label \
                in zip(flatten_cls_preds.detach(), flatten_center_and_strides.detach(), \
                       flatten_bboxes.detach(),gt_boxes,gt_labels):
            pos_num, label, label_weight, bbox_target = self._get_target_single(
                flatten_cls_pred, flatten_center_and_stride, flatten_bbox,
                gt_box, gt_label)
            pos_num_l.append(pos_num)
            label_l.append(label)
            label_weight_l.append(label_weight)
            bbox_target_l.append(bbox_target)

        labels = paddle.to_tensor(np.stack(label_l, axis=0))
        label_weights = paddle.to_tensor(np.stack(label_weight_l, axis=0))
        bbox_targets = paddle.to_tensor(np.stack(bbox_target_l, axis=0))

        center_and_strides_list = self._images_to_levels(
            flatten_center_and_strides, num_level_anchors)
        labels_list = self._images_to_levels(labels, num_level_anchors)
        label_weights_list = self._images_to_levels(label_weights,
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(bbox_targets,
                                                   num_level_anchors)
        num_total_pos = sum(pos_num_l)
        try:
            paddle.distributed.all_reduce(paddle.to_tensor(num_total_pos))
            num_total_pos = paddle.clip(
                num_total_pos / paddle.distributed.get_world_size(), min=1.)
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_vfl_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, center_and_strides, labels, label_weights, bbox_targets, stride in zip(
                cls_scores, bbox_preds, center_and_strides_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride):
            center_and_strides = center_and_strides.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])
            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = paddle.nonzero(
                paddle.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            # vfl
            vfl_score = np.zeros(cls_score.shape)

            if len(pos_inds) > 0:
                pos_bbox_targets = paddle.gather(bbox_targets, pos_inds, axis=0)
                pos_bbox_pred = paddle.gather(bbox_pred, pos_inds, axis=0)
                pos_centers = paddle.gather(
                    center_and_strides[:, :-2], pos_inds, axis=0) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = paddle.gather(
                    weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)

                # vfl
                pos_labels = paddle.gather(labels, pos_inds, axis=0)
                vfl_score[pos_inds.numpy(), pos_labels] = bbox_iou

                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
                target_corners = bbox2distance(pos_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape([-1])
                # regression loss
                loss_bbox = paddle.sum(
                    self.loss_bbox(pos_decode_bbox_pred,
                                   pos_decode_bbox_targets) * weight_targets)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = paddle.to_tensor([0], dtype='float32')

            # vfl loss
            num_pos_avg_per_gpu = num_total_pos
            vfl_score = paddle.to_tensor(vfl_score)
            loss_vfl = self.loss_vfl(
                cls_score, vfl_score, avg_factor=num_pos_avg_per_gpu)

            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_vfl_list.append(loss_vfl)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        try:
            paddle.distributed.all_reduce(paddle.to_tensor(avg_factor))
            avg_factor = paddle.clip(
                avg_factor / paddle.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)
        if avg_factor <= 0:
            loss_vfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = paddle.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_vfl = sum(loss_vfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss_states = dict(
            loss_vfl=loss_vfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states
