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
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/ld_head.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ppdet.modeling.bbox_utils import distance2bbox, bbox2distance, batch_distance2bbox
from ppdet.data.transform.atss_assigner import bbox_overlaps
from .gfl_head import GFLHead


@register
class LDGFLHead(GFLHead):
    """
    GFLHead for LD distill
    Args:
        conv_feat (object): Instance of 'FCOSFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of QualityFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'loss_ld', 'loss_ld_vlr', 'loss_kd', 'nms'
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
                 loss_ld='KnowledgeDistillationKLDivLoss',
                 loss_ld_vlr='KnowledgeDistillationKLDivLoss',
                 loss_kd='KnowledgeDistillationKLDivLoss',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):

        super(LDGFLHead, self).__init__(
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
        self.loss_ld = loss_ld
        self.loss_kd = loss_kd
        self.loss_ld_vlr = loss_ld_vlr

    def forward(self, fpn_feats):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        cls_logits_list = []
        bboxes_reg_list = []
        for stride, scale_reg, fpn_feat in zip(self.fpn_stride,
                                               self.scales_regs, fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat)
            cls_score = self.gfl_head_cls(conv_cls_feat)
            bbox_pred = scale_reg(self.gfl_head_reg(conv_reg_feat))

            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score
            if not self.training:
                cls_score = F.sigmoid(cls_score.transpose([0, 2, 3, 1]))
                bbox_pred = bbox_pred.transpose([0, 2, 3, 1])
                b, cell_h, cell_w, _ = paddle.shape(cls_score)
                y, x = self.get_single_level_center_point(
                    [cell_h, cell_w], stride, cell_offset=self.cell_offset)
                center_points = paddle.stack([x, y], axis=-1)
                cls_score = cls_score.reshape([b, -1, self.cls_out_channels])
                bbox_pred = self.distribution_project(bbox_pred) * stride
                bbox_pred = bbox_pred.reshape([b, cell_h * cell_w, 4])

                # NOTE: If keep_ratio=False and image shape value that
                # multiples of 32, distance2bbox not set max_shapes parameter
                # to speed up model prediction. If need to set max_shapes,
                # please use inputs['im_shape'].
                bbox_pred = batch_distance2bbox(
                    center_points, bbox_pred, max_shapes=None)

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return (cls_logits_list, bboxes_reg_list)

    def get_loss(self, gfl_head_outs, gt_meta, soft_label_list,
                 soft_targets_list):
        cls_logits, bboxes_reg = gfl_head_outs

        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_logits
        ]

        grid_cells_list = self._images_to_levels(gt_meta['grid_cells'],
                                                 num_level_anchors)

        labels_list = self._images_to_levels(gt_meta['labels'],
                                             num_level_anchors)

        label_weights_list = self._images_to_levels(gt_meta['label_weights'],
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(gt_meta['bbox_targets'],
                                                   num_level_anchors)
        # vlr regions                                         
        vlr_regions_list = self._images_to_levels(gt_meta['vlr_regions'],
                                                  num_level_anchors)

        num_total_pos = sum(gt_meta['pos_num'])
        try:
            num_total_pos = paddle.distributed.all_reduce(num_total_pos.clone(
            )) / paddle.distributed.get_world_size()
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_qfl_list, loss_ld_list, avg_factor = [], [], [], [], []
        loss_ld_vlr_list, loss_kd_list = [], []

        for cls_score, bbox_pred, grid_cells, labels, label_weights, bbox_targets, stride, soft_targets,\
                soft_label, vlr_region in zip(
                cls_logits, bboxes_reg, grid_cells_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride, soft_targets_list,
                soft_label_list, vlr_regions_list):

            grid_cells = grid_cells.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])

            soft_targets = soft_targets.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])

            soft_label = soft_label.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])

            # feture im
            # teacher_x = teacher_x.transpose([0, 2, 3, 1]).reshape([-1, 256])
            # x = x.transpose([0, 2, 3, 1]).reshape([-1, 256])  

            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])
            label_weights = label_weights.reshape([-1])

            vlr_region = vlr_region.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = paddle.nonzero(
                paddle.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            score = np.zeros(labels.shape)

            remain_inds = (vlr_region > 0).nonzero()

            if len(pos_inds) > 0:
                pos_bbox_targets = paddle.gather(bbox_targets, pos_inds, axis=0)
                pos_bbox_pred = paddle.gather(bbox_pred, pos_inds, axis=0)
                pos_grid_cells = paddle.gather(grid_cells, pos_inds, axis=0)

                pos_grid_cell_centers = self._grid_cells_to_center(
                    pos_grid_cells) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = paddle.gather(
                    weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_grid_cell_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)
                score[pos_inds.numpy()] = bbox_iou
                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])

                pos_soft_targets = paddle.gather(soft_targets, pos_inds, axis=0)
                soft_corners = pos_soft_targets.reshape([-1, self.reg_max + 1])

                target_corners = bbox2distance(pos_grid_cell_centers,
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

                # ld loss
                loss_ld = self.loss_ld(
                    pred_corners,
                    soft_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)

                loss_kd = self.loss_kd(
                    paddle.gather(
                        cls_score, pos_inds, axis=0),
                    paddle.gather(
                        soft_label, pos_inds, axis=0),
                    weight=paddle.gather(
                        label_weights, pos_inds, axis=0),
                    avg_factor=pos_inds.shape[0])

            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                loss_ld = bbox_pred.sum() * 0
                loss_kd = bbox_pred.sum() * 0
                weight_targets = paddle.to_tensor([0], dtype='float32')

            if len(remain_inds) > 0:
                neg_pred_corners = bbox_pred[remain_inds].reshape(
                    [-1, self.reg_max + 1])
                neg_soft_corners = soft_targets[remain_inds].reshape(
                    [-1, self.reg_max + 1])

                remain_targets = vlr_region[remain_inds]

                loss_ld_vlr = self.loss_ld_vlr(
                    neg_pred_corners,
                    neg_soft_corners,
                    weight=remain_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=16.0)
            else:
                loss_ld_vlr = bbox_pred.sum() * 0

            # qfl loss
            score = paddle.to_tensor(score)
            loss_qfl = self.loss_qfl(
                cls_score, (labels, score),
                weight=label_weights,
                avg_factor=num_total_pos)

            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_qfl_list.append(loss_qfl)
            loss_ld_list.append(loss_ld)
            loss_ld_vlr_list.append(loss_ld_vlr)
            loss_kd_list.append(loss_kd)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)  # + 1e-6
        try:
            avg_factor_clone = avg_factor.clone()
            tmp_avg_factor = paddle.distributed.all_reduce(avg_factor_clone)
            if tmp_avg_factor is not None:
                avg_factor = tmp_avg_factor
            else:
                avg_factor = avg_factor_clone
            avg_factor = paddle.clip(
                avg_factor / paddle.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)

        if avg_factor <= 0:
            loss_qfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = paddle.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_ld = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_ld_vlr = paddle.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_kd = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_qfl = sum(loss_qfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)
            loss_ld = sum(loss_ld_list)
            loss_ld_vlr = sum(loss_ld_vlr_list)
            loss_kd = sum(loss_kd_list)

        loss_states = dict(
            loss_qfl=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            loss_ld=loss_ld,
            loss_ld_vlr=loss_ld_vlr,
            loss_kd=loss_kd)

        return loss_states
