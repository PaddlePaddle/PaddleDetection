# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant
from ppdet.modeling.proposal_generator import AnchorGenerator
from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer
from .fcos_head import ScaleReg
from ppdet.modeling.bbox_utils import distance2bbox, bbox2distance, batch_distance2bbox, bbox_center
from paddle.fluid.dygraph import parallel_helper
from ppdet.data.transform.atss_assigner import bbox_overlaps

__all__ = ['SCRFDHead']


def batch_kps2distance(points, kps, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points: (Tensor): boxes center with shape (N, 2), "x, y" format.
        kps (Tensor): Shape (n, K), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """

    preds = []
    points = paddle.tile(
        points, repeat_times=[kps.shape[0] // points.shape[0], 1])
    for i in range(0, kps.shape[1], 2):
        px = kps[:, i] - points[:, i % 2]
        py = kps[:, i + 1] - points[:, i % 2 + 1]
        if max_dis is not None:
            px = paddle.clip(px, min=0, max=max_dis - eps)
            py = paddle.clip(py, min=0, max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return paddle.stack(preds, -1)
    # kps[..., 0::2] -= points[..., 0].unsqueeze(axis=-1)
    # kps[..., 1::2] -= points[..., 1].unsqueeze(axis=-1)
    # if max_dis is not None:
    #     kps = paddle.clip(kps, min=0, max=max_dis - eps)
    # return kps


def batch_distance2kps(points, kps, max_shape=None):
    preds = []
    points = paddle.tile(points, repeat_times=[kps.shape[0], 1])
    for i in range(0, kps.shape[2], 2):
        px = points[..., 0] + kps[..., i]
        py = points[..., 1] + kps[..., i + 1]
        if max_shape is not None:
            px = paddle.clip(px, min=0, max=max_shape[1])
            py = paddle.clip(py, min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return paddle.stack(preds, -1)


@register
class SCRFDFeat(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        share_cls_reg (bool): Whether to share the cls and reg output.
        act (str): The act of per layers.
        use_se (bool): Whether to use se module.
    """

    def __init__(self,
                 feat_in=24,
                 feat_out=64,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='gn',
                 norm_decay=0,
                 share_cls_reg=True,
                 fpn_stride_share=True,
                 act='relu',
                 norm_group=16,
                 use_dw_conv=False):
        super(SCRFDFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.norm_group = norm_group
        self.share_cls_reg = share_cls_reg
        self.fpn_stride_share = fpn_stride_share
        self.act = act
        self.feat_out = feat_out
        self.feat_in = feat_in
        self.cls_convs = []
        self.reg_convs = []
        if use_dw_conv:
            self.groups = feat_out
        else:
            self.groups = 1
        if fpn_stride_share:
            num_fpn_stride = 1

        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                if use_dw_conv:
                    cls_conv_dw = self.add_sublayer(
                        'cls_conv_dw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=3,
                            stride=1,
                            groups=feat_out,
                            norm_type=norm_type,
                            norm_groups=self.norm_groups,
                            bias_on=False,
                            lr_scale=1.))
                    cls_subnet_convs.append(cls_conv_dw)
                    cls_conv_pw = self.add_sublayer(
                        'cls_conv_pw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            norm_groups=self.norm_groups,
                            bias_on=False,
                            lr_scale=2.))
                    cls_subnet_convs.append(cls_conv_pw)
                else:
                    cls_conv = self.add_sublayer(
                        'cls_conv_{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=3,
                            stride=1,
                            groups=self.groups,
                            norm_type=norm_type,
                            norm_decay=norm_decay,
                            norm_groups=self.norm_group,
                            bias_on=False))
                    cls_subnet_convs.append(cls_conv)

                if not self.share_cls_reg:
                    if use_dw_conv:
                        reg_conv_dw = self.add_sublayer(
                            'reg_conv_dw{}.{}'.format(stage_idx, i),
                            ConvNormLayer(
                                ch_in=in_c,
                                ch_out=feat_out,
                                filter_size=5,
                                stride=1,
                                groups=feat_out,
                                norm_type=norm_type,
                                bias_on=False,
                                lr_scale=1.))
                        reg_subnet_convs.append(reg_conv_dw)
                        reg_conv_pw = self.add_sublayer(
                            'reg_conv_pw{}.{}'.format(stage_idx, i),
                            ConvNormLayer(
                                ch_in=in_c,
                                ch_out=feat_out,
                                filter_size=1,
                                stride=1,
                                norm_type=norm_type,
                                bias_on=False,
                                lr_scale=1.))
                        reg_subnet_convs.append(reg_conv_pw)
                    else:
                        reg_conv = self.add_sublayer(
                            'reg_conv_{}.{}'.format(stage_idx, i),
                            ConvNormLayer(
                                ch_in=in_c,
                                ch_out=feat_out,
                                filter_size=3,
                                stride=1,
                                groups=self.groups,
                                norm_type=norm_type,
                                norm_groups=self.norm_group,
                                bias_on=False))
                        reg_subnet_convs.append(reg_conv)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == "relu":
            x = F.relu(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        if not self.fpn_stride_share:
            assert stage_idx < len(self.cls_convs)
        else:
            stage_idx = 0

        cls_feat = fpn_feat
        reg_feat = fpn_feat
        kps_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            reg_feat = cls_feat
            kps_feat = cls_feat
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        return cls_feat, reg_feat, kps_feat


@register
class SCRFDHead(nn.Layer):
    """Used in RetinaNet proposed in paper https://arxiv.org/pdf/1708.02002.pdf
    """
    __inject__ = [
        'conv_feat', 'anchor_generator', 'bbox_assigner', 'loss_class',
        'loss_bbox', 'nms', 'loss_kps'
    ]

    def __init__(self,
                 num_classes=1,
                 conv_feat=None,
                 anchor_generator=None,
                 bbox_assigner=None,
                 loss_class=None,
                 loss_bbox=None,
                 nms_pre=1000,
                 nms=None,
                 use_kps=False,
                 num_kps=5,
                 loss_kps=None,
                 use_reg_scale=True):
        super(SCRFDHead, self).__init__()
        self.num_classes = num_classes
        # allow RetinaNet to use IoU based losses.
        self.conv_feat = conv_feat
        self.anchor_generator = anchor_generator
        self.bbox_assigner = bbox_assigner
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.nms_pre = nms_pre
        self.nms = nms
        self.cls_out_channels = num_classes
        self.use_reg_scale = use_reg_scale
        self.use_kps = use_kps
        self.NK = num_kps
        self.loss_kps = loss_kps
        self.init_layers()

    def init_layers(self):
        bias_init_value = -4.595
        num_anchors = self.anchor_generator.num_anchors
        self.num_anchors = num_anchors
        self.scrfd_cls = nn.Conv2D(
            in_channels=self.conv_feat.feat_out,
            out_channels=self.cls_out_channels * num_anchors,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=bias_init_value)))
        self.scrfd_reg = nn.Conv2D(
            in_channels=self.conv_feat.feat_out,
            out_channels=4 * num_anchors,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=0)))

        self.reg_scale = nn.LayerList()
        for i in range(len(self.anchor_generator.strides)):
            if self.use_reg_scale:
                self.reg_scale.append(ScaleReg())
            else:
                self.reg_scale.append(None)
        if self.use_kps:
            self.scrfd_kps = nn.Conv2D(
                in_channels=self.conv_feat.feat_out,
                out_channels=self.NK * 2 * num_anchors,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0)))

    def forward(self, neck_feats):
        # we use the same anchor for all images
        anchors, num_anchors_list, stride_tensor_list = self.anchor_generator(
            neck_feats, return_extra_info=True)
        cls_logits_list = []
        bboxes_pred_list = []
        kps_pred_list = []
        for i, neck_feat in enumerate(neck_feats):
            conv_cls_feat, conv_reg_feat, conv_kps_feat = self.conv_feat(
                neck_feat, i)
            cls_logits = self.scrfd_cls(conv_cls_feat)
            bbox_reg = self.scrfd_reg(conv_reg_feat)
            if self.use_kps:
                kps_pred = self.scrfd_kps(conv_kps_feat)
            else:
                kps_pred = paddle.zeros([
                    bbox_reg.shape[0], self.NK * 2 * self.num_anchors,
                    bbox_reg.shape[2], bbox_reg.shape[3]
                ])
            if self.reg_scale[i] is not None:
                bbox_reg = self.reg_scale[i](bbox_reg)

            cls_logits_list.append(
                cls_logits.transpose([0, 2, 3, 1]).reshape(
                    [0, -1, self.cls_out_channels]))
            anchor_center = bbox_center(anchors[
                i]) / self.anchor_generator.strides[i]
            anchor_center = paddle.broadcast_to(anchor_center, [
                bbox_reg.shape[0], anchor_center.shape[0],
                anchor_center.shape[1]
            ])
            bbox_reg = bbox_reg.transpose([0, 2, 3, 1]).reshape([0, -1, 4])
            bbox_pred = batch_distance2bbox(anchor_center, bbox_reg)
            if not self.training:
                bbox_pred *= self.anchor_generator.strides[i]
            bboxes_pred_list.append(bbox_pred)
            if self.use_kps:
                kps_pred = kps_pred.transpose([0, 2, 3, 1]).reshape(
                    [0, -1, self.NK * 2])
                if not self.training:
                    kps_pred = batch_distance2kps(anchor_center, kps_pred)
                    kps_pred *= self.anchor_generator.strides[i]
                kps_pred_list.append(kps_pred)

        cls_logits = paddle.concat(cls_logits_list, axis=1)
        bboxes_pred = paddle.concat(bboxes_pred_list, axis=1)
        if self.use_kps:
            kpses_pred = paddle.concat(kps_pred_list, axis=1)
        else:
            kpses_pred = None
        anchors = paddle.concat(anchors, axis=0)
        stride_tensor_list = paddle.concat(stride_tensor_list, axis=0)
        stride_tensor_list = paddle.unsqueeze(stride_tensor_list, axis=0)
        return (cls_logits, bboxes_pred, anchors, num_anchors_list,
                stride_tensor_list, kpses_pred)

    def get_loss(self, head_outputs, gt_meta):
        """Here we calculate loss for a batch of images.
        We assign anchors to gts in each image and gather all the assigned
        postive and negative samples. Then loss is calculated on the gathered
        samples.
        """
        cls_logits, bboxes_pred, anchors, num_anchors_list, stride_tensor_list, kpses_pred = head_outputs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        gt_kps_all = gt_meta['gt_keypoint']
        bs, num, num_kps, len_kps = gt_kps_all.shape
        gt_kps = gt_kps_all[..., :2].reshape([bs, num, -1])
        gt_kps_mask = gt_kps_all[..., 2].reshape(
            [bs, num, -1])[:, :, 0].unsqueeze(-1)
        gt_kps = gt_kps * (gt_kps_mask * pad_gt_mask)

        assigned_labels, assigned_bboxes, assigned_scores, assigned_kps = self.bbox_assigner(
            anchors,
            num_anchors_list,
            gt_labels,
            gt_bboxes,
            pad_gt_mask,
            bg_index=self.num_classes,
            pred_bboxes=bboxes_pred.detach() * stride_tensor_list,
            gt_kps=gt_kps)
        #  # rescale bbox
        assigned_bboxes /= stride_tensor_list
        assigned_kps /= stride_tensor_list
        flatten_cls_preds = cls_logits.reshape([-1, self.num_classes])
        flatten_bboxes = bboxes_pred.reshape([-1, 4])
        flatten_bbox_targets = assigned_bboxes.reshape([-1, 4])
        flatten_labels = assigned_labels.reshape([-1])
        flatten_assigned_scores = assigned_scores.reshape(
            [-1, self.num_classes])
        pos_inds = paddle.nonzero(
            paddle.logical_and((flatten_labels >= 0),
                               (flatten_labels < self.num_classes)),
            as_tuple=False).squeeze(1)
        num_total_pos = len(pos_inds)

        if num_total_pos > 0:
            pos_bbox_targets = paddle.gather(
                flatten_bbox_targets, pos_inds, axis=0)
            pos_decode_bbox_pred = paddle.gather(
                flatten_bboxes, pos_inds, axis=0)

            pos_cls_pred = paddle.gather(
                flatten_assigned_scores, pos_inds, axis=0)
            weight_targets = pos_cls_pred.detach()
            weight_targets = F.sigmoid(1 - weight_targets)
            # weight_targets = F.sigmoid(weight_targets)
            #  weight_targets = paddle.gather(
            #      weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)

            # regression loss
            loss_iou = paddle.sum(
                self.loss_bbox(pos_decode_bbox_pred, pos_bbox_targets,
                               weight_targets))
            # cal avg_factor
            avg_factor = weight_targets.sum()
            if paddle.fluid.core.is_compiled_with_dist(
            ) and parallel_helper._is_parallel_ctx_initialized():
                paddle.distributed.all_reduce(avg_factor)
                avg_factor = paddle.clip(
                    avg_factor / paddle.distributed.get_world_size(), min=1)
            loss_iou /= avg_factor

            if self.use_kps:
                flatten_kpses_pred = kpses_pred.reshape([-1, self.NK * 2])
                flatten_assigned_kps = assigned_kps.reshape([-1, self.NK * 2])
                pos_kps_pred_mask = paddle.gather(
                    flatten_assigned_kps, pos_inds,
                    axis=0).sum(axis=-1).unsqueeze(axis=-1)
                kpses_weight = paddle.where(pos_kps_pred_mask > 0,
                                            weight_targets,
                                            paddle.zeros_like(weight_targets))
                anchors_center = (bbox_center(anchors) /
                                  stride_tensor_list).squeeze(axis=0)
                pos_kps_targets = batch_kps2distance(anchors_center,
                                                     flatten_assigned_kps)
                pos_kps_targets = paddle.gather(
                    pos_kps_targets, pos_inds, axis=0)
                pos_kps_pred = paddle.gather(
                    flatten_kpses_pred, pos_inds, axis=0)

                loss_kps = self.loss_kps(pos_kps_pred,
                                         pos_kps_targets) * kpses_weight
                loss_kps = paddle.sum(loss_kps)
                if self.use_kps:
                    loss_kps /= avg_factor
            else:
                loss_kps = paddle.zeros([1])
        else:
            loss_iou = paddle.zeros([1])
            loss_kps = paddle.zeros([1])

        # classification loss
        num_total_pos = paddle.to_tensor(num_total_pos)
        if paddle.fluid.core.is_compiled_with_dist(
        ) and parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.all_reduce(num_total_pos)
            num_total_pos = paddle.clip(
                num_total_pos / paddle.distributed.get_world_size(), min=1)
        loss_cls = self.loss_class(
            flatten_cls_preds,
            (flatten_labels, paddle.flatten(assigned_scores).detach()),
            avg_factor=num_total_pos.detach())

        return {
            'loss_class': loss_cls,
            'loss_reg': loss_iou,
            'loss_kps': loss_kps
        }

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_bboxes, anchors, num_anchors_list, \
            stride_tensor_list, kps_pred = head_outs
        pred_scores = F.sigmoid(pred_scores.transpose([0, 2, 1]))

        for i in range(len(pred_bboxes)):
            pred_bboxes[i, :, 0] = pred_bboxes[i, :, 0].clip(
                min=0, max=img_shape[i, 1])
            pred_bboxes[i, :, 1] = pred_bboxes[i, :, 1].clip(
                min=0, max=img_shape[i, 0])
            pred_bboxes[i, :, 2] = pred_bboxes[i, :, 2].clip(
                min=0, max=img_shape[i, 1])
            pred_bboxes[i, :, 3] = pred_bboxes[i, :, 3].clip(
                min=0, max=img_shape[i, 0])
        # scale bbox to origin
        scale_factor = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num
