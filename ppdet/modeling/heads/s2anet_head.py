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
#
# The code is based on https://github.com/csuhan/s2anet/blob/master/mmdet/models/anchor_heads_rotated/s2anet_head.py

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant
from ppdet.core.workspace import register
from ppdet.modeling.proposal_generator.target_layer import RBoxAssigner
from ppdet.modeling.proposal_generator.anchor_generator import S2ANetAnchorGenerator
from ppdet.modeling.layers import AlignConv
from ..cls_utils import _get_class_default_kwargs
import numpy as np


@register
class S2ANetHead(nn.Layer):
    """
    S2Anet head
    Args:
        stacked_convs (int): number of stacked_convs
        feat_in (int): input channels of feat
        feat_out (int): output channels of feat
        num_classes (int): num_classes
        anchor_strides (list): stride of anchors
        anchor_scales (list): scale of anchors
        anchor_ratios (list): ratios of anchors
        target_means (list): target_means
        target_stds (list): target_stds
        align_conv_type (str): align_conv_type ['Conv', 'AlignConv']
        align_conv_size (int): kernel size of align_conv
        use_sigmoid_cls (bool): use sigmoid_cls or not
        reg_loss_weight (list): loss weight for regression
    """
    __shared__ = ['num_classes']
    __inject__ = ['anchor_assign', 'nms']

    def __init__(self,
                 stacked_convs=2,
                 feat_in=256,
                 feat_out=256,
                 num_classes=15,
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_scales=[4],
                 anchor_ratios=[1.0],
                 target_means=0.0,
                 target_stds=1.0,
                 align_conv_type='AlignConv',
                 align_conv_size=3,
                 use_sigmoid_cls=True,
                 anchor_assign=_get_class_default_kwargs(RBoxAssigner),
                 reg_loss_weight=[1.0, 1.0, 1.0, 1.0, 1.1],
                 cls_loss_weight=[1.1, 1.05],
                 reg_loss_type='l1',
                 nms_pre=2000,
                 nms='MultiClassNMS'):
        super(S2ANetHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.anchor_list = None
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_strides = paddle.to_tensor(anchor_strides)
        self.anchor_base_sizes = list(anchor_strides)
        self.means = paddle.ones(shape=[5]) * target_means
        self.stds = paddle.ones(shape=[5]) * target_stds
        assert align_conv_type in ['AlignConv', 'Conv', 'DCN']
        self.align_conv_type = align_conv_type
        self.align_conv_size = align_conv_size

        self.use_sigmoid_cls = use_sigmoid_cls
        self.cls_out_channels = num_classes if self.use_sigmoid_cls else num_classes + 1
        self.sampling = False
        self.anchor_assign = anchor_assign
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.alpha = 1.0
        self.beta = 1.0
        self.reg_loss_type = reg_loss_type
        self.nms_pre = nms_pre
        self.nms = nms
        self.fake_bbox = paddle.to_tensor(
            np.array(
                [[-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype='float32'))
        self.fake_bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))

        # anchor
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                S2ANetAnchorGenerator(anchor_base, anchor_scales,
                                      anchor_ratios))

        self.anchor_generators = nn.LayerList(self.anchor_generators)
        self.fam_cls_convs = nn.Sequential()
        self.fam_reg_convs = nn.Sequential()

        for i in range(self.stacked_convs):
            chan_in = self.feat_in if i == 0 else self.feat_out

            self.fam_cls_convs.add_sublayer(
                'fam_cls_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=chan_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.fam_cls_convs.add_sublayer('fam_cls_conv_{}_act'.format(i),
                                            nn.ReLU())

            self.fam_reg_convs.add_sublayer(
                'fam_reg_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=chan_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.fam_reg_convs.add_sublayer('fam_reg_conv_{}_act'.format(i),
                                            nn.ReLU())

        self.fam_reg = nn.Conv2D(
            self.feat_out,
            5,
            1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(0)))
        prior_prob = 0.01
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.fam_cls = nn.Conv2D(
            self.feat_out,
            self.cls_out_channels,
            1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(bias_init)))

        if self.align_conv_type == "AlignConv":
            self.align_conv = AlignConv(self.feat_out, self.feat_out,
                                        self.align_conv_size)
        elif self.align_conv_type == "Conv":
            self.align_conv = nn.Conv2D(
                self.feat_out,
                self.feat_out,
                self.align_conv_size,
                padding=(self.align_conv_size - 1) // 2,
                bias_attr=ParamAttr(initializer=Constant(0)))

        elif self.align_conv_type == "DCN":
            self.align_conv_offset = nn.Conv2D(
                self.feat_out,
                2 * self.align_conv_size**2,
                1,
                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                bias_attr=ParamAttr(initializer=Constant(0)))

            self.align_conv = paddle.vision.ops.DeformConv2D(
                self.feat_out,
                self.feat_out,
                self.align_conv_size,
                padding=(self.align_conv_size - 1) // 2,
                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                bias_attr=False)

        self.or_conv = nn.Conv2D(
            self.feat_out,
            self.feat_out,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(0)))

        # ODM
        self.odm_cls_convs = nn.Sequential()
        self.odm_reg_convs = nn.Sequential()

        for i in range(self.stacked_convs):
            ch_in = self.feat_out
            # ch_in = int(self.feat_out / 8) if i == 0 else self.feat_out

            self.odm_cls_convs.add_sublayer(
                'odm_cls_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=ch_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.odm_cls_convs.add_sublayer('odm_cls_conv_{}_act'.format(i),
                                            nn.ReLU())

            self.odm_reg_convs.add_sublayer(
                'odm_reg_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=self.feat_out,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.odm_reg_convs.add_sublayer('odm_reg_conv_{}_act'.format(i),
                                            nn.ReLU())

        self.odm_cls = nn.Conv2D(
            self.feat_out,
            self.cls_out_channels,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(bias_init)))
        self.odm_reg = nn.Conv2D(
            self.feat_out,
            5,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(0)))

    def forward(self, feats, targets=None):
        fam_reg_list, fam_cls_list = [], []
        odm_reg_list, odm_cls_list = [], []
        num_anchors_list, base_anchors_list, refine_anchors_list = [], [], []

        for i, feat in enumerate(feats):
            # get shape
            B = feat.shape[0]
            H, W = feat.shape[2], feat.shape[3]

            NA = H * W
            num_anchors_list.append(NA)

            fam_cls_feat = self.fam_cls_convs(feat)
            fam_cls = self.fam_cls(fam_cls_feat)
            # [N, CLS, H, W] --> [N, H, W, CLS]
            fam_cls = fam_cls.transpose([0, 2, 3, 1]).reshape(
                [B, NA, self.cls_out_channels])
            fam_cls_list.append(fam_cls)

            fam_reg_feat = self.fam_reg_convs(feat)
            fam_reg = self.fam_reg(fam_reg_feat)
            # [N, 5, H, W] --> [N, H, W, 5]
            fam_reg = fam_reg.transpose([0, 2, 3, 1]).reshape([B, NA, 5])
            fam_reg_list.append(fam_reg)

            # prepare anchor
            init_anchors = self.anchor_generators[i]((H, W),
                                                     self.anchor_strides[i])
            init_anchors = init_anchors.reshape([1, NA, 5])
            base_anchors_list.append(init_anchors.squeeze(0))

            if self.training:
                refine_anchor = self.bbox_decode(fam_reg.detach(), init_anchors)
            else:
                refine_anchor = self.bbox_decode(fam_reg, init_anchors)

            refine_anchors_list.append(refine_anchor)

            if self.align_conv_type == 'AlignConv':
                align_feat = self.align_conv(feat,
                                             refine_anchor.clone(), (H, W),
                                             self.anchor_strides[i])
            elif self.align_conv_type == 'DCN':
                align_offset = self.align_conv_offset(feat)
                align_feat = self.align_conv(feat, align_offset)
            elif self.align_conv_type == 'Conv':
                align_feat = self.align_conv(feat)

            or_feat = self.or_conv(align_feat)
            odm_reg_feat = or_feat
            odm_cls_feat = or_feat

            odm_reg_feat = self.odm_reg_convs(odm_reg_feat)
            odm_cls_feat = self.odm_cls_convs(odm_cls_feat)

            odm_cls = self.odm_cls(odm_cls_feat)
            # [N, CLS, H, W] --> [N, H, W, CLS]
            odm_cls = odm_cls.transpose([0, 2, 3, 1]).reshape(
                [B, NA, self.cls_out_channels])
            odm_cls_list.append(odm_cls)

            odm_reg = self.odm_reg(odm_reg_feat)
            # [N, 5, H, W] --> [N, H, W, 5]
            odm_reg = odm_reg.transpose([0, 2, 3, 1]).reshape([B, NA, 5])
            odm_reg_list.append(odm_reg)

        if self.training:
            return self.get_loss([
                fam_cls_list, fam_reg_list, odm_cls_list, odm_reg_list,
                num_anchors_list, base_anchors_list, refine_anchors_list
            ], targets)
        else:
            odm_bboxes_list = []
            for odm_reg, refine_anchor in zip(odm_reg_list,
                                              refine_anchors_list):
                odm_bboxes = self.bbox_decode(odm_reg, refine_anchor)
                odm_bboxes_list.append(odm_bboxes)
            return [odm_bboxes_list, odm_cls_list]

    def get_bboxes(self, head_outs):
        perd_bboxes_list, pred_scores_list = head_outs
        batch = pred_scores_list[0].shape[0]
        bboxes, bbox_num = [], []
        for i in range(batch):
            pred_scores_per_image = [t[i] for t in pred_scores_list]
            pred_bboxes_per_image = [t[i] for t in perd_bboxes_list]
            bbox_per_image, bbox_num_per_image = self.get_bboxes_single(
                pred_scores_per_image, pred_bboxes_per_image)
            bboxes.append(bbox_per_image)
            bbox_num.append(bbox_num_per_image)

        bboxes = paddle.concat(bboxes)
        bbox_num = paddle.concat(bbox_num)
        return bboxes, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to
        get final prediction.
        Args:
            bboxes(Tensor): bboxes [N, 10]
            bbox_num(Tensor): bbox_num
            im_shape(Tensor): [1 2]
            scale_factor(Tensor): [1 2]
        Returns:
            bbox_pred(Tensor): The output is the prediction with shape [N, 8]
                               including labels, scores and bboxes. The size of
                               bboxes are corresponding to the original image.
        """
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

        origin_shape_list = []
        scale_factor_list = []
        # scale_factor: scale_y, scale_x
        for i in range(bbox_num.shape[0]):
            expand_shape = paddle.expand(origin_shape[i:i + 1, :],
                                         [bbox_num[i], 2])
            scale_y, scale_x = scale_factor[i, 0:1], scale_factor[i, 1:2]
            scale = paddle.concat([
                scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x,
                scale_y
            ])
            expand_scale = paddle.expand(scale, [bbox_num[i], 8])
            origin_shape_list.append(expand_shape)
            scale_factor_list.append(expand_scale)

        origin_shape_list = paddle.concat(origin_shape_list)
        scale_factor_list = paddle.concat(scale_factor_list)

        # bboxes: [N, 10], label, score, bbox
        pred_label_score = bboxes[:, 0:2]
        pred_bbox = bboxes[:, 2:]

        # rescale bbox to original image
        pred_bbox = pred_bbox.reshape([-1, 8])
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = origin_shape_list[:, 0]
        origin_w = origin_shape_list[:, 1]

        bboxes = scaled_bbox
        zeros = paddle.zeros_like(origin_h)
        x1 = paddle.maximum(paddle.minimum(bboxes[:, 0], origin_w - 1), zeros)
        y1 = paddle.maximum(paddle.minimum(bboxes[:, 1], origin_h - 1), zeros)
        x2 = paddle.maximum(paddle.minimum(bboxes[:, 2], origin_w - 1), zeros)
        y2 = paddle.maximum(paddle.minimum(bboxes[:, 3], origin_h - 1), zeros)
        x3 = paddle.maximum(paddle.minimum(bboxes[:, 4], origin_w - 1), zeros)
        y3 = paddle.maximum(paddle.minimum(bboxes[:, 5], origin_h - 1), zeros)
        x4 = paddle.maximum(paddle.minimum(bboxes[:, 6], origin_w - 1), zeros)
        y4 = paddle.maximum(paddle.minimum(bboxes[:, 7], origin_h - 1), zeros)
        pred_bbox = paddle.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)
        pred_result = paddle.concat([pred_label_score, pred_bbox], axis=1)
        return pred_result

    def get_bboxes_single(self, cls_score_list, bbox_pred_list):
        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred in zip(cls_score_list, bbox_pred_list):
            if self.use_sigmoid_cls:
                scores = F.sigmoid(cls_score)
            else:
                scores = F.softmax(cls_score, axis=-1)

            if scores.shape[0] > self.nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores = paddle.max(scores, axis=1)
                else:
                    max_scores = paddle.max(scores[:, :-1], axis=1)

                topk_val, topk_inds = paddle.topk(max_scores, self.nms_pre)
                bbox_pred = paddle.gather(bbox_pred, topk_inds)
                scores = paddle.gather(scores, topk_inds)

            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)

        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        mlvl_scores = paddle.concat(mlvl_scores)

        mlvl_polys = self.rbox2poly(mlvl_bboxes).unsqueeze(0)
        mlvl_scores = paddle.transpose(mlvl_scores, [1, 0]).unsqueeze(0)

        bbox, bbox_num, _ = self.nms(mlvl_polys, mlvl_scores)
        if bbox.shape[0] <= 0:
            bbox = self.fake_bbox
            bbox_num = self.fake_bbox_num

        return bbox, bbox_num

    def smooth_l1_loss(self, pred, label, delta=1.0 / 9.0):
        """
        Args:
            pred: pred score
            label: label
            delta: delta
        Returns: loss
        """
        assert pred.shape == label.shape and label.numel() > 0
        assert delta > 0
        diff = paddle.abs(pred - label)
        loss = paddle.where(diff < delta, 0.5 * diff * diff / delta,
                            diff - 0.5 * delta)
        return loss

    def get_fam_loss(self, fam_target, s2anet_head_out, reg_loss_type='l1'):
        (labels, label_weights, bbox_targets, bbox_weights, bbox_gt_bboxes,
         pos_inds, neg_inds) = fam_target
        fam_cls_branch_list, fam_reg_branch_list, odm_cls_branch_list, odm_reg_branch_list, num_anchors_list = s2anet_head_out

        fam_cls_losses = []
        fam_bbox_losses = []
        st_idx = 0
        num_total_samples = len(pos_inds) + len(
            neg_inds) if self.sampling else len(pos_inds)
        num_total_samples = max(1, num_total_samples)

        for idx, feat_anchor_num in enumerate(num_anchors_list):
            # step1:  get data
            feat_labels = labels[st_idx:st_idx + feat_anchor_num]
            feat_label_weights = label_weights[st_idx:st_idx + feat_anchor_num]

            feat_bbox_targets = bbox_targets[st_idx:st_idx + feat_anchor_num, :]
            feat_bbox_weights = bbox_weights[st_idx:st_idx + feat_anchor_num, :]

            # step2: calc cls loss
            feat_labels = feat_labels.reshape(-1)
            feat_label_weights = feat_label_weights.reshape(-1)

            fam_cls_score = fam_cls_branch_list[idx]
            fam_cls_score = paddle.squeeze(fam_cls_score, axis=0)
            fam_cls_score1 = fam_cls_score

            feat_labels = paddle.to_tensor(feat_labels)
            feat_labels_one_hot = paddle.nn.functional.one_hot(
                feat_labels, self.cls_out_channels + 1)
            feat_labels_one_hot = feat_labels_one_hot[:, 1:]
            feat_labels_one_hot.stop_gradient = True

            num_total_samples = paddle.to_tensor(
                num_total_samples, dtype='float32', stop_gradient=True)

            fam_cls = F.sigmoid_focal_loss(
                fam_cls_score1,
                feat_labels_one_hot,
                normalizer=num_total_samples,
                reduction='none')

            feat_label_weights = feat_label_weights.reshape(
                feat_label_weights.shape[0], 1)
            feat_label_weights = np.repeat(
                feat_label_weights, self.cls_out_channels, axis=1)
            feat_label_weights = paddle.to_tensor(
                feat_label_weights, stop_gradient=True)

            fam_cls = fam_cls * feat_label_weights
            fam_cls_total = paddle.sum(fam_cls)
            fam_cls_losses.append(fam_cls_total)

            # step3: regression loss
            feat_bbox_targets = paddle.to_tensor(
                feat_bbox_targets, dtype='float32', stop_gradient=True)
            feat_bbox_targets = paddle.reshape(feat_bbox_targets, [-1, 5])

            fam_bbox_pred = fam_reg_branch_list[idx]
            fam_bbox_pred = paddle.squeeze(fam_bbox_pred, axis=0)
            fam_bbox_pred = paddle.reshape(fam_bbox_pred, [-1, 5])
            fam_bbox = self.smooth_l1_loss(fam_bbox_pred, feat_bbox_targets)
            loss_weight = paddle.to_tensor(
                self.reg_loss_weight, dtype='float32', stop_gradient=True)
            fam_bbox = paddle.multiply(fam_bbox, loss_weight)
            feat_bbox_weights = paddle.to_tensor(
                feat_bbox_weights, stop_gradient=True)

            fam_bbox = fam_bbox * feat_bbox_weights
            fam_bbox_total = paddle.sum(fam_bbox) / num_total_samples
            fam_bbox_losses.append(fam_bbox_total)
            st_idx += feat_anchor_num

        fam_cls_loss = paddle.add_n(fam_cls_losses)
        fam_cls_loss_weight = paddle.to_tensor(
            self.cls_loss_weight[0], dtype='float32', stop_gradient=True)
        fam_cls_loss = fam_cls_loss * fam_cls_loss_weight
        fam_reg_loss = paddle.add_n(fam_bbox_losses)
        return fam_cls_loss, fam_reg_loss

    def get_odm_loss(self, odm_target, s2anet_head_out, reg_loss_type='l1'):
        (labels, label_weights, bbox_targets, bbox_weights, bbox_gt_bboxes,
         pos_inds, neg_inds) = odm_target
        fam_cls_branch_list, fam_reg_branch_list, odm_cls_branch_list, odm_reg_branch_list, num_anchors_list = s2anet_head_out

        odm_cls_losses = []
        odm_bbox_losses = []
        st_idx = 0
        num_total_samples = len(pos_inds) + len(
            neg_inds) if self.sampling else len(pos_inds)
        num_total_samples = max(1, num_total_samples)

        for idx, feat_anchor_num in enumerate(num_anchors_list):
            # step1:  get data
            feat_labels = labels[st_idx:st_idx + feat_anchor_num]
            feat_label_weights = label_weights[st_idx:st_idx + feat_anchor_num]

            feat_bbox_targets = bbox_targets[st_idx:st_idx + feat_anchor_num, :]
            feat_bbox_weights = bbox_weights[st_idx:st_idx + feat_anchor_num, :]

            # step2: calc cls loss
            feat_labels = feat_labels.reshape(-1)
            feat_label_weights = feat_label_weights.reshape(-1)

            odm_cls_score = odm_cls_branch_list[idx]
            odm_cls_score = paddle.squeeze(odm_cls_score, axis=0)
            odm_cls_score1 = odm_cls_score

            feat_labels = paddle.to_tensor(feat_labels)
            feat_labels_one_hot = paddle.nn.functional.one_hot(
                feat_labels, self.cls_out_channels + 1)
            feat_labels_one_hot = feat_labels_one_hot[:, 1:]
            feat_labels_one_hot.stop_gradient = True

            num_total_samples = paddle.to_tensor(
                num_total_samples, dtype='float32', stop_gradient=True)
            odm_cls = F.sigmoid_focal_loss(
                odm_cls_score1,
                feat_labels_one_hot,
                normalizer=num_total_samples,
                reduction='none')

            feat_label_weights = feat_label_weights.reshape(
                feat_label_weights.shape[0], 1)
            feat_label_weights = np.repeat(
                feat_label_weights, self.cls_out_channels, axis=1)
            feat_label_weights = paddle.to_tensor(feat_label_weights)
            feat_label_weights.stop_gradient = True

            odm_cls = odm_cls * feat_label_weights
            odm_cls_total = paddle.sum(odm_cls)
            odm_cls_losses.append(odm_cls_total)

            # # step3: regression loss
            feat_bbox_targets = paddle.to_tensor(
                feat_bbox_targets, dtype='float32')
            feat_bbox_targets = paddle.reshape(feat_bbox_targets, [-1, 5])
            feat_bbox_targets.stop_gradient = True

            odm_bbox_pred = odm_reg_branch_list[idx]
            odm_bbox_pred = paddle.squeeze(odm_bbox_pred, axis=0)
            odm_bbox_pred = paddle.reshape(odm_bbox_pred, [-1, 5])
            odm_bbox = self.smooth_l1_loss(odm_bbox_pred, feat_bbox_targets)

            loss_weight = paddle.to_tensor(
                self.reg_loss_weight, dtype='float32', stop_gradient=True)
            odm_bbox = paddle.multiply(odm_bbox, loss_weight)
            feat_bbox_weights = paddle.to_tensor(
                feat_bbox_weights, stop_gradient=True)

            odm_bbox = odm_bbox * feat_bbox_weights
            odm_bbox_total = paddle.sum(odm_bbox) / num_total_samples

            odm_bbox_losses.append(odm_bbox_total)
            st_idx += feat_anchor_num

        odm_cls_loss = paddle.add_n(odm_cls_losses)
        odm_cls_loss_weight = paddle.to_tensor(
            self.cls_loss_weight[1], dtype='float32', stop_gradient=True)
        odm_cls_loss = odm_cls_loss * odm_cls_loss_weight
        odm_reg_loss = paddle.add_n(odm_bbox_losses)
        return odm_cls_loss, odm_reg_loss

    def get_loss(self, head_outs, inputs):
        fam_cls_list, fam_reg_list, odm_cls_list, odm_reg_list, \
            num_anchors_list, base_anchors_list, refine_anchors_list = head_outs

        # compute loss
        fam_cls_loss_lst = []
        fam_reg_loss_lst = []
        odm_cls_loss_lst = []
        odm_reg_loss_lst = []

        batch = len(inputs['gt_rbox'])
        for i in range(batch):
            # data_format: (xc, yc, w, h, theta)
            gt_mask = inputs['pad_gt_mask'][i, :, 0]
            gt_idx = paddle.nonzero(gt_mask).squeeze(-1)
            gt_bboxes = paddle.gather(inputs['gt_rbox'][i], gt_idx).numpy()
            gt_labels = paddle.gather(inputs['gt_class'][i], gt_idx).numpy()
            is_crowd = paddle.gather(inputs['is_crowd'][i], gt_idx).numpy()
            gt_labels = gt_labels + 1

            anchors_per_image = np.concatenate(base_anchors_list)

            fam_cls_per_image = [t[i] for t in fam_cls_list]
            fam_reg_per_image = [t[i] for t in fam_reg_list]
            odm_cls_per_image = [t[i] for t in odm_cls_list]
            odm_reg_per_image = [t[i] for t in odm_reg_list]
            im_s2anet_head_out = (fam_cls_per_image, fam_reg_per_image,
                                  odm_cls_per_image, odm_reg_per_image,
                                  num_anchors_list)
            # FAM
            im_fam_target = self.anchor_assign(anchors_per_image, gt_bboxes,
                                               gt_labels, is_crowd)
            if im_fam_target is not None:
                im_fam_cls_loss, im_fam_reg_loss = self.get_fam_loss(
                    im_fam_target, im_s2anet_head_out, self.reg_loss_type)
                fam_cls_loss_lst.append(im_fam_cls_loss)
                fam_reg_loss_lst.append(im_fam_reg_loss)

            # ODM
            refine_anchors_per_image = [t[i] for t in refine_anchors_list]
            refine_anchors_per_image = paddle.concat(
                refine_anchors_per_image).numpy()
            im_odm_target = self.anchor_assign(refine_anchors_per_image,
                                               gt_bboxes, gt_labels, is_crowd)

            if im_odm_target is not None:
                im_odm_cls_loss, im_odm_reg_loss = self.get_odm_loss(
                    im_odm_target, im_s2anet_head_out, self.reg_loss_type)
                odm_cls_loss_lst.append(im_odm_cls_loss)
                odm_reg_loss_lst.append(im_odm_reg_loss)

        fam_cls_loss = paddle.add_n(fam_cls_loss_lst) / batch
        fam_reg_loss = paddle.add_n(fam_reg_loss_lst) / batch
        odm_cls_loss = paddle.add_n(odm_cls_loss_lst) / batch
        odm_reg_loss = paddle.add_n(odm_reg_loss_lst) / batch
        loss = fam_cls_loss + fam_reg_loss + odm_cls_loss + odm_reg_loss

        return {
            'loss': loss,
            'fam_cls_loss': fam_cls_loss,
            'fam_reg_loss': fam_reg_loss,
            'odm_cls_loss': odm_cls_loss,
            'odm_reg_loss': odm_reg_loss
        }

    def bbox_decode(self, preds, anchors, wh_ratio_clip=1e-6):
        """decode bbox from deltas
        Args:
            preds: [B, L, 5]
            anchors: [1, L, 5]
        return:
            bboxes: [B, L, 5]
        """
        preds = paddle.add(paddle.multiply(preds, self.stds), self.means)

        dx, dy, dw, dh, dangle = paddle.split(preds, 5, axis=-1)
        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = paddle.clip(dw, min=-max_ratio, max=max_ratio)
        dh = paddle.clip(dh, min=-max_ratio, max=max_ratio)

        rroi_x, rroi_y, rroi_w, rroi_h, rroi_angle = paddle.split(
            anchors, 5, axis=-1)

        gx = dx * rroi_w * paddle.cos(rroi_angle) - dy * rroi_h * paddle.sin(
            rroi_angle) + rroi_x
        gy = dx * rroi_w * paddle.sin(rroi_angle) + dy * rroi_h * paddle.cos(
            rroi_angle) + rroi_y
        gw = rroi_w * dw.exp()
        gh = rroi_h * dh.exp()
        ga = np.pi * dangle + rroi_angle
        ga = (ga + np.pi / 4) % np.pi - np.pi / 4
        bboxes = paddle.concat([gx, gy, gw, gh, ga], axis=-1)
        return bboxes

    def rbox2poly(self, rboxes):
        """
        rboxes: [x_ctr,y_ctr,w,h,angle]
        to
        polys: [x0,y0,x1,y1,x2,y2,x3,y3]
        """
        N = rboxes.shape[0]

        x_ctr = rboxes[:, 0]
        y_ctr = rboxes[:, 1]
        width = rboxes[:, 2]
        height = rboxes[:, 3]
        angle = rboxes[:, 4]

        tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5

        normal_rects = paddle.stack(
            [tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y], axis=0)
        normal_rects = paddle.reshape(normal_rects, [2, 4, N])
        normal_rects = paddle.transpose(normal_rects, [2, 0, 1])

        sin, cos = paddle.sin(angle), paddle.cos(angle)
        # M: [N,2,2]
        M = paddle.stack([cos, -sin, sin, cos], axis=0)
        M = paddle.reshape(M, [2, 2, N])
        M = paddle.transpose(M, [2, 0, 1])

        # polys: [N,8]
        polys = paddle.matmul(M, normal_rects)
        polys = paddle.transpose(polys, [2, 1, 0])
        polys = paddle.reshape(polys, [-1, N])
        polys = paddle.transpose(polys, [1, 0])

        tmp = paddle.stack(
            [x_ctr, y_ctr, x_ctr, y_ctr, x_ctr, y_ctr, x_ctr, y_ctr], axis=1)
        polys = polys + tmp
        return polys
