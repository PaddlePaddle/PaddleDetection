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

import math, paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant
from ppdet.modeling.proposal_generator import AnchorGenerator
from ppdet.core.workspace import register
from ppdet.modeling.heads.fcos_head import FCOSFeat

__all__ = ['RetinaHead']

@register
class RetinaFeat(FCOSFeat):
    """We use FCOSFeat to construct conv layers in RetinaNet.
    We rename FCOSFeat to RetinaFeat to avoid confusion.
    """
    pass

@register
class RetinaAnchorGenerator(AnchorGenerator):
    def __init__(self,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[8.0, 16.0, 32.0, 64.0, 128.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.0):
        anchor_sizes = []
        for s in strides:
            anchor_sizes.append([
                s * octave_base_scale * 2**(i/scales_per_octave) \
                for i in range(scales_per_octave)])
        super(RetinaAnchorGenerator, self).__init__(
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            strides=strides,
            variance=variance,
            offset=offset)

@register
class RetinaHead(nn.Layer):
    """Used in RetinaNet proposed in paper https://arxiv.org/pdf/1708.02002.pdf
    """
    __inject__ = [
        'conv_feat', 'anchor_generator', 'bbox_assigner',
        'bbox_coder', 'loss_class', 'loss_bbox', 'nms']
    def __init__(self,
                 num_classes=80,
                 prior_prob=0.01,
                 decode_reg_out=False,
                 conv_feat=None,
                 anchor_generator=None,
                 bbox_assigner=None,
                 bbox_coder=None,
                 loss_class=None,
                 loss_bbox=None,
                 nms_pre=1000,
                 nms=None):
        super(RetinaHead, self).__init__()
        self.num_classes = num_classes
        self.prior_prob = prior_prob
        # allow RetinaNet to use IoU based losses.
        self.decode_reg_out = decode_reg_out
        self.conv_feat = conv_feat
        self.anchor_generator = anchor_generator
        self.bbox_assigner = bbox_assigner
        self.bbox_coder = bbox_coder
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.nms_pre = nms_pre
        self.nms = nms
        self.cls_out_channels = num_classes
        self.init_layers()

    def init_layers(self):
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        num_anchors = self.anchor_generator.num_anchors
        self.retina_cls = nn.Conv2D(
            in_channels=self.conv_feat.feat_out,
            out_channels=self.cls_out_channels * num_anchors,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=bias_init_value)))
        self.retina_reg = nn.Conv2D(
            in_channels=self.conv_feat.feat_out,
            out_channels=4 * num_anchors,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=0)))

    def forward(self, neck_feats):
        cls_logits_list = []
        bboxes_reg_list = []
        for neck_feat in neck_feats:
            conv_cls_feat, conv_reg_feat = self.conv_feat(neck_feat)
            cls_logits = self.retina_cls(conv_cls_feat)
            bbox_reg = self.retina_reg(conv_reg_feat)
            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)
        return (cls_logits_list, bboxes_reg_list)

    def get_loss(self, head_outputs, meta):
        """Here we calculate loss for a batch of images.
        We assign anchors to gts in each image and gather all the assigned
        postive and negative samples. Then loss is calculated on the gathered
        samples.
        """
        cls_logits, bboxes_reg = head_outputs
        # we use the same anchor for all images
        anchors = self.anchor_generator(cls_logits)
        anchors = paddle.concat(anchors)

        # matches: contain gt_inds
        # match_labels: -1(ignore), 0(neg) or 1(pos)
        matches_list, match_labels_list = [], []
        # assign anchors to gts, no sampling is involved
        for gt_bbox in meta['gt_bbox']:
            matches, match_labels = self.bbox_assigner(anchors, gt_bbox)
            matches_list.append(matches)
            match_labels_list.append(match_labels)
        # reshape network outputs
        cls_logits = [_.transpose([0, 2, 3, 1]) for _ in cls_logits]
        cls_logits = [_.reshape([0, -1, self.cls_out_channels]) \
                      for _ in cls_logits]
        bboxes_reg = [_.transpose([0, 2, 3, 1]) for _ in bboxes_reg]
        bboxes_reg = [_.reshape([0, -1, 4]) for _ in bboxes_reg]
        cls_logits = paddle.concat(cls_logits, axis=1)
        bboxes_reg = paddle.concat(bboxes_reg, axis=1)

        cls_pred_list, cls_tar_list = [], []
        reg_pred_list, reg_tar_list = [], []
        # find and gather preds and targets in each image
        for matches, match_labels, cls_logit, bbox_reg, gt_bbox, gt_class in \
            zip(matches_list, match_labels_list, cls_logits, bboxes_reg,
                meta['gt_bbox'], meta['gt_class']):
            pos_mask = (match_labels == 1)
            neg_mask = (match_labels == 0)
            chosen_mask = paddle.logical_or(pos_mask, neg_mask)

            gt_class = gt_class.reshape([-1])
            bg_class = paddle.to_tensor(
                [self.num_classes], dtype=gt_class.dtype)
            # a trick to assign num_classes to negative targets
            gt_class = paddle.concat([gt_class, bg_class])
            matches = paddle.where(
                neg_mask, paddle.full_like(matches, gt_class.size-1), matches)

            cls_pred = cls_logit[chosen_mask]
            cls_tar  = gt_class[matches[chosen_mask]]
            reg_pred = bbox_reg[pos_mask].reshape([-1, 4])
            reg_tar = gt_bbox[matches[pos_mask]].reshape([-1, 4])
            if self.decode_reg_out:
                reg_pred = self.bbox_coder.decode(
                    anchors[pos_mask], reg_pred)
            else:
                reg_tar = self.bbox_coder.encode(anchors[pos_mask], reg_tar)
            cls_pred_list.append(cls_pred)
            cls_tar_list.append(cls_tar)
            reg_pred_list.append(reg_pred)
            reg_tar_list.append(reg_tar)
        cls_pred = paddle.concat(cls_pred_list)
        cls_tar  = paddle.concat(cls_tar_list)
        reg_pred = paddle.concat(reg_pred_list)
        reg_tar  = paddle.concat(reg_tar_list)
        avg_factor = max(1.0, reg_pred.shape[0])
        cls_loss = self.loss_class(
            cls_pred, cls_tar, reduction='sum')/avg_factor
        if reg_pred.size == 0:
            reg_loss = bboxes_reg[0][0].sum() * 0
        else:
            reg_loss = self.loss_bbox(
                reg_pred, reg_tar, reduction='sum')/avg_factor
        return dict(loss_cls=cls_loss, loss_reg=reg_loss)

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
        for anchor, cls_score, bbox_pred in zip(anchors, cls_scores, bbox_preds):
            cls_score = cls_score.reshape([-1, self.num_classes])
            bbox_pred = bbox_pred.reshape([-1, 4])
            if self.nms_pre is not None and cls_score.shape[0] > self.nms_pre:
                max_score = cls_score.max(axis=1)
                _, topk_inds = max_score.topk(self.nms_pre)
                bbox_pred = bbox_pred.gather(topk_inds)
                anchor    = anchor.gather(topk_inds)
                cls_score = cls_score.gather(topk_inds)
            bbox_pred = self.bbox_coder.decode(
                anchor, bbox_pred, max_shape=im_shape)
            bbox_pred = bbox_pred.squeeze()
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
                anchors,
                cls_score_list,
                bbox_pred_list,
                im_shape[img_id],
                scale_factor[img_id])
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = paddle.stack(batch_bboxes, axis=0)
        batch_scores = paddle.stack(batch_scores, axis=0)
        return batch_bboxes, batch_scores

    def post_process(self, head_outputs, im_shape, scale_factor):
        cls_scores, bbox_preds = head_outputs
        anchors = self.anchor_generator(cls_scores)
        cls_scores = [_.transpose([0, 2, 3, 1]) for _ in cls_scores]
        bbox_preds = [_.transpose([0, 2, 3, 1]) for _ in bbox_preds]
        bboxes, scores = self.decode(
            anchors, cls_scores, bbox_preds, im_shape, scale_factor)
        bbox_pred, bbox_num, _ = self.nms(bboxes, scores)
        return bbox_pred, bbox_num
