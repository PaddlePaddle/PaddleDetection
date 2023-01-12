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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling import ops

__all__ = ['FCOSLoss']


def flatten_tensor(inputs, channel_first=False):
    """
    Flatten a Tensor
    Args:
        inputs (Tensor): 4-D Tensor with shape [N, C, H, W] or [N, H, W, C]
        channel_first (bool): If true the dimension order of Tensor is 
            [N, C, H, W], otherwise is [N, H, W, C]
    Return:
        output_channel_last (Tensor): The flattened Tensor in channel_last style
    """
    if channel_first:
        input_channel_last = paddle.transpose(inputs, perm=[0, 2, 3, 1])
    else:
        input_channel_last = inputs
    output_channel_last = paddle.flatten(
        input_channel_last, start_axis=0, stop_axis=2)
    return output_channel_last


@register
class FCOSLoss(nn.Layer):
    """
    FCOSLoss
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type (str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights (float): weight for location loss
        quality (str): quality branch, centerness/iou
    """

    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="giou",
                 reg_weights=1.0,
                 quality='centerness'):
        super(FCOSLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights
        self.quality = quality

    def __iou_loss(self,
                   pred,
                   targets,
                   positive_mask,
                   weights=None,
                   return_iou=False):
        """
        Calculate the loss for location prediction
        Args:
            pred (Tensor): bounding boxes prediction
            targets (Tensor): targets for positive samples
            positive_mask (Tensor): mask of positive samples
            weights (Tensor): weights for each positive samples
        Return:
            loss (Tensor): location loss
        """
        plw = pred[:, 0] * positive_mask
        pth = pred[:, 1] * positive_mask
        prw = pred[:, 2] * positive_mask
        pbh = pred[:, 3] * positive_mask

        tlw = targets[:, 0] * positive_mask
        tth = targets[:, 1] * positive_mask
        trw = targets[:, 2] * positive_mask
        tbh = targets[:, 3] * positive_mask
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        clw = paddle.maximum(plw, tlw)
        crw = paddle.maximum(prw, trw)
        cth = paddle.maximum(pth, tth)
        cbh = paddle.maximum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
            area_predict + area_target - area_inter + 1.0)
        ious = ious * positive_mask

        if return_iou:
            return ious

        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - paddle.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        return loss

    def forward(self, cls_logits, bboxes_reg, centerness, tag_labels,
                tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): list of Tensor, which is predicted
                score for all anchor points with shape [N, M, C]
            bboxes_reg (list): list of Tensor, which is predicted
                offsets for all anchor points with shape [N, M, 4]
            centerness (list): list of Tensor, which is predicted
                centerness for all anchor points with shape [N, M, 1]
            tag_labels (list): list of Tensor, which is category
                targets for each anchor point
            tag_bboxes (list): list of Tensor, which is bounding
                boxes targets for positive samples
            tag_center (list): list of Tensor, which is centerness
                targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        cls_logits_flatten_list = []
        bboxes_reg_flatten_list = []
        centerness_flatten_list = []
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(
                flatten_tensor(cls_logits[lvl], True))
            bboxes_reg_flatten_list.append(
                flatten_tensor(bboxes_reg[lvl], True))
            centerness_flatten_list.append(
                flatten_tensor(centerness[lvl], True))

            tag_labels_flatten_list.append(
                flatten_tensor(tag_labels[lvl], False))
            tag_bboxes_flatten_list.append(
                flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(
                flatten_tensor(tag_center[lvl], False))

        cls_logits_flatten = paddle.concat(cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = paddle.concat(bboxes_reg_flatten_list, axis=0)
        centerness_flatten = paddle.concat(centerness_flatten_list, axis=0)

        tag_labels_flatten = paddle.concat(tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = paddle.concat(tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = paddle.concat(tag_center_flatten_list, axis=0)
        tag_labels_flatten.stop_gradient = True
        tag_bboxes_flatten.stop_gradient = True
        tag_center_flatten.stop_gradient = True

        mask_positive_bool = tag_labels_flatten > 0
        mask_positive_bool.stop_gradient = True
        mask_positive_float = paddle.cast(mask_positive_bool, dtype="float32")
        mask_positive_float.stop_gradient = True

        num_positive_fp32 = paddle.sum(mask_positive_float)
        num_positive_fp32.stop_gradient = True
        num_positive_int32 = paddle.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_int32.stop_gradient = True

        normalize_sum = paddle.sum(tag_center_flatten * mask_positive_float)
        normalize_sum.stop_gradient = True

        # 1. cls_logits: sigmoid_focal_loss
        # expand onehot labels
        num_classes = cls_logits_flatten.shape[-1]
        tag_labels_flatten = paddle.squeeze(tag_labels_flatten, axis=-1)
        tag_labels_flatten_bin = F.one_hot(
            tag_labels_flatten, num_classes=1 + num_classes)
        tag_labels_flatten_bin = tag_labels_flatten_bin[:, 1:]
        # sigmoid_focal_loss
        cls_loss = F.sigmoid_focal_loss(
            cls_logits_flatten, tag_labels_flatten_bin) / num_positive_fp32

        if self.quality == 'centerness':
            # 2. bboxes_reg: giou_loss
            mask_positive_float = paddle.squeeze(mask_positive_float, axis=-1)
            tag_center_flatten = paddle.squeeze(tag_center_flatten, axis=-1)
            reg_loss = self.__iou_loss(
                bboxes_reg_flatten,
                tag_bboxes_flatten,
                mask_positive_float,
                weights=tag_center_flatten)
            reg_loss = reg_loss * mask_positive_float / normalize_sum

            # 3. centerness: sigmoid_cross_entropy_with_logits_loss
            centerness_flatten = paddle.squeeze(centerness_flatten, axis=-1)
            quality_loss = ops.sigmoid_cross_entropy_with_logits(
                centerness_flatten, tag_center_flatten)
            quality_loss = quality_loss * mask_positive_float / num_positive_fp32

        elif self.quality == 'iou':
            # 2. bboxes_reg: giou_loss
            mask_positive_float = paddle.squeeze(mask_positive_float, axis=-1)
            tag_center_flatten = paddle.squeeze(tag_center_flatten, axis=-1)
            reg_loss = self.__iou_loss(
                bboxes_reg_flatten,
                tag_bboxes_flatten,
                mask_positive_float,
                weights=None)
            reg_loss = reg_loss * mask_positive_float / num_positive_fp32
            # num_positive_fp32 is num_foreground

            # 3. centerness: sigmoid_cross_entropy_with_logits_loss
            centerness_flatten = paddle.squeeze(centerness_flatten, axis=-1)
            gt_ious = self.__iou_loss(
                bboxes_reg_flatten,
                tag_bboxes_flatten,
                mask_positive_float,
                weights=None,
                return_iou=True)
            quality_loss = ops.sigmoid_cross_entropy_with_logits(
                centerness_flatten, gt_ious)
            quality_loss = quality_loss * mask_positive_float / num_positive_fp32
        else:
            raise Exception(f'Unknown quality type: {self.quality}')

        loss_all = {
            "loss_cls": paddle.sum(cls_loss),
            "loss_box": paddle.sum(reg_loss),
            "loss_quality": paddle.sum(quality_loss),
        }
        return loss_all
