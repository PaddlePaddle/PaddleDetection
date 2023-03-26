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
from functools import partial

__all__ = ['FCOSLoss', 'FCOSLossMILC', 'FCOSLossCR']


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

    def _iou_loss(self,
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
            reg_loss = self._iou_loss(
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
            reg_loss = self._iou_loss(
                bboxes_reg_flatten,
                tag_bboxes_flatten,
                mask_positive_float,
                weights=None)
            reg_loss = reg_loss * mask_positive_float / num_positive_fp32
            # num_positive_fp32 is num_foreground

            # 3. centerness: sigmoid_cross_entropy_with_logits_loss
            centerness_flatten = paddle.squeeze(centerness_flatten, axis=-1)
            gt_ious = self._iou_loss(
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


@register
class FCOSLossMILC(FCOSLoss):
    """
    FCOSLossMILC for ARSL in semi-det(ssod)
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type (str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights (float): weight for location loss
    """

    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="giou",
                 reg_weights=1.0):
        super(FCOSLossMILC, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights

    def iou_loss(self, pred, targets, weights=None, avg_factor=None):
        """
        Calculate the loss for location prediction
        Args:
            pred (Tensor): bounding boxes prediction
            targets (Tensor): targets for positive samples
            weights (Tensor): weights for each positive samples
        Return:
            loss (Tensor): location loss
        """
        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]
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
        ious = ious

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
        loss = paddle.sum(loss)
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss

    # temp function: calcualate iou between bbox and target
    def _bbox_overlap_align(self, pred, targets):
        assert pred.shape[0] == targets.shape[0], \
        'the pred should be aligned with target.'

        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
            area_predict + area_target - area_inter + 1.0)

        return ious

    def iou_based_soft_label_loss(self,
                                  pred,
                                  target,
                                  alpha=0.75,
                                  gamma=2.0,
                                  iou_weighted=False,
                                  implicit_iou=None,
                                  avg_factor=None):
        assert pred.shape == target.shape
        pred = F.sigmoid(pred)
        target = target.cast(pred.dtype)

        if implicit_iou is not None:
            pred = pred * implicit_iou

        if iou_weighted:
            focal_weight = (pred - target).abs().pow(gamma) * target * (target > 0.0).cast('float32') + \
                alpha * (pred - target).abs().pow(gamma) * \
                (target <= 0.0).cast('float32')
        else:
            focal_weight = (pred - target).abs().pow(gamma) * (target > 0.0).cast('float32') + \
                alpha * (pred - target).abs().pow(gamma) * \
                (target <= 0.0).cast('float32')

        # focal loss
        loss = F.binary_cross_entropy(
            pred, target, reduction='none') * focal_weight
        if avg_factor is not None:
            loss = loss / avg_factor
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

        # find positive index
        mask_positive_bool = tag_labels_flatten > 0
        mask_positive_bool.stop_gradient = True
        mask_positive_float = paddle.cast(mask_positive_bool, dtype="float32")
        mask_positive_float.stop_gradient = True

        num_positive_fp32 = paddle.sum(mask_positive_float)
        num_positive_fp32.stop_gradient = True
        num_positive_int32 = paddle.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_int32.stop_gradient = True

        # centerness target is used as reg weight
        normalize_sum = paddle.sum(tag_center_flatten * mask_positive_float)
        normalize_sum.stop_gradient = True

        # 1. IoU-Based soft label loss
        # calculate iou
        with paddle.no_grad():
            pos_ind = paddle.nonzero(
                tag_labels_flatten.reshape([-1]) > 0).reshape([-1])
            pos_pred = bboxes_reg_flatten[pos_ind]
            pos_target = tag_bboxes_flatten[pos_ind]
            bbox_iou = self._bbox_overlap_align(pos_pred, pos_target)
        # pos labels
        pos_labels = tag_labels_flatten[pos_ind].squeeze(1)
        cls_target = paddle.zeros(cls_logits_flatten.shape)
        cls_target[pos_ind, pos_labels - 1] = bbox_iou
        cls_loss = self.iou_based_soft_label_loss(
            cls_logits_flatten,
            cls_target,
            implicit_iou=F.sigmoid(centerness_flatten),
            avg_factor=num_positive_fp32)

        # 2. bboxes_reg: giou_loss
        mask_positive_float = paddle.squeeze(mask_positive_float, axis=-1)
        tag_center_flatten = paddle.squeeze(tag_center_flatten, axis=-1)
        reg_loss = self._iou_loss(
            bboxes_reg_flatten,
            tag_bboxes_flatten,
            mask_positive_float,
            weights=tag_center_flatten)
        reg_loss = reg_loss * mask_positive_float / normalize_sum

        # 3. iou loss
        pos_iou_pred = paddle.squeeze(centerness_flatten, axis=-1)[pos_ind]
        loss_iou = ops.sigmoid_cross_entropy_with_logits(pos_iou_pred, bbox_iou)
        loss_iou = loss_iou / num_positive_fp32 * 0.5

        loss_all = {
            "loss_cls": paddle.sum(cls_loss),
            "loss_box": paddle.sum(reg_loss),
            'loss_iou': paddle.sum(loss_iou),
        }

        return loss_all


# Concat multi-level feature maps by image
def levels_to_images(mlvl_tensor):
    batch_size = mlvl_tensor[0].shape[0]
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].shape[1]
    for t in mlvl_tensor:
        t = t.transpose([0, 2, 3, 1])
        t = t.reshape([batch_size, -1, channels])
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [paddle.concat(item, axis=0) for item in batch_list]


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@register
class FCOSLossCR(FCOSLossMILC):
    """
    FCOSLoss of Consistency Regularization
    """

    def __init__(self,
                 iou_loss_type="giou",
                 cls_weight=2.0,
                 reg_weight=2.0,
                 iou_weight=0.5,
                 hard_neg_mining_flag=True):
        super(FCOSLossCR, self).__init__()
        self.iou_loss_type = iou_loss_type
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.iou_weight = iou_weight
        self.hard_neg_mining_flag = hard_neg_mining_flag

    def iou_loss(self, pred, targets, weights=None, avg_factor=None):
        """
            Calculate the loss for location prediction
            Args:
                pred (Tensor): bounding boxes prediction
                targets (Tensor): targets for positive samples
                weights (Tensor): weights for each positive samples
            Return:
                loss (Tensor): location loss
            """
        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]
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
        ious = ious

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
        loss = paddle.sum(loss)
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss

    # calcualate iou between bbox and target
    def bbox_overlap_align(self, pred, targets):
        assert pred.shape[0] == targets.shape[0], \
        'the pred should be aligned with target.'

        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
            area_predict + area_target - area_inter + 1.0)
        return ious

    # cls loss: iou-based soft lable with joint iou
    def quality_focal_loss(self,
                           stu_cls,
                           targets,
                           quality=None,
                           weights=None,
                           alpha=0.75,
                           gamma=2.0,
                           avg_factor='sum'):
        stu_cls = F.sigmoid(stu_cls)
        if quality is not None:
            stu_cls = stu_cls * F.sigmoid(quality)

        focal_weight = (stu_cls - targets).abs().pow(gamma) * (targets > 0.0).cast('float32') + \
            alpha * (stu_cls - targets).abs().pow(gamma) * \
            (targets <= 0.0).cast('float32')

        loss = F.binary_cross_entropy(
            stu_cls, targets, reduction='none') * focal_weight

        if weights is not None:
            loss = loss * weights.reshape([-1, 1])
        loss = paddle.sum(loss)
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss

    # generate points according to feature maps
    def compute_locations_by_level(self, fpn_stride, h, w):
        """
        Compute locations of anchor points of each FPN layer
        Return:
            Anchor points locations of current FPN feature map
        """
        shift_x = paddle.arange(0, w * fpn_stride, fpn_stride)
        shift_y = paddle.arange(0, h * fpn_stride, fpn_stride)
        shift_x = paddle.unsqueeze(shift_x, axis=0)
        shift_y = paddle.unsqueeze(shift_y, axis=1)
        shift_x = paddle.expand(shift_x, shape=[h, w])
        shift_y = paddle.expand(shift_y, shape=[h, w])
        shift_x = paddle.reshape(shift_x, shape=[-1])
        shift_y = paddle.reshape(shift_y, shape=[-1])
        location = paddle.stack(
            [shift_x, shift_y], axis=-1) + float(fpn_stride) / 2
        return location

    # decode bbox from ltrb to x1y1x2y2
    def decode_bbox(self, ltrb, points):
        assert ltrb.shape[0] == points.shape[0], \
        "When decoding bbox in one image, the num of loc should be same with points."
        bbox_decoding = paddle.stack(
            [
                points[:, 0] - ltrb[:, 0], points[:, 1] - ltrb[:, 1],
                points[:, 0] + ltrb[:, 2], points[:, 1] + ltrb[:, 3]
            ],
            axis=1)
        return bbox_decoding

    # encode bbox from x1y1x2y2 to ltrb
    def encode_bbox(self, bbox, points):
        assert bbox.shape[0] == points.shape[0], \
        "When encoding bbox in one image, the num of bbox should be same with points."
        bbox_encoding = paddle.stack(
            [
                points[:, 0] - bbox[:, 0], points[:, 1] - bbox[:, 1],
                bbox[:, 2] - points[:, 0], bbox[:, 3] - points[:, 1]
            ],
            axis=1)
        return bbox_encoding

    def calcualate_iou(self, gt_bbox, predict_bbox):
        # bbox area
        gt_area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * \
             (gt_bbox[:, 3] - gt_bbox[:, 1])
        predict_area = (predict_bbox[:, 2] - predict_bbox[:, 0]) * \
             (predict_bbox[:, 3] - predict_bbox[:, 1])
        # overlop area
        lt = paddle.fmax(gt_bbox[:, None, :2], predict_bbox[None, :, :2])
        rb = paddle.fmin(gt_bbox[:, None, 2:], predict_bbox[None, :, 2:])
        wh = paddle.clip(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        # iou
        iou = overlap / (gt_area[:, None] + predict_area[None, :] - overlap)
        return iou

    # select potential positives from hard negatives 
    def hard_neg_mining(self,
                        cls_score,
                        loc_ltrb,
                        quality,
                        pos_ind,
                        hard_neg_ind,
                        loc_mask,
                        loc_targets,
                        iou_thresh=0.6):
        # get points locations and strides
        points_list = []
        strides_list = []
        scale_list = []
        scale = [0, 1, 2, 3, 4]
        for fpn_scale, fpn_stride, HW in zip(scale, self.fpn_stride,
                                             self.lvl_hw):
            h, w = HW
            lvl_points = self.compute_locations_by_level(fpn_stride, h, w)
            points_list.append(lvl_points)
            lvl_strides = paddle.full([h * w, 1], fpn_stride)
            strides_list.append(lvl_strides)
            lvl_scales = paddle.full([h * w, 1], fpn_scale)
            scale_list.append(lvl_scales)
        points = paddle.concat(points_list, axis=0)
        strides = paddle.concat(strides_list, axis=0)
        scales = paddle.concat(scale_list, axis=0)

        # cls scores
        cls_vals = F.sigmoid(cls_score) * F.sigmoid(quality)
        max_vals = paddle.max(cls_vals, axis=-1)
        class_ind = paddle.argmax(cls_vals, axis=-1)

        ### calculate iou between positive and hard negative
        # decode pos bbox
        pos_cls = max_vals[pos_ind]
        pos_loc = loc_ltrb[pos_ind].reshape([-1, 4])
        pos_strides = strides[pos_ind]
        pos_points = points[pos_ind].reshape([-1, 2])
        pos_loc = pos_loc * pos_strides
        pos_bbox = self.decode_bbox(pos_loc, pos_points)
        pos_scales = scales[pos_ind]
        # decode hard negative bbox
        hard_neg_loc = loc_ltrb[hard_neg_ind].reshape([-1, 4])
        hard_neg_strides = strides[hard_neg_ind]
        hard_neg_points = points[hard_neg_ind].reshape([-1, 2])
        hard_neg_loc = hard_neg_loc * hard_neg_strides
        hard_neg_bbox = self.decode_bbox(hard_neg_loc, hard_neg_points)
        hard_neg_scales = scales[hard_neg_ind]
        # iou between pos bbox and hard negative bbox
        hard_neg_pos_iou = self.calcualate_iou(hard_neg_bbox, pos_bbox)

        ### select potential positives from hard negatives
        # scale flag
        scale_temp = paddle.abs(
            pos_scales.reshape([-1])[None, :] - hard_neg_scales.reshape([-1])
            [:, None])
        scale_flag = (scale_temp <= 1.)
        # iou flag
        iou_flag = (hard_neg_pos_iou >= iou_thresh)
        # same class flag
        pos_class = class_ind[pos_ind]
        hard_neg_class = class_ind[hard_neg_ind]
        class_flag = pos_class[None, :] - hard_neg_class[:, None]
        class_flag = (class_flag == 0)
        # hard negative point inside positive bbox flag
        ltrb_temp = paddle.stack(
            [
                hard_neg_points[:, None, 0] - pos_bbox[None, :, 0],
                hard_neg_points[:, None, 1] - pos_bbox[None, :, 1],
                pos_bbox[None, :, 2] - hard_neg_points[:, None, 0],
                pos_bbox[None, :, 3] - hard_neg_points[:, None, 1]
            ],
            axis=-1)
        inside_flag = ltrb_temp.min(axis=-1) > 0
        # reset iou
        valid_flag = (iou_flag & class_flag & inside_flag & scale_flag)
        invalid_iou = paddle.zeros_like(hard_neg_pos_iou)
        hard_neg_pos_iou = paddle.where(valid_flag, hard_neg_pos_iou,
                                        invalid_iou)
        pos_hard_neg_max_iou = hard_neg_pos_iou.max(axis=-1)
        # selece potential pos
        potential_pos_ind = (pos_hard_neg_max_iou > 0.)
        num_potential_pos = paddle.nonzero(potential_pos_ind).shape[0]
        if num_potential_pos == 0:
            return None

        ### calculate loc target：aggregate all matching bboxes as the bbox targets of potential pos
        # prepare data
        potential_points = hard_neg_points[potential_pos_ind].reshape([-1, 2])
        potential_strides = hard_neg_strides[potential_pos_ind]
        potential_valid_flag = valid_flag[potential_pos_ind]
        potential_pos_ind = hard_neg_ind[potential_pos_ind]

        # get cls and box of matching positives
        pos_cls = max_vals[pos_ind]
        expand_pos_bbox = paddle.expand(
            pos_bbox,
            shape=[num_potential_pos, pos_bbox.shape[0], pos_bbox.shape[1]])
        expand_pos_cls = paddle.expand(
            pos_cls, shape=[num_potential_pos, pos_cls.shape[0]])
        invalid_cls = paddle.zeros_like(expand_pos_cls)
        expand_pos_cls = paddle.where(potential_valid_flag, expand_pos_cls,
                                      invalid_cls)
        expand_pos_cls = paddle.unsqueeze(expand_pos_cls, axis=-1)
        # aggregate box based on cls_score
        agg_bbox = (expand_pos_bbox * expand_pos_cls).sum(axis=1) \
            / expand_pos_cls.sum(axis=1)
        agg_ltrb = self.encode_bbox(agg_bbox, potential_points)
        agg_ltrb = agg_ltrb / potential_strides

        # loc target for all pos
        loc_targets[potential_pos_ind] = agg_ltrb
        loc_mask[potential_pos_ind] = 1.

        return loc_mask, loc_targets

    # get training targets
    def get_targets_per_img(self, tea_cls, tea_loc, tea_iou, stu_cls, stu_loc,
                            stu_iou):

        ### sample selection
        # prepare datas
        tea_cls_scores = F.sigmoid(tea_cls) * F.sigmoid(tea_iou)
        class_ind = paddle.argmax(tea_cls_scores, axis=-1)
        max_vals = paddle.max(tea_cls_scores, axis=-1)
        cls_mask = paddle.zeros_like(
            max_vals
        )  # set cls valid mask: pos is 1, hard_negative and negative are 0.
        num_pos, num_hard_neg = 0, 0

        # mean-std selection
        # using nonzero to turn index from bool to int, because the index will be used to compose two-dim index in following.
        # using squeeze rather than reshape to avoid errors when no score is larger than thresh.
        candidate_ind = paddle.nonzero(max_vals >= 0.1).squeeze(axis=-1)
        num_candidate = candidate_ind.shape[0]
        if num_candidate > 0:
            # pos thresh = mean + std to select pos samples
            candidate_score = max_vals[candidate_ind]
            candidate_score_mean = candidate_score.mean()
            candidate_score_std = candidate_score.std()
            pos_thresh = (candidate_score_mean + candidate_score_std).clip(
                max=0.4)
            # select pos
            pos_ind = paddle.nonzero(max_vals >= pos_thresh).squeeze(axis=-1)
            num_pos = pos_ind.shape[0]
            # select hard negatives as potential pos
            hard_neg_ind = (max_vals >= 0.1) & (max_vals < pos_thresh)
            hard_neg_ind = paddle.nonzero(hard_neg_ind).squeeze(axis=-1)
            num_hard_neg = hard_neg_ind.shape[0]
        # if not positive, directly select top-10 as pos.
        if (num_pos == 0):
            num_pos = 10
            _, pos_ind = paddle.topk(max_vals, k=num_pos)
        cls_mask[pos_ind] = 1.

        ### Consistency Regularization Training targets
        # cls targets
        pos_class_ind = class_ind[pos_ind]
        cls_targets = paddle.zeros_like(tea_cls)
        cls_targets[pos_ind, pos_class_ind] = tea_cls_scores[pos_ind,
                                                             pos_class_ind]
        # hard negative cls target
        if num_hard_neg != 0:
            cls_targets[hard_neg_ind] = tea_cls_scores[hard_neg_ind]
        # loc targets
        loc_targets = paddle.zeros_like(tea_loc)
        loc_targets[pos_ind] = tea_loc[pos_ind]
        # iou targets
        iou_targets = paddle.zeros(
            shape=[tea_iou.shape[0]], dtype=tea_iou.dtype)
        iou_targets[pos_ind] = F.sigmoid(
            paddle.squeeze(
                tea_iou, axis=-1)[pos_ind])

        loc_mask = cls_mask.clone()
        # select potential positive from hard negatives for loc_task training
        if (num_hard_neg > 0) and self.hard_neg_mining_flag:
            results = self.hard_neg_mining(tea_cls, tea_loc, tea_iou, pos_ind,
                                           hard_neg_ind, loc_mask, loc_targets)
            if results is not None:
                loc_mask, loc_targets = results
                loc_pos_ind = paddle.nonzero(loc_mask > 0.).squeeze(axis=-1)
                iou_targets[loc_pos_ind] = F.sigmoid(
                    paddle.squeeze(
                        tea_iou, axis=-1)[loc_pos_ind])

        return cls_mask, loc_mask, \
               cls_targets, loc_targets, iou_targets

    def forward(self, student_prediction, teacher_prediction):
        stu_cls_lvl, stu_loc_lvl, stu_iou_lvl = student_prediction
        tea_cls_lvl, tea_loc_lvl, tea_iou_lvl, self.fpn_stride = teacher_prediction

        # H and W of level (used for aggregating targets)
        self.lvl_hw = []
        for t in tea_cls_lvl:
            _, _, H, W = t.shape
            self.lvl_hw.append([H, W])

        # levels to images
        stu_cls_img = levels_to_images(stu_cls_lvl)
        stu_loc_img = levels_to_images(stu_loc_lvl)
        stu_iou_img = levels_to_images(stu_iou_lvl)
        tea_cls_img = levels_to_images(tea_cls_lvl)
        tea_loc_img = levels_to_images(tea_loc_lvl)
        tea_iou_img = levels_to_images(tea_iou_lvl)

        with paddle.no_grad():
            cls_mask, loc_mask, \
            cls_targets, loc_targets, iou_targets = multi_apply(
                self.get_targets_per_img,
                tea_cls_img,
                tea_loc_img,
                tea_iou_img,
                stu_cls_img,
                stu_loc_img,
                stu_iou_img
            )

        # flatten preditction
        stu_cls = paddle.concat(stu_cls_img, axis=0)
        stu_loc = paddle.concat(stu_loc_img, axis=0)
        stu_iou = paddle.concat(stu_iou_img, axis=0)
        # flatten targets
        cls_mask = paddle.concat(cls_mask, axis=0)
        loc_mask = paddle.concat(loc_mask, axis=0)
        cls_targets = paddle.concat(cls_targets, axis=0)
        loc_targets = paddle.concat(loc_targets, axis=0)
        iou_targets = paddle.concat(iou_targets, axis=0)

        ### Training Weights and avg factor
        # find positives
        cls_pos_ind = paddle.nonzero(cls_mask > 0.).squeeze(axis=-1)
        loc_pos_ind = paddle.nonzero(loc_mask > 0.).squeeze(axis=-1)
        # cls weight
        cls_sample_weights = paddle.ones([cls_targets.shape[0]])
        cls_avg_factor = paddle.max(cls_targets[cls_pos_ind],
                                    axis=-1).sum().item()
        # loc weight
        loc_sample_weights = paddle.max(cls_targets[loc_pos_ind], axis=-1)
        loc_avg_factor = loc_sample_weights.sum().item()
        # iou weight
        iou_sample_weights = paddle.ones([loc_pos_ind.shape[0]])
        iou_avg_factor = loc_pos_ind.shape[0]

        ### unsupervised loss
        # cls loss
        loss_cls = self.quality_focal_loss(
            stu_cls,
            cls_targets,
            quality=stu_iou,
            weights=cls_sample_weights,
            avg_factor=cls_avg_factor) * self.cls_weight
        # iou loss
        pos_stu_iou = paddle.squeeze(stu_iou, axis=-1)[loc_pos_ind]
        pos_iou_targets = iou_targets[loc_pos_ind]
        loss_iou = F.binary_cross_entropy(
            F.sigmoid(pos_stu_iou), pos_iou_targets,
            reduction='none') * iou_sample_weights
        loss_iou = loss_iou.sum() / iou_avg_factor * self.iou_weight
        # box loss
        pos_stu_loc = stu_loc[loc_pos_ind]
        pos_loc_targets = loc_targets[loc_pos_ind]

        loss_box = self.iou_loss(
            pos_stu_loc,
            pos_loc_targets,
            weights=loc_sample_weights,
            avg_factor=loc_avg_factor)
        loss_box = loss_box * self.reg_weight

        loss_all = {
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_iou": loss_iou,
        }
        return loss_all
