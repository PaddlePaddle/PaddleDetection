from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

INF = 1e8
__all__ = ['FCOSLoss']


def flatten_tensor(inputs, channel_first=False):
    """
    Flatten a Tensor
    Args:
        inputs  (Variables): Input Tensor
        channel_first(bool): if true the dimension order of
            Tensor is [N, C, H, W], otherwise is [N, H, W, C]
    Return:
        input_channel_last (Variables): The flattened Tensor in channel_last style
    """
    if channel_first:
        input_channel_last = paddle.transpose(
            inputs, perm=[0, 2, 3, 1])
    else:
        input_channel_last = inputs
    output_channel_last = paddle.flatten(input_channel_last, start_axis=0, stop_axis=2) # [N*H*W, C]
    return output_channel_last


def sigmoid_cross_entropy_with_logits_loss(inputs, label, ignore_index=-100, normalize=False):
    output = F.binary_cross_entropy_with_logits(inputs, label, reduction='none')
    mask_tensor = paddle.cast(label != ignore_index, 'float32')
    output = paddle.multiply(output, mask_tensor)
    if normalize:
        sum_valid_mask = paddle.sum(mask_tensor)
        output = output / sum_valid_mask
    return output


@register
class FCOSLoss(nn.Layer):
    """
    FCOSLoss
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type(str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights(float): weight for location loss
    """
    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="giou",
                 reg_weights=1.0):
        super(FCOSLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights

    def __iou_loss(self, pred, targets, positive_mask, weights=None):
        """
        Calculate the loss for location prediction
        Args:
            pred          (Variables): bounding boxes prediction
            targets       (Variables): targets for positive samples
            positive_mask (Variables): mask of positive samples
            weights       (Variables): weights for each positive samples
        Return:
            loss (Varialbes): location loss
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

    def forward(self, cls_logits, bboxes_reg, centerness, tag_labels, tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): list of Variables, which is predicted
                score for all anchor points with shape [N, M, C]
            bboxes_reg (list): list of Variables, which is predicted
                offsets for all anchor points with shape [N, M, 4]
            centerness (list): list of Variables, which is predicted
                centerness for all anchor points with shape [N, M, 1]
            tag_labels (list): list of Variables, which is category
                targets for each anchor point
            tag_bboxes (list): list of Variables, which is bounding
                boxes targets for positive samples
            tag_center (list): list of Variables, which is centerness
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

        cls_logits_flatten = paddle.concat(
            cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = paddle.concat(
            bboxes_reg_flatten_list, axis=0)
        centerness_flatten = paddle.concat(
            centerness_flatten_list, axis=0)

        tag_labels_flatten = paddle.concat(
            tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = paddle.concat(
            tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = paddle.concat(
            tag_center_flatten_list, axis=0)
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
        categories = cls_logits_flatten.shape[-1]
        tag_labels_bin = np.zeros(shape=(tag_labels_flatten.shape[0], categories))
        tag_labels_flatten_b = paddle.squeeze(tag_labels_flatten, axis=-1).numpy()
        inds = np.nonzero((tag_labels_flatten_b > 0) & (tag_labels_flatten_b <= categories))[0]
        if len(inds) > 0:
            tag_labels_bin[inds, tag_labels_flatten_b[inds]-1] = 1
        tag_labels_flatten_bin = paddle.to_tensor(tag_labels_bin.astype('float32'))
        # sigmoid_focal_loss
        cls_loss = F.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten_bin) / num_positive_fp32

        # 2. bboxes_reg: giou_loss
        mask_positive_float = paddle.squeeze(mask_positive_float, axis=-1)
        tag_center_flatten = paddle.squeeze(tag_center_flatten, axis=-1)
        reg_loss = self.__iou_loss(bboxes_reg_flatten, tag_bboxes_flatten,
            mask_positive_float, weights=tag_center_flatten)
        reg_loss = reg_loss * mask_positive_float / normalize_sum

        # 3. centerness: sigmoid_cross_entropy_with_logits_loss
        centerness_flatten = paddle.squeeze(centerness_flatten, axis=-1)
        ctn_loss = sigmoid_cross_entropy_with_logits_loss(centerness_flatten, tag_center_flatten)
        ctn_loss = ctn_loss * mask_positive_float / num_positive_fp32

        loss_all = {
            "loss_centerness": paddle.sum(ctn_loss),
            "loss_cls": paddle.sum(cls_loss),
            "loss_box": paddle.sum(reg_loss)
        }
        return loss_all
