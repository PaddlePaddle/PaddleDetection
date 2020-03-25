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

import numpy as np

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import CreateTensorFromNumpy

INF = 1e8
__all__ = ['FCOSLoss']


@register
@serializable
class FCOSLoss(object):
    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="IoU",
                 reg_weights=1.0):
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights

    def __flatten_tensor(self, input, channel_first=False):
        if channel_first:
            # input = input * 0 + 1.0
            input_channel_last = fluid.layers.transpose(input, perm=[0, 2, 3, 1])
        else:
            input_channel_last = input
        input_channel_last = fluid.layers.flatten(input_channel_last, axis=3)
     #   input_channel_last = fluid.layers.Print(input_channel_last, summarize=-1, message="wxx")
        print("XXXXXXXXXDEBUG HOLY SHIT BB ", input_channel_last.shape)
        return input_channel_last

    def __flatten_target(self, input):
        print("XXXXXXXXXDEBUG HOLY SHIT ", input.shape)
        input_channel_last = fluid.layers.flatten(input, axis=3)
        # input_channel_last = fluid.layers.reshape(input, shape=(-1, input.shape[-1]))
        print("XXXXXXXXXDEBUG HOLY SHIT AA ", input_channel_last.shape)
        return input_channel_last

    def __iou_loss(self, pred, targets, positive_mask, weights=None):
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
        area_target = (tlw + trw) * (tth + tbh)
        area_predict = (plw + prw) * (pth + pbh)
        ilw = fluid.layers.elementwise_min(plw, tlw)
        irw = fluid.layers.elementwise_min(prw, trw)
        ith = fluid.layers.elementwise_min(pth, tth)
        ibh = fluid.layers.elementwise_min(pbh, tbh)
        clw = fluid.layers.elementwise_max(plw, tlw)
        crw = fluid.layers.elementwise_max(prw, trw)
        cth = fluid.layers.elementwise_max(pth, tth)
        cbh = fluid.layers.elementwise_max(pbh, tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (area_predict + area_target - area_inter + 1.0)
        ious = ious * positive_mask
        # fluid.layers.Print(fluid.layers.reduce_max(ious))
        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - fluid.layers.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        return loss

    def __call__(self, cls_logits, bboxes_reg, centerness,
                 tag_labels, tag_bboxes, tag_center):
        cls_logits_flatten_list = []
        bboxes_reg_flatten_list = []
        centerness_flatten_list = []
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        # tag_scores_flatten = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        # fluid.layers.Print(cls_logits[-1])
        # fluid.layers.Print(fluid.layers.reduce_mean(cls_logits[-1]))
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(self.__flatten_tensor(cls_logits[num_lvl -1 - lvl], True))
            bboxes_reg_flatten_list.append(self.__flatten_tensor(bboxes_reg[num_lvl -1 - lvl], True))
            centerness_flatten_list.append(self.__flatten_tensor(centerness[num_lvl -1 - lvl], True))
            tag_labels_flatten_list.append(self.__flatten_tensor(tag_labels[lvl], False))
            tag_bboxes_flatten_list.append(self.__flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(self.__flatten_tensor(tag_center[lvl], False))

            # tag_labels_flatten.append(self.__flatten_target(tag_labels[lvl]))
            # tag_bboxes_flatten.append(self.__flatten_target(tag_bboxes[lvl]))
            # # tag_scores_flatten.append(self.__flatten_tensor(tag_scores[lvl]))
            # tag_center_flatten.append(self.__flatten_target(tag_center[lvl]))

            # tag_labels_flatten.append(tag_labels[lvl])
            # tag_bboxes_flatten.append(tag_bboxes[lvl])
            # tag_center_flatten.append(tag_center[lvl])

        cls_logits_flatten = fluid.layers.concat(cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = fluid.layers.concat(bboxes_reg_flatten_list, axis=0)
        centerness_flatten = fluid.layers.concat(centerness_flatten_list, axis=0)
        tag_labels_flatten = fluid.layers.concat(tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = fluid.layers.concat(tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = fluid.layers.concat(tag_center_flatten_list, axis=0)
        print("XXXXXXXXXXXXXXXGWDEBUG ", cls_logits_flatten)
        print("XXXXXXXXXXXXXXXGWDEBUG ", bboxes_reg_flatten)
        print("XXXXXXXXXXXXXXXGWDEBUG ", centerness_flatten)
        print("XXXXXXXXXXXXXXXGWDEBUG ", tag_bboxes_flatten)
        print("XXXXXXXXXXXXXXXGWDEBUG ", tag_labels_flatten)
        # tag_scores_flatten = fluid.layers.concat(tag_scores_flatten, axis=0)
        #for xx in tag_center_flatten:
        tag_labels_flatten.stop_gradient = True
        tag_bboxes_flatten.stop_gradient = True
        tag_center_flatten.stop_gradient = True
        print("XXXXXXXXXXXXXXXDEBUG ", tag_labels_flatten.shape)
        print("XXXXXXXXXXXXXXXDEBUG tag_bboxes_flatten ", tag_bboxes_flatten.shape)
        print("XXXXXXXXXXXXXXXDEBUG bboxes_reg_flatten ", bboxes_reg_flatten.shape)
        print("XXXXXXXXXXXXXXXDEBUG tag_center_flatten ", tag_center_flatten.shape)

        mask_positive = tag_labels_flatten > 0
        # fluid.layers.Print(mask_positive)
        n0_index = fluid.layers.where(mask_positive)
        # fluid.layers.Print(n0_index)
        mask_positive.stop_gradient = True
        mask_positive_float = fluid.layers.cast(mask_positive, dtype="float32")
        # fluid.layers.Print(mask_positive_float)
        mask_positive_float.stop_gradient = True
        num_positive_fp32 = fluid.layers.reduce_sum(mask_positive_float)
        num_positive_int32 = fluid.layers.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_fp32.stop_gradient = True
        num_positive_int32.stop_gradient = True
        normalize_sum = fluid.layers.sum(tag_center_flatten)
        normalize_sum.stop_gradient = True
        normalize_sum = fluid.layers.reduce_sum(mask_positive_float * normalize_sum)
        normalize_sum.stop_gradient = True
        # fluid.layers.Print(fluid.layers.shape(cls_logits_flatten))
        # fluid.layers.Print(fluid.layers.shape(tag_labels_flatten))
        # fluid.layers.Print(fluid.layers.shape(centerness_flatten))
        # fluid.layers.Print(fluid.layers.shape(tag_center_flatten))
        # fluid.layers.Print(fluid.layers.shape(bboxes_reg_flatten))
        # fluid.layers.Print(fluid.layers.shape(tag_bboxes_flatten))
        # fluid.layers.Print(cls_logits_flatten)
        # fluid.layers.Print(tag_labels_flatten)
        # fluid.layers.Print(centerness_flatten)
        # fluid.layers.Print(tag_center_flatten)
        # fluid.layers.Print(bboxes_reg_flatten)
        # fluid.layers.Print(tag_bboxes_flatten)
        cls_loss = fluid.layers.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten, num_positive_int32) / num_positive_fp32
        # cls_loss = fluid.layers.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten, num_positive_int32)
        # fluid.layers.Print(num_positive_fp32)
        # fluid.layers.Print(cls_loss)
        # fluid.layers.Print(fluid.layers.reduce_sum(cls_loss))
        # fluid.layers.Print(fluid.layers.reduce_max(cls_loss))
        # fluid.layers.Print(fluid.layers.reduce_mean(cls_loss))
        # fluid.layers.Print(cls_loss)
        # fluid.layers.Print(centerness_flatten)
        # fluid.layers.Print(tag_center_flatten)
        # fluid.layers.Print(normalize_sum)
        # reg_loss = self.__iou_loss(bboxes_reg_flatten, tag_bboxes_flatten, tag_center_flatten)
        # mask_positive_reg = fluid.layers.expand_as(mask_positive_float, bboxes_reg_flatten)
        # mask_positive_reg.stop_gradient = True
        # reg_loss = self.__iou_loss(bboxes_reg_flatten * mask_positive_reg, tag_bboxes_flatten * mask_positive_reg, tag_center_flatten * mask_positive_reg) * mask_positive_float / normalize_sum
        reg_loss = self.__iou_loss(bboxes_reg_flatten, tag_bboxes_flatten, mask_positive_float, tag_center_flatten) * mask_positive_float / normalize_sum
        # reg_loss = self.__iou_loss(bboxes_reg_flatten, tag_bboxes_flatten, tag_center_flatten) * mask_positive_float / normalize_sum
        # fluid.layers.Print(fluid.layers.gather_nd(bboxes_reg_flatten * mask_positive_float, index_n0))
        # centerness_flatten_sigmoid = fluid.layers.sigmoid(centerness_flatten)
        # index_n0 = fluid.layers.where(tag_center_flatten != 0)
        # tag_reg_value_0 = fluid.layers.gather_nd(tag_bboxes_flatten * mask_positive_float, index_n0)
        # ctn_value_0 = fluid.layers.gather_nd(tag_center_flatten, index_n0)
        # fluid.layers.Print(tag_reg_value_0)
        # fluid.layers.Print(tag_value_0)
        # fluid.layers.Print(ctn_value_0)
        # fluid.layers.Print(fluid.layers.shape(index_n0))
        # fluid.layers.Print(fluid.layers.gather_nd(bboxes_reg_flatten * mask_positive_float, index_n0))
        # fluid.layers.Print(fluid.layers.gather_nd(tag_bboxes_flatten * mask_positive_float, index_n0), summarize=-1)
        # fluid.layers.Print(fluid.layers.gather_nd(tag_center_flatten * mask_positive_float, index_n0), summarize=-1)
        # fluid.layers.Print(fluid.layers.gather_nd(tag_labels_flatten * mask_positive_float, index_n0), summarize=-1)
        # fluid.layers.Print(fluid.layers.gather_nd(reg_loss * mask_positive_float, index_n0))
        ctn_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=centerness_flatten,
#            input=centerness_flatten,
            label=tag_center_flatten) * mask_positive_float / num_positive_fp32
#            soft_label=True) * mask_positive_float / num_positive_int32
#            soft_label=True) * mask_positive_float / num_positive_int32
        # fluid.layers.Print(fluid.layers.reduce_mean(ctn_loss))
        loss_all = {
            "loss_centerness": fluid.layers.reduce_sum(ctn_loss),
            "loss_cls": fluid.layers.reduce_sum(cls_loss),
            "loss_box": fluid.layers.reduce_sum(reg_loss)
        }
        return loss_all


