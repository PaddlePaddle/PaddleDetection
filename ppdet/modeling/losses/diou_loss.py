# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NumpyArrayInitializer

from paddle import fluid
from ppdet.core.workspace import register, serializable
from .giou_loss import GiouLoss

__all__ = ['DiouLoss']


@register
@serializable
class DiouLoss(GiouLoss):

    def __init__(self,loss_weight=10., is_cls_agnostic=False, num_classes=81, max_height=608, max_width=608,use_xiou_loss_yolo=False,use_complete_iou_loss=True):
        super(DiouLoss, self).__init__(
            loss_weight=loss_weight,
            is_cls_agnostic=is_cls_agnostic,
            num_classes=num_classes)
        self._MAX_HI = max_height
        self._MAX_WI = max_width
        self._use_xiou_loss_yolo = use_xiou_loss_yolo
        self.use_complete_iou_loss = use_complete_iou_loss

    def __call__(self, *args, **kwargs):
        if self._use_xiou_loss_yolo:
            _x, _y, _w, _h, _tx, _ty, _tw, _th, _anchors, _downsample_ratio, _batch_size = args[0], args[1], args[
                2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]
            return self._diou_loss_yolo(_x, _y, _w, _h, _tx, _ty, _tw, _th, _anchors, _downsample_ratio,
                                        _batch_size, self.loss_weight, self._MAX_HI, self._MAX_WI)
        else:
            x, y, inside_weight, outside_weight = kwargs['x'], kwargs['y'], kwargs['inside_weight'], kwargs[
                'outside_weight']
            return self.diou_loss(x, y, inside_weight, outside_weight, self.loss_weight, self.is_cls_agnostic,
                                  self.num_classes, bbox_reg_weight=[0.1, 0.1, 0.2, 0.2])

    def diou_loss(self, x, y, inside_weight, outside_weight, loss_weight, is_cls_agnostic, num_classes,
                      bbox_reg_weight):

        eps = 1.e-10
        x1, y1, x2, y2 = self.bbox_transform(x, bbox_reg_weight)
        x1g, y1g, x2g, y2g = self.bbox_transform(y, bbox_reg_weight)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g
        hg = y2g - y1g

        x2 = fluid.layers.elementwise_max(x1, x2)
        y2 = fluid.layers.elementwise_max(y1, y2)

        # A and B
        xkis1 = fluid.layers.elementwise_max(x1, x1g)
        ykis1 = fluid.layers.elementwise_max(y1, y1g)
        xkis2 = fluid.layers.elementwise_min(x2, x2g)
        ykis2 = fluid.layers.elementwise_min(y2, y2g)

        # A or B
        xc1 = fluid.layers.elementwise_min(x1, x1g)
        yc1 = fluid.layers.elementwise_min(y1, y1g)
        xc2 = fluid.layers.elementwise_max(x2, x2g)
        yc2 = fluid.layers.elementwise_max(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * fluid.layers.greater_than(
            xkis2, xkis1) * fluid.layers.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + eps
        iouk = intsctk / unionk

        # DIOU term
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        diou_term = (dist_intersection + eps) / (dist_union + eps)

        # CIOU term
        ciou_term = 0
        if self.use_complete_iou_loss:
            ar_gt = wg / hg
            ar_pred = w / h
            arctan = fluid.layers.atan(ar_gt) - fluid.layers.atan(ar_pred)
            ar_loss = 4. / np.pi / np.pi * arctan * arctan
            alpha = ar_loss / (1 - iouk + ar_loss + eps)
            alpha.stop_gradient = True
            ciou_term = alpha * ar_loss

        iou_weights = 1
        if inside_weight is not None and outside_weight is not None:
            inside_weight = fluid.layers.reshape(inside_weight, shape=(-1, 4))
            outside_weight = fluid.layers.reshape(outside_weight, shape=(-1, 4))

            inside_weight = fluid.layers.reduce_mean(inside_weight, dim=1)
            outside_weight = fluid.layers.reduce_mean(outside_weight, dim=1)

            iou_weights = inside_weight * outside_weight

        class_weight = 2 if self.is_cls_agnostic else self.num_classes
        diou = fluid.layers.reduce_mean(
            (1 - iouk + ciou_term + diou_term) * iou_weights) * class_weight

        return diou * self.loss_weight

    def _diou_loss_yolo(self, x, y, w, h, tx, ty, tw, th, anchors, downsample_ratio, batch_size, loss_weight,MAX_HI, MAX_WI):
        eps = 1.e-10
        x1, y1, x2, y2 = self._bbox_transform(
            x, y, w, h, anchors, downsample_ratio, batch_size, False)
        x1g, y1g, x2g, y2g = self._bbox_transform(
            tx, ty, tw, th, anchors, downsample_ratio, batch_size, True)

        #central coordinates
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g
        hg = y2g - y1g

        x2 = fluid.layers.elementwise_max(x1, x2)
        y2 = fluid.layers.elementwise_max(y1, y2)
        # A and B
        xkis1 = fluid.layers.elementwise_max(x1, x1g)
        ykis1 = fluid.layers.elementwise_max(y1, y1g)
        xkis2 = fluid.layers.elementwise_min(x2, x2g)
        ykis2 = fluid.layers.elementwise_min(y2, y2g)
        # A or B
        xc1 = fluid.layers.elementwise_min(x1, x1g)
        yc1 = fluid.layers.elementwise_min(y1, y1g)
        xc2 = fluid.layers.elementwise_max(x2, x2g)
        yc2 = fluid.layers.elementwise_max(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * fluid.layers.greater_than(
            xkis2, xkis1) * fluid.layers.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + eps
        iouk = intsctk / unionk

        # diou_loss
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        diou_term = (dist_intersection + eps) / (dist_union + eps)

        loss_diou = 1. - iouk + diou_term
        loss_diou = loss_diou * self.loss_weight

        return loss_diou
