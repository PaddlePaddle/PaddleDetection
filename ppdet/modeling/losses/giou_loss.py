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

from paddle import fluid
from ppdet.core.workspace import register, serializable

__all__ = ['GiouLoss']


@register
@serializable
class GiouLoss(object):
    '''
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): diou loss weight, default as 10 in faster-rcnn
        is_cls_agnostic (bool): flag of class-agnostic
        num_classes (int): class num
    '''
    __shared__ = ['num_classes']

    def __init__(self, loss_weight=10., is_cls_agnostic=False, num_classes=81):
        super(GiouLoss, self).__init__()
        self.loss_weight = loss_weight
        self.is_cls_agnostic = is_cls_agnostic
        self.num_classes = num_classes

    # deltas: NxMx4
    def bbox_transform(self, deltas, weights):
        wx, wy, ww, wh = weights

        deltas = fluid.layers.reshape(deltas, shape=(0, -1, 4))

        dx = fluid.layers.slice(deltas, axes=[2], starts=[0], ends=[1]) * wx
        dy = fluid.layers.slice(deltas, axes=[2], starts=[1], ends=[2]) * wy
        dw = fluid.layers.slice(deltas, axes=[2], starts=[2], ends=[3]) * ww
        dh = fluid.layers.slice(deltas, axes=[2], starts=[3], ends=[4]) * wh

        dw = fluid.layers.clip(dw, -1.e10, np.log(1000. / 16))
        dh = fluid.layers.clip(dh, -1.e10, np.log(1000. / 16))

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = fluid.layers.exp(dw)
        pred_h = fluid.layers.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        x1 = fluid.layers.reshape(x1, shape=(-1, ))
        y1 = fluid.layers.reshape(y1, shape=(-1, ))
        x2 = fluid.layers.reshape(x2, shape=(-1, ))
        y2 = fluid.layers.reshape(y2, shape=(-1, ))

        return x1, y1, x2, y2

    def __call__(self,
                 x,
                 y,
                 inside_weight=None,
                 outside_weight=None,
                 bbox_reg_weight=[0.1, 0.1, 0.2, 0.2]):
        eps = 1.e-10
        x1, y1, x2, y2 = self.bbox_transform(x, bbox_reg_weight)
        x1g, y1g, x2g, y2g = self.bbox_transform(y, bbox_reg_weight)

        x2 = fluid.layers.elementwise_max(x1, x2)
        y2 = fluid.layers.elementwise_max(y1, y2)

        xkis1 = fluid.layers.elementwise_max(x1, x1g)
        ykis1 = fluid.layers.elementwise_max(y1, y1g)
        xkis2 = fluid.layers.elementwise_min(x2, x2g)
        ykis2 = fluid.layers.elementwise_min(y2, y2g)

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

        area_c = (xc2 - xc1) * (yc2 - yc1) + eps
        miouk = iouk - ((area_c - unionk) / area_c)

        iou_weights = 1
        if inside_weight is not None and outside_weight is not None:
            inside_weight = fluid.layers.reshape(inside_weight, shape=(-1, 4))
            outside_weight = fluid.layers.reshape(outside_weight, shape=(-1, 4))

            inside_weight = fluid.layers.reduce_mean(inside_weight, dim=1)
            outside_weight = fluid.layers.reduce_mean(outside_weight, dim=1)

            iou_weights = inside_weight * outside_weight

        class_weight = 2 if self.is_cls_agnostic else self.num_classes
        iouk = fluid.layers.reduce_mean((1 - iouk) * iou_weights) * class_weight
        miouk = fluid.layers.reduce_mean(
            (1 - miouk) * iou_weights) * class_weight

        return miouk * self.loss_weight
