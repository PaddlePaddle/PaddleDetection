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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NumpyArrayInitializer

from paddle import fluid
from ppdet.core.workspace import register, serializable
from .iou_loss import IouLoss

__all__ = ['DiouLossYolo']


@register
@serializable
class DiouLossYolo(IouLoss):
    """
    Distance-IoU Loss, see https://arxiv.org/abs/1911.08287
    Args:
        loss_weight (float): diou loss weight, default is 5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self, loss_weight=5, max_height=608, max_width=608):
        self._loss_weight = loss_weight
        self._MAX_HI = max_height
        self._MAX_WI = max_width

    def __call__(self,
                 x,
                 y,
                 w,
                 h,
                 tx,
                 ty,
                 tw,
                 th,
                 anchors,
                 downsample_ratio,
                 batch_size,
                 eps=1.e-10):
        '''
        Args:
            x  | y | w | h  ([Variables]): the output of yolov3 for encoded x|y|w|h
            tx |ty |tw |th  ([Variables]): the target of yolov3 for encoded x|y|w|h
            anchors ([float]): list of anchors for current output layer
            downsample_ratio (float): the downsample ratio for current output layer
            batch_size (int): training batch size
            eps (float): the decimal to prevent the denominator eqaul zero
        '''
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
        loss_diou = loss_diou * self._loss_weight

        return loss_diou
