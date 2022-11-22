# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable

__all__ = ['ProbIoULoss']


def gbb_form(boxes):
    xy, wh, angle = paddle.split(boxes, [2, 2, 1], axis=-1)
    return paddle.concat([xy, wh.pow(2) / 12., angle], axis=-1)


def rotated_form(a_, b_, angles):
    cos_a = paddle.cos(angles)
    sin_a = paddle.sin(angles)
    a = a_ * paddle.pow(cos_a, 2) + b_ * paddle.pow(sin_a, 2)
    b = a_ * paddle.pow(sin_a, 2) + b_ * paddle.pow(cos_a, 2)
    c = (a_ - b_) * cos_a * sin_a
    return a, b, c


def probiou_loss(pred, target, eps=1e-3, mode='l1'):
    """
        pred    -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours predicted box ;in case of HBB angle == 0
        target  -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours target    box ;in case of HBB angle == 0
        eps     -> threshold to avoid infinite values
        mode    -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper

    """

    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:,
                                     0], gbboxes1[:,
                                                  1], gbboxes1[:,
                                                               2], gbboxes1[:,
                                                                            3], gbboxes1[:,
                                                                                         4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:,
                                     0], gbboxes2[:,
                                                  1], gbboxes2[:,
                                                               2], gbboxes2[:,
                                                                            3], gbboxes2[:,
                                                                                         4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    t1 = 0.25 * ((a1 + a2) * (paddle.pow(y1 - y2, 2)) + (b1 + b2) * (paddle.pow(x1 - x2, 2))) + \
         0.5 * ((c1+c2)*(x2-x1)*(y1-y2))
    t2 = (a1 + a2) * (b1 + b2) - paddle.pow(c1 + c2, 2)
    t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)
    t3 = 0.5 * paddle.log(t2 / (4 * paddle.sqrt(F.relu(t3_)) + eps))

    B_d = (t1 / t2) + t3
    # B_d = t1 + t2 + t3

    B_d = paddle.clip(B_d, min=eps, max=100.0)
    l1 = paddle.sqrt(1.0 - paddle.exp(-B_d) + eps)
    l_i = paddle.pow(l1, 2.0)
    l2 = -paddle.log(1.0 - l_i + eps)

    if mode == 'l1':
        probiou = l1
    if mode == 'l2':
        probiou = l2

    return probiou


@serializable
@register
class ProbIoULoss(object):
    """ ProbIoU Loss, refer to https://arxiv.org/abs/2106.06072 for details """

    def __init__(self, mode='l1', eps=1e-3):
        super(ProbIoULoss, self).__init__()
        self.mode = mode
        self.eps = eps

    def __call__(self, pred_rboxes, assigned_rboxes):
        return probiou_loss(pred_rboxes, assigned_rboxes, self.eps, self.mode)
