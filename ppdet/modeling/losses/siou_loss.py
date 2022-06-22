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

import paddle
import math

from paddle import Tensor

from ppdet.core.workspace import register, serializable

__all__ = ['SIoULoss', ]


@register
@serializable
class SIoULoss(object):
    '''SIoU Loss: More Powerful Learning for Bounding Box Regression, 
        https://arxiv.org/pdf/2205.12740.pdf
    '''

    def __init__(self,
                 theta=2,
                 loss_weight=1.0,
                 eps: float=1e-9,
                 box_fmt='xyxy',
                 keep_dim=False,
                 reduction='none') -> None:
        self.theta = theta
        self.eps = eps
        self.loss_weight = loss_weight
        self.box_fmt = box_fmt
        self.reduction = reduction
        self.keep_dim = keep_dim

    def __call__(self, boxa: Tensor, boxb: Tensor):
        loss = siou(
            boxa,
            boxb,
            theta=self.theta,
            box_fmt=self.box_fmt,
            eps=self.eps,
            reduction=self.reduction)

        if self.keep_dim:
            loss = loss.unsqueeze(-1)

        return self.loss_weight * loss


def angle_cost(
        boxa: Tensor,
        boxb: Tensor,
        box_fmt='cxcywh',
        eps=1e-7,
        reduction='none', ):
    '''angle cost
    args:
        boxa, pred, [..., 4]
        boxb, gt  , [..., 4]
    '''
    boxa = box_convert(boxa, in_fmt=box_fmt, out_fmt='cxcywh')
    boxb = box_convert(boxb, in_fmt=box_fmt, out_fmt='cxcywh')

    cxa, cya, _, _ = boxa.unbind(-1)
    cxb, cyb, _, _ = boxb.unbind(-1)

    # ch = paddle.maximum(cyb, cya) - paddle.minimum(cyb, cya)
    # cw = paddle.maximum(cxb, cxa) - paddle.minimum(cxb, cxa)

    sigma = ((cxb - cxa).pow(2) + (cyb - cya).pow(2)).sqrt()
    alpha_w = paddle.abs(cxb - cxa) / sigma
    alpha_h = paddle.abs(cyb - cya) / sigma
    threshold = pow(2, 0.5) / 2
    alpha = paddle.where(alpha_w > threshold, alpha_h, alpha_w)

    angle = paddle.asin(alpha)

    # angle = paddle.asin(paddle.clip(ch / (sigma + eps), min=-1, max=1))
    # angle = paddle.atan2(ch, cw)

    loss_angle = 1 - 2 * paddle.sin(angle - math.pi / 4).pow(2)

    return reduction_tensor(loss_angle, reduction=reduction)


def distance_cost(
        boxa: Tensor,
        boxb: Tensor,
        box_fmt='cxcywh',
        eps=1e-7,
        reduction='none', ):
    '''distance cost
    args:
        boxa, pred, [..., 4]
        boxb, gt  , [..., 4]
    '''
    boxa = box_convert(boxa, in_fmt=box_fmt, out_fmt='cxcywh')
    boxb = box_convert(boxb, in_fmt=box_fmt, out_fmt='cxcywh')

    cxa, cya, ha, wa = boxa.unbind(-1)
    cxb, cyb, hb, wb = boxb.unbind(-1)

    loss_angle = angle_cost(boxa, boxb, eps=eps, reduction='none')

    ch = paddle.maximum(cyb, cya) - paddle.minimum(cyb, cya) + (ha + hb) / 2
    cw = paddle.maximum(cxb, cxa) - paddle.minimum(cxb, cxa) + (wa + wb) / 2

    r_x = ((cxb - cxa) / (cw + eps)).pow(2)
    r_y = ((cyb - cya) / (ch + eps)).pow(2)
    gamma = 2 - loss_angle

    loss_distance_x = 1 - math.e**(-gamma * r_x)
    loss_distance_y = 1 - math.e**(-gamma * r_y)
    loss_distance = loss_distance_x + loss_distance_y

    return reduction_tensor(loss_distance, reduction=reduction)


def shape_cost(
        boxa: Tensor,
        boxb: Tensor,
        theta=2,
        box_fmt='cxcywh',
        eps=1e-7,
        reduction='none', ):
    '''shape cost
    args:
        boxa, pred, [..., 4]
        boxb, gt  , [..., 4]
    '''
    boxa = box_convert(boxa, in_fmt=box_fmt, out_fmt='cxcywh')
    boxb = box_convert(boxb, in_fmt=box_fmt, out_fmt='cxcywh')

    _, _, wa, ha = boxa.unbind(-1)
    _, _, wb, hb = boxb.unbind(-1)

    w_w = paddle.abs(wa - wb) / (paddle.maximum(wa, wb) + eps)
    w_h = paddle.abs(ha - hb) / (paddle.maximum(ha, hb) + eps)
    loss_shape_x = (1 - math.e**(-w_w))**theta
    loss_shape_y = (1 - math.e**(-w_h))**theta
    loss_shape = loss_shape_x + loss_shape_y

    return reduction_tensor(loss_shape, reduction=reduction)


def box_iou(
        boxa: Tensor,
        boxb: Tensor,
        box_fmt='cxcywh',
        eps=1e-7,
        reduction='none', ):
    '''iou cost
    args:
        boxa, pred, [..., 4]
        boxb, gt  , [..., 4]
    '''
    boxa = box_convert(boxa, in_fmt=box_fmt, out_fmt='xyxy')
    boxb = box_convert(boxb, in_fmt=box_fmt, out_fmt='xyxy')

    b1_x1, b1_y1, b1_x2, b1_y2 = boxa.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxb.unbind(-1)

    # Intersection
    inter = (paddle.minimum(b1_x2, b2_x2) - paddle.maximum(b1_x1, b2_x1)).clip(0) * \
            (paddle.minimum(b1_y2, b2_y2) - paddle.maximum(b1_y1, b2_y1)).clip(0)

    # Union
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return reduction_tensor(iou, reduction=reduction)


def siou(boxa: Tensor,
         boxb: Tensor,
         box_fmt='cxcywh',
         theta=4,
         eps: float=1e-7,
         reduction='none'):
    '''SIoU Loss: More Powerful Learning for Bounding Box Regression, https://arxiv.org/pdf/2205.12740.pdf
    args:
        boxa, pred, [..., 4]
        boxb, gt  , [..., 4]
    '''
    loss_shape = shape_cost(
        boxa,
        boxb,
        theta=theta,
        box_fmt=box_fmt,
        eps=eps,
        reduction='none', )
    loss_distance = distance_cost(
        boxa, boxb, box_fmt=box_fmt, eps=eps, reduction='none')
    iou = box_iou(boxa, boxb, eps=eps, box_fmt=box_fmt, reduction='none')

    loss = 1 - iou + (loss_distance + loss_shape) / 2

    return reduction_tensor(loss, reduction=reduction)


def box_convert(boxes: Tensor, in_fmt='xyxy', out_fmt='cxcywh'):
    '''boxes convert
    '''
    if in_fmt == out_fmt:
        return boxes

    if in_fmt == 'xyxy' and out_fmt == 'cxcywh':
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        return paddle.stack((cx, cy, w, h), axis=-1)

    elif in_fmt == 'cxcywh' and out_fmt == 'xyxy':
        cx, cy, w, h = boxes.unbind(-1)
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        return paddle.stack((x1, y1, x2, y2), axis=-1)

    else:
        raise AttributeError('')


def reduction_tensor(tensor: Tensor, reduction: str='none'):
    if reduction == 'none':
        return tensor
    elif reduction == 'sum':
        return tensor.sum()
    elif reduction == 'mean':
        return tensor.mean()
    else:
        raise AttributeError('')