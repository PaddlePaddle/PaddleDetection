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

import paddle

from ppdet.core.workspace import register, serializable
from ..bbox_utils import bbox_iou

__all__ = ['IouLoss', 'GIoULoss', 'DIouLoss']


@register
@serializable
class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou


@register
@serializable
class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = paddle.maximum(x1, x1g)
        ykis1 = paddle.maximum(y1, y1g)
        xkis2 = paddle.minimum(x2, x2g)
        ykis2 = paddle.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clip(0)
        h_inter = (ykis2 - ykis1).clip(0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = paddle.minimum(x1, x1g)
        yc1 = paddle.minimum(y1, y1g)
        xc2 = paddle.maximum(x2, x2g)
        yc2 = paddle.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = paddle.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh
                        ) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = paddle.sum(giou * iou_weight)
        else:
            loss = paddle.mean(giou * iou_weight)
        return loss * self.loss_weight


@register
@serializable
class DIouLoss(GIoULoss):
    """
    Distance-IoU Loss, see https://arxiv.org/abs/1911.08287
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        use_complete_iou_loss (bool): whether to use complete iou loss
    """

    def __init__(self, loss_weight=1., eps=1e-10, use_complete_iou_loss=True):
        super(DIouLoss, self).__init__(loss_weight=loss_weight, eps=eps)
        self.use_complete_iou_loss = use_complete_iou_loss

    def __call__(self, pbox, gbox, iou_weight=1.):
        x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g
        hg = y2g - y1g

        x2 = paddle.maximum(x1, x2)
        y2 = paddle.maximum(y1, y2)

        # A and B
        xkis1 = paddle.maximum(x1, x1g)
        ykis1 = paddle.maximum(y1, y1g)
        xkis2 = paddle.minimum(x2, x2g)
        ykis2 = paddle.minimum(y2, y2g)

        # A or B
        xc1 = paddle.minimum(x1, x1g)
        yc1 = paddle.minimum(y1, y1g)
        xc2 = paddle.maximum(x2, x2g)
        yc2 = paddle.maximum(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * paddle.greater_than(
            xkis2, xkis1) * paddle.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + self.eps
        iouk = intsctk / unionk

        # DIOU term
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        diou_term = (dist_intersection + self.eps) / (dist_union + self.eps)

        # CIOU term
        ciou_term = 0
        if self.use_complete_iou_loss:
            ar_gt = wg / hg
            ar_pred = w / h
            arctan = paddle.atan(ar_gt) - paddle.atan(ar_pred)
            ar_loss = 4. / np.pi / np.pi * arctan * arctan
            alpha = ar_loss / (1 - iouk + ar_loss + self.eps)
            alpha.stop_gradient = True
            ciou_term = alpha * ar_loss

        diou = paddle.mean((1 - iouk + ciou_term + diou_term) * iou_weight)

        return diou * self.loss_weight
