# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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

# The code is based on:
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/varifocal_loss.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling import ops
from paddle.base.framework import in_dygraph_mode
__all__ = ['VarifocalLoss']


def varifocal_loss(pred,
                   target,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   use_sigmoid=True):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
    """
    # pred and target should be of the same size
    assert len(pred.shape) == len(target.shape) # rank
    if in_dygraph_mode():
        assert pred.shape == target.shape
    if use_sigmoid:
        pred_new = F.sigmoid(pred)
    else:
        pred_new = pred
    target = target.cast(pred.dtype)
    if iou_weighted:
        focal_weight = target * (target > 0.0).cast('float32') + \
            alpha * (pred_new - target).abs().pow(gamma) * \
            (target <= 0.0).cast('float32')
    else:
        focal_weight = (target > 0.0).cast('float32') + \
            alpha * (pred_new - target).abs().pow(gamma) * \
            (target <= 0.0).cast('float32')

    if use_sigmoid:
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
    else:
        loss = F.binary_cross_entropy(
            pred, target, reduction='none') * focal_weight
        loss = loss.sum(axis=1)
    return loss


@register
@serializable
class VarifocalLoss(nn.Layer):
    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        Returns:
            Tensor: The calculated loss
        """
        loss = self.loss_weight * varifocal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            iou_weighted=self.iou_weighted,
            use_sigmoid=self.use_sigmoid)

        if weight is not None:
            loss = loss * weight
        if avg_factor is None:
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if self.reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif self.reduction != 'none':
                raise ValueError(
                    'avg_factor can not be used with reduction="sum"')
        return loss
