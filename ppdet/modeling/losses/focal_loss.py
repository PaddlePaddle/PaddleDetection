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
import paddle.nn.functional as F
import paddle.nn as nn
from ppdet.core.workspace import register

__all__ = ['FocalLoss', 'Weighted_FocalLoss']

@register
class FocalLoss(nn.Layer):
    """A wrapper around paddle.nn.functional.sigmoid_focal_loss.
    Args:
        use_sigmoid (bool): currently only support use_sigmoid=True
        alpha (float): parameter alpha in Focal Loss
        gamma (float): parameter gamma in Focal Loss
        loss_weight (float): final loss will be multiplied by this
    """
    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.25,
                 gamma=2.0,
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid == True, \
            'Focal Loss only supports sigmoid at the moment'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, target, reduction='none'):
        """forward function.
        Args:
            pred (Tensor): logits of class prediction, of shape (N, num_classes)
            target (Tensor): target class label, of shape (N, )
            reduction (str): the way to reduce loss, one of (none, sum, mean)
        """
        num_classes = pred.shape[1]
        target = F.one_hot(target, num_classes+1).cast(pred.dtype)
        target = target[:, :-1].detach()
        loss = F.sigmoid_focal_loss(
            pred, target, alpha=self.alpha, gamma=self.gamma,
            reduction=reduction)
        return loss * self.loss_weight


@register
class Weighted_FocalLoss(FocalLoss):
    """A wrapper around paddle.nn.functional.sigmoid_focal_loss.
    Args:
        use_sigmoid (bool): currently only support use_sigmoid=True
        alpha (float): parameter alpha in Focal Loss
        gamma (float): parameter gamma in Focal Loss
        loss_weight (float): final loss will be multiplied by this
    """
    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.25,
                 gamma=2.0,
                 loss_weight=1.0,
                 reduction="mean"):
        super(FocalLoss, self).__init__()
        assert use_sigmoid == True, \
            'Focal Loss only supports sigmoid at the moment'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """forward function.
        Args:
            pred (Tensor): logits of class prediction, of shape (N, num_classes)
            target (Tensor): target class label, of shape (N, )
            reduction (str): the way to reduce loss, one of (none, sum, mean)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        num_classes = pred.shape[1]
        target = F.one_hot(target, num_classes + 1).astype(pred.dtype)
        target = target[:, :-1].detach()
        loss = F.sigmoid_focal_loss(
            pred, target, alpha=self.alpha, gamma=self.gamma,
            reduction='none')

        if weight is not None:
            if weight.shape != loss.shape:
                if weight.shape[0] == loss.shape[0]:
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = weight.reshape((-1, 1))
                else:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    assert weight.numel() == loss.numel()
                    weight = weight.reshape((loss.shape[0], -1))
            assert weight.ndim == loss.ndim
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = 1e-10
                loss = loss.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')

        return loss * self.loss_weight
