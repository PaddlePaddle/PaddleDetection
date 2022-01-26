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
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

__all__ = ['SmoothL1Loss']

@register
class SmoothL1Loss(nn.Layer):
    """Smooth L1 Loss.
    Args:
        beta (float): controls smooth region, it becomes L1 Loss when beta=0.0
        loss_weight (float): the final loss will be multiplied by this 
    """
    def __init__(self,
                 beta=1.0,
                 loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        assert beta >= 0
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, reduction='none'):
        """forward function, based on fvcore.
        Args:
            pred (Tensor): prediction tensor
            target (Tensor): target tensor, pred.shape must be the same as target.shape
            reduction (str): the way to reduce loss, one of (none, sum, mean)
        """
        assert reduction in ('none', 'sum', 'mean')
        target = target.detach()
        if self.beta < 1e-5:
            loss = paddle.abs(pred - target)
        else:
            n = paddle.abs(pred - target)
            cond = n < self.beta
            loss = paddle.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        if reduction == 'mean':
            loss = loss.mean() if loss.size > 0 else 0.0 * loss.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight
