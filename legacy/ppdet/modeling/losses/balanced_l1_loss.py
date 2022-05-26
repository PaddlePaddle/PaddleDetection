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

__all__ = ['BalancedL1Loss']


@register
@serializable
class BalancedL1Loss(object):
    """
    Balanced L1 Loss, see https://arxiv.org/abs/1904.02701
    Args:
        alpha (float): hyper parameter of BalancedL1Loss, see more details in the paper
        gamma (float): hyper parameter of BalancedL1Loss, see more details in the paper
        beta  (float): hyper parameter of BalancedL1Loss, see more details in the paper
        loss_weights (float): loss weight
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.loss_weight = loss_weight

    def __call__(
            self,
            x,
            y,
            inside_weight=None,
            outside_weight=None, ):
        alpha = self.alpha
        gamma = self.gamma
        beta = self.beta
        loss_weight = self.loss_weight
        diff = fluid.layers.abs(x - y)
        b = np.e**(gamma / alpha) - 1
        less_beta = diff < beta
        ge_beta = diff >= beta
        less_beta = fluid.layers.cast(x=less_beta, dtype='float32')
        ge_beta = fluid.layers.cast(x=ge_beta, dtype='float32')
        less_beta.stop_gradient = True
        ge_beta.stop_gradient = True
        loss_1 = less_beta * (
            alpha / b * (b * diff + 1) * fluid.layers.log(b * diff / beta + 1) -
            alpha * diff)
        loss_2 = ge_beta * (gamma * diff + gamma / b - alpha * beta)
        iou_weights = 1.0
        if inside_weight is not None and outside_weight is not None:
            iou_weights = inside_weight * outside_weight
        loss = (loss_1 + loss_2) * iou_weights
        loss = fluid.layers.reduce_sum(loss, dim=-1) * loss_weight
        return loss
