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
from ppdet.core.workspace import register, serializable

__all__ = ['CTFocalLoss']


@register
@serializable
class CTFocalLoss(object):
    """
    CTFocalLoss
    Args:
        loss_weight (float):  loss weight
        gamma (float):  gamma parameter for Focal Loss
    """

    def __init__(self, loss_weight=1., gamma=2.0):
        self.loss_weight = loss_weight
        self.gamma = gamma

    def __call__(self, pred, target):
        """
        Calculate the loss
        Args:
            pred(Tensor): heatmap prediction
            target(Tensor): target for positive samples
        Return:
            ct_focal_loss (Tensor): Focal Loss used in CornerNet & CenterNet.
                Note that the values in target are in [0, 1] since gaussian is
                used to reduce the punishment and we treat [0, 1) as neg example.
        """
        fg_map = paddle.cast(target == 1, 'float32')
        fg_map.stop_gradient = True
        bg_map = paddle.cast(target < 1, 'float32')
        bg_map.stop_gradient = True

        neg_weights = paddle.pow(1 - target, 4) * bg_map
        pos_loss = 0 - paddle.log(pred) * paddle.pow(1 - pred,
                                                     self.gamma) * fg_map
        neg_loss = 0 - paddle.log(1 - pred) * paddle.pow(
            pred, self.gamma) * neg_weights
        pos_loss = paddle.sum(pos_loss)
        neg_loss = paddle.sum(neg_loss)

        fg_num = paddle.sum(fg_map)
        ct_focal_loss = (pos_loss + neg_loss) / (
            fg_num + paddle.cast(fg_num == 0, 'float32'))
        return ct_focal_loss * self.loss_weight
