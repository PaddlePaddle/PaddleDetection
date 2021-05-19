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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant
from ppdet.core.workspace import register

__all__ = ['FairMOTLoss']


@register
class FairMOTLoss(nn.Layer):
    def __init__(self):
        super(FairMOTLoss, self).__init__()
        self.det_weight = self.create_parameter(
            shape=[1], default_initializer=Constant(-1.85))
        self.reid_weight = self.create_parameter(
            shape=[1], default_initializer=Constant(-1.05))

    def forward(self, det_loss, reid_loss):
        loss = paddle.exp(-self.det_weight) * det_loss + paddle.exp(
            -self.reid_weight) * reid_loss + (self.det_weight + self.reid_weight
                                              )
        loss *= 0.5
        return {'loss': loss}
