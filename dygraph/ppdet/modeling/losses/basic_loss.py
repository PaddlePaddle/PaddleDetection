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
from ppdet.core.workspace import register, serializable

__all__ = ['L1Loss']


@register
@serializable
class L1Loss(nn.Layer):
    """
    L1 Loss: abs(x - y)
    """

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
        return paddle.abs(x - y)
