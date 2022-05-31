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

from paddle import fluid
from ppdet.core.workspace import register, serializable

__all__ = ['SmoothL1Loss']


@register
@serializable
class SmoothL1Loss(object):
    '''
    Smooth L1 loss
    Args:
        sigma (float): hyper param in smooth l1 loss 
    '''

    def __init__(self, sigma=1.0):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma

    def __call__(self, x, y, inside_weight=None, outside_weight=None):
        return fluid.layers.smooth_l1(
            x,
            y,
            inside_weight=inside_weight,
            outside_weight=outside_weight,
            sigma=self.sigma)
