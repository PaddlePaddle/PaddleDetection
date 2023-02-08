# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List


def get_bn_running_state_names(model: nn.Layer) -> List[str]:
    """Get all bn state full names including running mean and variance
    """
    names = []
    for n, m in model.named_sublayers():
        if isinstance(m, (nn.BatchNorm2D, nn.SyncBatchNorm)):
            assert hasattr(m, '_mean'), f'assert {m} has _mean'
            assert hasattr(m, '_variance'), f'assert {m} has _variance'
            running_mean = f'{n}._mean'
            running_var = f'{n}._variance'
            names.extend([running_mean, running_var])

    return names
