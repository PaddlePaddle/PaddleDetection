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

import paddle
import paddle.nn as nn

from ppdet.core.workspace import register, serializable
from .resnet import ResNet

__all__ = ['ResNeXt']


@register
@serializable
class ResNeXt(ResNet):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=101,
                 groups=64,
                 base_width=4,
                 base_channels=64,
                 variant='d',
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=None,
                 num_stages=4,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0]):
        super(ResNeXt, self).__init__(depth, variant, norm_type, norm_decay,
                                      freeze_norm, freeze_at, return_idx,
                                      dcn_v2_stages, num_stages, lr_mult_list)
        self._model_type = 'ResNeXt'
        self.groups = groups
        self.base_width = base_width
        self.base_channels = base_channels

