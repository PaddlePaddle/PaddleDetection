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

import paddle
import paddle.nn as nn
from ppdet.core.workspace import register
from .. import layers as L


@register
class HrnetHead(nn.Layer):
    __inject__ = ['loss']

    def __init__(self,
                 num_joints,
                 loss='HrnetLoss',
                 width=32,
                 data_format='NCHW'):
        """
        Head for Hrnet network

        Args:
            num_joints (int): number of keypoints
            hrloss (object): HrnetLoss instance
            width (int): hrnet channel width
            data_format (str): data format, NCHW or NHWC
        """
        super(HrnetHead, self).__init__()
        self.loss = loss

        self.num_joints = num_joints
        self.final_conv = L.Conv2d(width, num_joints, 1, 1, 0, bias=True)

    def forward(self, feats, targets=None):
        x1 = feats[0]
        hrnet_outputs = self.final_conv(x1)
        if self.training:
            return self.loss(hrnet_outputs, targets)

        return hrnet_outputs
