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
from ..backbones.hrnet import BasicBlock


@register
class HrHRNetHead(nn.Layer):
    __inject__ = ['loss']

    def __init__(self, num_joints, loss='HrHRNetLoss', swahr=False, width=32):
        """
        Head for HigherHRNet network

        Args:
            num_joints (int): number of keypoints
            hrloss (object): HrHRNetLoss instance
            swahr (bool): whether to use swahr
            width (int): hrnet channel width
        """
        super(HrHRNetHead, self).__init__()
        self.loss = loss

        self.num_joints = num_joints
        num_featout1 = num_joints * 2
        num_featout2 = num_joints
        self.swahr = swahr
        self.conv1 = L.Conv2d(width, num_featout1, 1, 1, 0, bias=True)
        self.conv2 = L.Conv2d(width, num_featout2, 1, 1, 0, bias=True)
        self.deconv = nn.Sequential(
            L.ConvTranspose2d(
                num_featout1 + width, width, 4, 2, 1, 0, bias=False),
            L.BatchNorm2d(width),
            L.ReLU())
        self.blocks = nn.Sequential(*(BasicBlock(
            num_channels=width,
            num_filters=width,
            has_se=False,
            freeze_norm=False,
            name='HrHRNetHead_{}'.format(i)) for i in range(4)))

        self.interpolate = L.Upsample(2, mode='bilinear')
        self.concat = L.Concat(dim=1)
        if swahr:
            self.scalelayer0 = nn.Sequential(
                L.Conv2d(
                    width, num_joints, 1, 1, 0, bias=True),
                L.BatchNorm2d(num_joints),
                L.ReLU(),
                L.Conv2d(
                    num_joints,
                    num_joints,
                    9,
                    1,
                    4,
                    groups=num_joints,
                    bias=True))
            self.scalelayer1 = nn.Sequential(
                L.Conv2d(
                    width, num_joints, 1, 1, 0, bias=True),
                L.BatchNorm2d(num_joints),
                L.ReLU(),
                L.Conv2d(
                    num_joints,
                    num_joints,
                    9,
                    1,
                    4,
                    groups=num_joints,
                    bias=True))

    def forward(self, feats, targets=None):
        x1 = feats[0]
        xo1 = self.conv1(x1)
        x2 = self.blocks(self.deconv(self.concat((x1, xo1))))
        xo2 = self.conv2(x2)
        num_joints = self.num_joints
        if self.training:
            heatmap1, tagmap = paddle.split(xo1, 2, axis=1)
            if self.swahr:
                so1 = self.scalelayer0(x1)
                so2 = self.scalelayer1(x2)
                hrhrnet_outputs = ([heatmap1, so1], [xo2, so2], tagmap)
                return self.loss(hrhrnet_outputs, targets)
            else:
                hrhrnet_outputs = (heatmap1, xo2, tagmap)
                return self.loss(hrhrnet_outputs, targets)

        # averaged heatmap, upsampled tagmap
        upsampled = self.interpolate(xo1)
        avg = (upsampled[:, :num_joints] + xo2[:, :num_joints]) / 2
        return avg, upsampled[:, num_joints:]
