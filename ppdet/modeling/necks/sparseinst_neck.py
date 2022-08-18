# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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

# This code is based on https://github.com/hustvl/SparseInst
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ..layers import Conv2d
from ..initializer import kaiming_uniform_, kaiming_normal_, reset_parameters

from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['InstanceContextEncoder']


class PyramidPoolingModule(nn.Layer):
    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.LayerList(
            [self._make_stage(in_channels, channels, size) for size in sizes])
        self.bottleneck = Conv2d(in_channels + len(sizes) * channels,
                                 in_channels, 1)
        reset_parameters(self.bottleneck)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        reset_parameters(conv)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        priors = [
            F.interpolate(
                F.relu_(stage(feats)),
                size=(h, w),
                mode='bilinear',
                align_corners=False) for stage in self.stages
        ] + [feats]
        out = F.relu_(self.bottleneck(paddle.concat(priors, 1)))
        return out


@register
@serializable
class InstanceContextEncoder(nn.Layer):
    """ 
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion 
    """

    def __init__(self, num_channels, in_channels):
        super().__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(
                self.num_channels, self.num_channels, 3, padding=1)

            kaiming_uniform_(lateral_conv.weight, a=1)
            kaiming_uniform_(output_conv.weight, a=1)

            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.LayerList(fpn_laterals)
        self.fpn_outputs = nn.LayerList(fpn_outputs)
        # ppm
        self.ppm = PyramidPoolingModule(self.num_channels,
                                        self.num_channels // 4)
        # final fusion
        self.fusion = Conv2d(self.num_channels * 3, self.num_channels, 1)
        kaiming_normal_(self.fusion.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, in_features):
        features = [in_features[i] for i in range(len(in_features) - 1, -1, -1)]
        prev_features = self.ppm(self.fpn_laterals[0](features[0]))
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(
                features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(
                prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [outputs[0]] + [
            F.interpolate(
                x, size, mode='bilinear', align_corners=False)
            for x in outputs[1:]
        ]
        features = self.fusion(paddle.concat(features, axis=1))
        return features

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.num_channels, stride=1)]
