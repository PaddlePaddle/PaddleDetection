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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

__all__ = ['PAN']


@register
@serializable
class PAN(nn.Layer):
    """
    Path Aggregation Network, see https://arxiv.org/abs/1803.01534

    Args:
        in_channels (list[int]): input channels of each level which can be
            derived from the output shape of backbone by from_config
        out_channel (list[int]): output channel of each level
        spatial_scales (list[float]): the spatial scales between input feature
            maps and original input image which can be derived from the output
            shape of backbone by from_config
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        norm_type (string|None): The normalization type in FPN module. If
            norm_type is None, norm will not be used after conv and if
            norm_type is string, bn, gn, sync_bn are available. default None
    """

    def __init__(self,
                 in_channels,
                 out_channel,
                 spatial_scales=[0.125, 0.0625, 0.03125],
                 start_level=0,
                 end_level=-1,
                 norm_type=None):
        super(PAN, self).__init__()
        self.out_channel = out_channel
        self.num_ins = len(in_channels)
        self.spatial_scales = spatial_scales
        if end_level == -1:
            self.end_level = self.num_ins
        else:
            # if end_level < inputs, no extra level is allowed
            self.end_level = end_level
            assert end_level <= len(in_channels)
        self.start_level = start_level
        self.norm_type = norm_type
        self.lateral_convs = []

        for i in range(self.start_level, self.end_level):
            in_c = in_channels[i - self.start_level]
            if self.norm_type is not None:
                lateral = self.add_sublayer(
                    'pan_lateral' + str(i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=self.out_channel,
                        filter_size=1,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=in_c)))
            else:
                lateral = self.add_sublayer(
                    'pan_lateral' + str(i),
                    nn.Conv2D(
                        in_channels=in_c,
                        out_channels=self.out_channel,
                        kernel_size=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=in_c))))
            self.lateral_convs.append(lateral)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def forward(self, body_feats):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(body_feats[i + self.start_level]))
        num_levels = len(laterals)
        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='bilinear', )
            laterals[lvl - 1] += upsample

        outs = [laterals[i] for i in range(num_levels)]
        for i in range(0, num_levels - 1):
            outs[i + 1] += F.interpolate(
                outs[i], scale_factor=0.5, mode='bilinear')

        return outs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=1. / s)
            for s in self.spatial_scales
        ]
