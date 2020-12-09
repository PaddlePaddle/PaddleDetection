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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Layer
from paddle.nn import Conv2D
from paddle.nn.initializer import XavierUniform
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register, serializable


@register
@serializable
class FPN(Layer):
    def __init__(self,
                 in_channels,
                 out_channel,
                 min_level=0,
                 max_level=4,
                 spatial_scale=[0.25, 0.125, 0.0625, 0.03125]):

        super(FPN, self).__init__()
        self.lateral_convs = []
        self.fpn_convs = []
        fan = out_channel * 3 * 3

        for i in range(min_level, max_level):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i]
            lateral = self.add_sublayer(
                lateral_name,
                Conv2D(
                    in_channels=in_c,
                    out_channels=out_channel,
                    kernel_size=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=in_c)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            self.lateral_convs.append(lateral)

            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            fpn_conv = self.add_sublayer(
                fpn_name,
                Conv2D(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=fan)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            self.fpn_convs.append(fpn_conv)

        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale

    def forward(self, body_feats):
        laterals = []
        for lvl in range(self.min_level, self.max_level):
            laterals.append(self.lateral_convs[lvl](body_feats[lvl]))

        for i in range(self.min_level + 1, self.max_level):
            lvl = self.max_level + self.min_level - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest', )
            laterals[lvl - 1] = laterals[lvl - 1] + upsample

        fpn_output = []
        for lvl in range(self.min_level, self.max_level):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        extension = F.max_pool2d(fpn_output[-1], 1, stride=2)

        spatial_scale = self.spatial_scale + [self.spatial_scale[-1] * 0.5]
        fpn_output.append(extension)
        return fpn_output, spatial_scale
