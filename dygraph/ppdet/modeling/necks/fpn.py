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
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Layer
from paddle.nn import Conv2D
from paddle.nn.initializer import XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable


@register
@serializable
class FPN(Layer):
    def __init__(self,
                 in_channels,
                 out_channel,
                 min_level=0,
                 max_level=4,
                 spatial_scale=[0.25, 0.125, 0.0625, 0.03125],
                 has_extra_convs=False,
                 use_c5=True,
                 relu_before_extra_convs=True):

        super(FPN, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale
        self.has_extra_convs = has_extra_convs
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs

        self.lateral_convs = []
        self.fpn_convs = []
        fan = out_channel * 3 * 3

        self.num_backbone_stages = len(spatial_scale)
        self.num_outs = self.max_level - self.min_level + 1
        self.highest_backbone_level = self.min_level + self.num_backbone_stages - 1

        for i in range(self.min_level, self.highest_backbone_level + 1):
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

        # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
        if self.has_extra_convs and self.num_outs > self.num_backbone_stages:
            for lvl in range(self.highest_backbone_level + 1,
                             self.max_level + 1):  # P6 P7 ...
                if lvl == self.highest_backbone_level + 1 and self.use_c5:
                    in_c = in_channels[self.highest_backbone_level]
                else:
                    in_c = out_channel
                extra_fpn_name = 'fpn_{}'.format(lvl + 2)
                extra_fpn_conv = self.add_sublayer(
                    extra_fpn_name,
                    Conv2D(
                        in_channels=in_c,
                        out_channels=out_channel,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=fan)),
                        bias_attr=ParamAttr(
                            learning_rate=2., regularizer=L2Decay(0.))))
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, body_feats):
        laterals = []
        used_backbone_levels = len(self.spatial_scale)
        for i in range(used_backbone_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        used_backbone_levels = len(self.spatial_scale)
        for i in range(used_backbone_levels - 1):
            idx = used_backbone_levels - 1 - i
            upsample = F.interpolate(
                laterals[idx],
                scale_factor=2.,
                mode='nearest', )
            laterals[idx - 1] += upsample

        fpn_output = []
        for lvl in range(self.min_level, self.highest_backbone_level + 1):
            i = lvl - self.min_level
            fpn_output.append(self.fpn_convs[i](laterals[i]))

        spatial_scales = self.spatial_scale
        if self.num_outs > len(fpn_output):
            # use max pool to get more levels on top of outputs (Faster R-CNN, Mask R-CNN)
            if not self.has_extra_convs:
                fpn_output.append(F.max_pool2d(fpn_output[-1], 1, stride=2))
                spatial_scales = spatial_scales + [spatial_scales[-1] * 0.5]
            # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[used_backbone_levels](
                    extra_source))
                spatial_scales = spatial_scales + [spatial_scales[-1] * 0.5]
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[i](F.relu(fpn_output[
                            -1])))
                    else:
                        fpn_output.append(self.fpn_convs[i](fpn_output[-1]))
                    spatial_scales = spatial_scales + [spatial_scales[-1] * 0.5]
        return fpn_output, spatial_scales
