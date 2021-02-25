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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

__all__ = ['FPN']


@register
@serializable
class FPN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channel,
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 has_extra_convs=False,
                 extra_stage=1,
                 use_c5=True,
                 norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 relu_before_extra_convs=True):
        super(FPN, self).__init__()
        self.out_channel = out_channel
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.]
        self.spatial_scales = spatial_scales
        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        self.lateral_convs = []
        self.fpn_convs = []
        fan = out_channel * 3 * 3

        # stage index 0,1,2,3 stands for res2,res3,res4,res5 on ResNet Backbone
        # 0 <= st_stage < ed_stage <= 3
        st_stage = 4 - len(in_channels)
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i - st_stage]
            if self.norm_type == 'gn':
                lateral = self.add_sublayer(
                    lateral_name,
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=out_channel,
                        filter_size=1,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        norm_name=lateral_name + '_norm',
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=in_c),
                        name=lateral_name))
            else:
                lateral = self.add_sublayer(
                    lateral_name,
                    nn.Conv2D(
                        in_channels=in_c,
                        out_channels=out_channel,
                        kernel_size=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=in_c))))
            self.lateral_convs.append(lateral)

            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            if self.norm_type == 'gn':
                fpn_conv = self.add_sublayer(
                    fpn_name,
                    ConvNormLayer(
                        ch_in=out_channel,
                        ch_out=out_channel,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        norm_name=fpn_name + '_norm',
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=fan),
                        name=fpn_name))
            else:
                fpn_conv = self.add_sublayer(
                    fpn_name,
                    nn.Conv2D(
                        in_channels=out_channel,
                        out_channels=out_channel,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=fan))))
            self.fpn_convs.append(fpn_conv)

        # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
        if self.has_extra_convs:
            for i in range(self.extra_stage):
                lvl = ed_stage + 1 + i
                if i == 0 and self.use_c5:
                    in_c = in_channels[-1]
                else:
                    in_c = out_channel
                extra_fpn_name = 'fpn_{}'.format(lvl + 2)
                if self.norm_type == 'gn':
                    extra_fpn_conv = self.add_sublayer(
                        extra_fpn_name,
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=out_channel,
                            filter_size=3,
                            stride=2,
                            norm_type=self.norm_type,
                            norm_decay=self.norm_decay,
                            norm_name=extra_fpn_name + '_norm',
                            freeze_norm=self.freeze_norm,
                            initializer=XavierUniform(fan_out=fan),
                            name=extra_fpn_name))
                else:
                    extra_fpn_conv = self.add_sublayer(
                        extra_fpn_name,
                        nn.Conv2D(
                            in_channels=in_c,
                            out_channels=out_channel,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            weight_attr=ParamAttr(
                                initializer=XavierUniform(fan_out=fan))))
                self.fpn_convs.append(extra_fpn_conv)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }

    def forward(self, body_feats):
        laterals = []
        num_levels = len(body_feats)
        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest', )
            laterals[lvl - 1] += upsample

        fpn_output = []
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        if self.extra_stage > 0:
            # use max pool to get more levels on top of outputs (Faster R-CNN, Mask R-CNN)
            if not self.has_extra_convs:
                assert self.extra_stage == 1, 'extra_stage should be 1 if FPN has not extra convs'
                fpn_output.append(F.max_pool2d(fpn_output[-1], 1, stride=2))
            # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[num_levels](extra_source))

                for i in range(1, self.extra_stage):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[num_levels + i](F.relu(
                            fpn_output[-1])))
                    else:
                        fpn_output.append(self.fpn_convs[num_levels + i](
                            fpn_output[-1]))
        return fpn_output

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=1. / s)
            for s in self.spatial_scales
        ]
