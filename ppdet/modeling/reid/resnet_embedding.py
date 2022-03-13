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

import os
import paddle
import paddle.nn.functional as F
from paddle import nn
from .resnet import ResNet50, ResNet101
from ppdet.core.workspace import register

__all__ = ['ResNetEmbedding']


@register
class ResNetEmbedding(nn.Layer):
    in_planes = 2048
    def __init__(self, model_name='ResNet50', last_stride=1):
        super(ResNetEmbedding, self).__init__()
        assert model_name in ['ResNet50', 'ResNet101'], "Unsupported ReID arch: {}".format(model_name)
        self.base = eval(model_name)(last_conv_stride=last_stride)
        self.gap = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.bn = nn.BatchNorm1D(self.in_planes, bias_attr=False)

    def forward(self, x):
        base_out = self.base(x)
        global_feat = self.gap(base_out)
        global_feat = self.flatten(global_feat)
        global_feat = self.bn(global_feat)
        return global_feat
