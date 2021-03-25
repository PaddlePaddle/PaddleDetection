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
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from .resnet import ResNet, Blocks, BasicBlock, BottleNeck

__all__ = ['SENet', 'SERes5Head']


@register
@serializable
class SENet(ResNet):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 std_senet=True,
                 num_stages=4):

        super(SENet, self).__init__(
            depth=depth,
            variant=variant,
            lr_mult_list=lr_mult_list,
            ch_in=128,
            groups=groups,
            base_width=base_width,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            freeze_at=freeze_at,
            return_idx=return_idx,
            dcn_v2_stages=dcn_v2_stages,
            std_senet=std_senet,
            num_stages=num_stages)


@register
class SERes5Head(nn.Layer):
    def __init__(self,
                 depth=50,
                 variant='b',
                 lr_mult=1.0,
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 dcn_v2=False,
                 freeze_norm=False,
                 std_senet=True):
        super(SERes5Head, self).__init__()
        ch_out = 512
        ch_in = 256 if depth < 50 else 1024
        na = NameAdapter(self)
        block = BottleNeck if depth >= 50 else BasicBlock
        self.res5 = Blocks(
            block,
            ch_in,
            ch_out,
            count=3,
            name_adapter=na,
            stage_num=5,
            variant=variant,
            groups=groups,
            base_width=base_width,
            lr=lr_mult,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            dcn_v2=dcn_v2,
            std_senet=std_senet)
        self.ch_out = ch_out * block.expansion

    @property
    def out_shape(self):
        return [ShapeSpec(
            channels=self.ch_out,
            stride=16, )]

    def forward(self, roi_feat):
        y = self.res5(roi_feat)
        return y
