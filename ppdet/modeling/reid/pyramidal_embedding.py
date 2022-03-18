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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant
from paddle import ParamAttr
from .resnet import ResNet50, ResNet101
from ppdet.core.workspace import register

__all__ = ['PCBPyramid']


@register
class PCBPyramid(nn.Layer):
    """
    PCB (Part-based Convolutional Baseline), see https://arxiv.org/abs/1711.09349,
    Pyramidal Person Re-IDentification, see https://arxiv.org/abs/1810.12193

    Args:
        input_ch (int): Number of channels of the input feature.
        num_stripes (int): Number of sub-parts.
        used_levels (tuple): Whether the level is used, 1 means used.
        num_classes (int): Number of classes for identities, default 751 in
            Market-1501 dataset.
        last_conv_stride (int): Stride of the last conv.
        last_conv_dilation (int): Dilation of the last conv.
        num_conv_out_channels (int): Number of channels of conv feature.
    """

    def __init__(self,
                 input_ch=2048,
                 model_name='ResNet101',
                 num_stripes=6,
                 used_levels=(1, 1, 1, 1, 1, 1),
                 num_classes=751,
                 last_conv_stride=1,
                 last_conv_dilation=1,
                 num_conv_out_channels=128):
        super(PCBPyramid, self).__init__()
        self.num_stripes = num_stripes
        self.used_levels = used_levels
        self.num_classes = num_classes

        self.num_in_each_level = [i for i in range(self.num_stripes, 0, -1)]
        self.num_branches = sum(self.num_in_each_level)

        assert model_name in ['ResNet50', 'ResNet101'], "Unsupported ReID arch: {}".format(model_name)
        self.base = eval(model_name)(
            lr_mult=0.1,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.pyramid_conv_list0, self.pyramid_fc_list0 = self.basic_branch(
            num_conv_out_channels, input_ch)

    def basic_branch(self, num_conv_out_channels, input_ch):
        # the level indexes are defined from fine to coarse,
        # the branch will contain one more part than that of its previous level
        # the sliding step is set to 1
        pyramid_conv_list = nn.LayerList()
        pyramid_fc_list = nn.LayerList()

        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            pyramid_conv_list.append(
                nn.Sequential(
                    nn.Conv2D(input_ch, num_conv_out_channels, 1),
                    nn.BatchNorm2D(num_conv_out_channels), nn.ReLU()))

        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            fc = nn.Linear(
                in_features=num_conv_out_channels,
                out_features=self.num_classes,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.001)),
                bias_attr=ParamAttr(initializer=Constant(value=0.)))
            pyramid_fc_list.append(fc)
        return pyramid_conv_list, pyramid_fc_list

    def pyramid_forward(self, feat):
        each_stripe_size = int(feat.shape[2] / self.num_stripes)

        feat_list, logits_list = [], []
        idx_levels = 0
        used_branches = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1
            idx_in_each_level = idx_branches - sum(self.num_in_each_level[
                0:idx_levels])
            stripe_size_in_each_level = each_stripe_size * (idx_levels + 1)
            start = idx_in_each_level * each_stripe_size
            end = start + stripe_size_in_each_level

            k = feat.shape[-1]
            local_feat_avgpool = F.avg_pool2d(
                feat[:, :, start:end, :],
                kernel_size=(stripe_size_in_each_level, k))
            local_feat_maxpool = F.max_pool2d(
                feat[:, :, start:end, :],
                kernel_size=(stripe_size_in_each_level, k))
            local_feat = local_feat_avgpool + local_feat_maxpool

            local_feat = self.pyramid_conv_list0[used_branches](local_feat)
            local_feat = paddle.reshape(
                local_feat, shape=[local_feat.shape[0], -1])
            feat_list.append(local_feat)

            local_logits = self.pyramid_fc_list0[used_branches](
                self.dropout_layer(local_feat))
            logits_list.append(local_logits)

            used_branches += 1

        return feat_list, logits_list

    def forward(self, x):
        feat = self.base(x)
        assert feat.shape[2] % self.num_stripes == 0
        feat_list, logits_list = self.pyramid_forward(feat)
        feat_out = paddle.concat(feat_list, axis=-1)
        return feat_out
