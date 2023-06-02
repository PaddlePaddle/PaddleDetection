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
#
# This code is based on: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MultiHeadAttention(nn.MultiHeadAttention):
    def __init__(self, embed_dim, num_heads, output_dim=None):
        super(MultiHeadAttention, self).__init__(embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, output_dim or embed_dim)


# ResNet
class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.relu2 = nn.ReLU()

        self.avgpool = nn.AvgPool2D(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu3 = nn.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2D(stride)), ("0", nn.Conv2D(
                    inplanes,
                    planes * self.expansion,
                    1,
                    stride=1,
                    bias_attr=False)), ("1", nn.BatchNorm2D(planes *
                                                            self.expansion))]))

    def forward(self, x):
        dentity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2D(nn.Layer):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        positional_embedding = self.create_parameter(
            shape=(spacial_dim**2 + 1, embed_dim),
            default_initializer=Assign(
                paddle.randn((spacial_dim**2 + 1, embed_dim)) / embed_dim**0.5))
        self.add_parameter("positional_embedding", positional_embedding)

        self.attn = MultiHeadAttention(embed_dim, num_heads, output_dim)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1],
                       x.shape[2] * x.shape[3])).transpose((2, 0, 1))
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)
        x = x + self.positional_embedding.unsqueeze(1)
        x = x.transpose((1, 0, 2))
        x = self.attn(query=x, key=x, value=x)
        x = x.transpose((1, 0, 2))
        return x[0]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.cast(paddle.float32))
        return ret.cast(orig_type)


class QuickGELU(nn.Layer):
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Layer):
    def __init__(self, d_model, n_head, droplayer_p=0.0, attn_mask=None):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(("c_fc", nn.Linear(d_model, d_model * 4)),
                                 ("gelu", QuickGELU()),
                                 ("c_proj", nn.Linear(d_model * 4, d_model)))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.droplayer_p = droplayer_p

    def get_drop_pattern(self, x):
        if self.training and self.droplayer_p:
            shape = (x.shape[0], ) + (1, ) * (len(x.shape) - 1)
            p = self.droplayer_p * paddle.ones(shape)
            return paddle.bernoulli(p)
        else:
            return 0.0

    def attention(self, x):
        self.attn_mask = self.attn_mask.cast(
            dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, attn_mask=self.attn_mask)

    def forward(self, x):
        y = self.attention(self.ln_1(x))
        drop_pattern = self.get_drop_pattern(y)
        x = x + y * (1.0 - drop_pattern)
        y = self.mlp(self.ln_2(x))
        drop_pattern = self.get_drop_pattern(y)
        x = x + y * (1.0 - drop_pattern)
        return x


class Transformer(nn.Layer):
    def __init__(self,
                 width,
                 layers,
                 heads,
                 stochastic_droplayer_rate=0.0,
                 attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.stochastic_droplayer_rate = stochastic_droplayer_rate
        blocks = []
        for i in range(self.layers):
            droplayer_p = (i / max(self.layers - 1,
                                   1)) * self.stochastic_droplayer_rate
            blocks.append(
                ResidualAttentionBlock(width, heads, droplayer_p, attn_mask))
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.resblocks(x)
