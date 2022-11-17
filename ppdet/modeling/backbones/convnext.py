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
'''
Modified from https://github.com/facebookresearch/ConvNeXt
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant

import numpy as np

from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from .transformer_utils import DropPath, trunc_normal_, zeros_

__all__ = ['ConvNeXt']


class Block(nn.Layer):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in Pypaddle
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(
            dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = self.create_parameter(
                shape=(dim, ),
                attr=ParamAttr(initializer=Constant(layer_scale_init_value)))
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Layer):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.weight = self.create_parameter(
            shape=(normalized_shape, ),
            attr=ParamAttr(initializer=Constant(1.)))
        self.bias = self.create_parameter(
            shape=(normalized_shape, ),
            attr=ParamAttr(initializer=Constant(0.)))

        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@register
@serializable
class ConvNeXt(nn.Layer):
    r""" ConvNeXt
        A Pypaddle impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'dims': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'dims': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'dims': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'dims': [256, 512, 1024, 2048]
        },
    }

    def __init__(
            self,
            arch='tiny',
            in_chans=3,
            drop_path_rate=0.,
            layer_scale_init_value=1e-6,
            return_idx=[1, 2, 3],
            norm_output=True,
            pretrained=None, ):
        super().__init__()
        depths = self.arch_settings[arch]['depths']
        dims = self.arch_settings[arch]['dims']
        self.downsample_layers = nn.LayerList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2D(
                in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(
                dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(
                    dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2D(
                    dims[i], dims[i + 1], kernel_size=2, stride=2), )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.LayerList(
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(* [
                Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.return_idx = return_idx
        self.dims = [dims[i] for i in return_idx]  # [::-1]

        self.norm_output = norm_output
        if norm_output:
            self.norms = nn.LayerList([
                LayerNorm(
                    c, eps=1e-6, data_format="channels_first")
                for c in self.dims
            ])

        self.apply(self._init_weights)

        if pretrained is not None:
            if 'http' in pretrained:  #URL
                path = paddle.utils.download.get_weights_path_from_url(
                    pretrained)
            else:  #model in local path
                path = pretrained
            self.set_state_dict(paddle.load(path))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            zeros_(m.bias)

    def forward_features(self, x):
        output = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            output.append(x)

        outputs = [output[i] for i in self.return_idx]
        if self.norm_output:
            outputs = [self.norms[i](out) for i, out in enumerate(outputs)]

        return outputs

    def forward(self, x):
        x = self.forward_features(x['image'])
        return x

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self.dims]
