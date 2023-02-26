# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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
"""
this code is base on mmdet: git@github.com:open-mmlab/mmdetection.git
"""
import paddle.nn as nn

from ppdet.core.workspace import register, serializable
from ..backbones.hrnet import ConvNormLayer
from ..shape_spec import ShapeSpec
from ..initializer import xavier_uniform_, constant_

__all__ = ['ChannelMapper']


@register
@serializable
class ChannelMapper(nn.Layer):
    """Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There
            would be extra_convs when num_outs larger than the length
            of in_channels.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 norm_type="gn",
                 norm_groups=32,
                 act='relu',
                 num_outs=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ChannelMapper, self).__init__()
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.LayerList()
        for in_channel in in_channels:
            self.convs.append(
                ConvNormLayer(
                    ch_in=in_channel,
                    ch_out=out_channels,
                    filter_size=kernel_size,
                    norm_type='gn',
                    norm_groups=32,
                    act=act))

        if num_outs > len(in_channels):
            self.extra_convs = nn.LayerList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvNormLayer(
                        ch_in=in_channel,
                        ch_out=out_channels,
                        filter_size=3,
                        stride=2,
                        norm_type='gn',
                        norm_groups=32,
                        act=act))
        self.init_weights()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=1. / s)
            for s in self.spatial_scales
        ]

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.rank() > 1:
                xavier_uniform_(p)
                if hasattr(p, 'bias') and p.bias is not None:
                    constant_(p.bais)
