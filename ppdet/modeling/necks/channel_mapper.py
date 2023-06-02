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

__all__ = ['ChannelMapper', 'DETRChannelMapper']


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


@register
class DETRChannelMapper(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(
            self,
            backbone_num_channels=[512, 1024, 2048],
            hidden_dim=256,
            num_feature_levels=4,
            weight_attr=None,
            bias_attr=None, ):
        super(DETRChannelMapper, self).__init__()
        assert len(backbone_num_channels) <= num_feature_levels
        self.num_feature_levels = num_feature_levels
        self.input_proj = nn.LayerList()
        for in_channels in backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)))
        in_channels = backbone_num_channels[-1]
        for _ in range(num_feature_levels - len(backbone_num_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)))
            in_channels = hidden_dim

    def _reset_parameters(self):
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_num_channels': [i.channels for i in input_shape], }

    def forward(self, src_feats):
        srcs = []
        for i in range(len(src_feats)):
            srcs.append(self.input_proj[i](src_feats[i]))
        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for i in range(len_srcs, self.num_feature_levels):
                if i == len_srcs:
                    srcs.append(self.input_proj[i](src_feats[-1]))
                else:
                    srcs.append(self.input_proj[i](srcs[-1]))
        return srcs
