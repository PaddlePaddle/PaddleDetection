import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

__all__ = ['ChannelMapper']


@register
class ChannelMapper(nn.Layer):
    __shared__ = ['hidden_dim']
    def __init__(self,
                 backbone_num_channels=[512, 1024, 2048],
                 hidden_dim=256,
                 num_feature_levels=4,
                 weight_attr=None,
                 bias_attr=None,
                 ):
        super(ChannelMapper, self).__init__()
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
