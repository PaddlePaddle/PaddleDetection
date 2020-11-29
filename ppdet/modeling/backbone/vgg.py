from __future__ import division

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.nn import Conv2D, MaxPool2D
from ppdet.core.workspace import register, serializable

__all__ = ['VGG']

VGG_cfg = {16: [2, 2, 3, 3, 3], 19: [2, 2, 4, 4, 4]}


class ConvBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 pool_size=2,
                 pool_stride=2,
                 name=None):
        super(ConvBlock, self).__init__()

        self.groups = groups
        self.conv0 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name=name + "1_weights"),
            bias_attr=False)
        self.conv_out_list = []
        for i in range(1, groups):
            conv_out = self.add_sublayer(
                'conv{}'.format(i),
                Conv2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(
                        name=name + "{}_weights".format(i + 1)),
                    bias_attr=False))
            self.conv_out_list.append(conv_out)

        self.pool = MaxPool2D(
            kernel_size=pool_size,
            stride=pool_stride,
            padding=(pool_size - 1) // 2)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = F.relu(out)
        for conv_i in self.conv_out_list:
            out = conv_i(out)
            out = F.relu(out)
        out = self.pool(out)
        return out


class ExtraBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 padding,
                 stride,
                 kernel_size,
                 name=None):
        super(ExtraBlock, self).__init__()

        self.conv0 = Conv2D(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv1 = Conv2D(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.relu(out)
        return out


class L2NormScale(nn.Layer):
    def __init__(self, num_channels, scale=1.0):
        super(L2NormScale, self).__init__()
        self.scale = self.create_parameter(
            attr=ParamAttr(initializer=paddle.nn.initializer.Constant(scale)),
            shape=[num_channels])

    def forward(self, inputs):
        out = F.normalize(inputs, axis=1, eps=1e-10)
        out = self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            out) * out
        return out


# @register
# @serializable
class VGG(nn.Layer):
    def __init__(self,
                 depth=16,
                 normalizations=[20., -1, -1, -1, -1, -1],
                 extra_block_filters=[[256, 512, 1, 2, 3], [128, 256, 1, 2, 3],
                                      [128, 256, 0, 1, 3],
                                      [128, 256, 0, 1, 3]]):
        super(VGG, self).__init__()

        assert depth in [16, 19], \
                "depth as 16/19 supported currently, but got {}".format(depth)
        self.depth = depth
        self.groups = VGG_cfg[depth]
        self.normalizations = normalizations
        self.extra_block_filters = extra_block_filters

        self.conv_block_0 = ConvBlock(
            3, 64, self.groups[0], 2, 2, name="conv1_")
        self.conv_block_1 = ConvBlock(
            64, 128, self.groups[1], 2, 2, name="conv2_")
        self.conv_block_2 = ConvBlock(
            128, 256, self.groups[2], 2, 2, name="conv3_")
        self.conv_block_3 = ConvBlock(
            256, 512, self.groups[3], 2, 2, name="conv4_")
        self.conv_block_4 = ConvBlock(
            512, 512, self.groups[4], 3, 1, name="conv5_")

        self.fc6 = Conv2D(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=6,
            dilation=6)
        self.fc7 = Conv2D(
            in_channels=1024,
            out_channels=1024,
            kernel_size=1,
            stride=1,
            padding=0)

        # extra block
        self.extra_convs = []
        last_channels = 1024
        for i, v in enumerate(self.extra_block_filters):
            assert len(v) == 5, "extra_block_filters size not fix"
            extra_conv = self.add_sublayer("conv{}".format(6 + i),
                                           ExtraBlock(last_channels, v[0], v[1],
                                                      v[2], v[3], v[4]))
            last_channels = v[1]
            self.extra_convs.append(extra_conv)

        self.norms = []
        for i, n in enumerate(self.normalizations):
            if n != -1:
                norm = self.add_sublayer("norm{}".format(i),
                                         L2NormScale(
                                             self.extra_block_filters[i][1], n))
            else:
                norm = None
            self.norms.append(norm)

    def forward(self, inputs):
        outputs = []

        out = self.conv_block_0(inputs)
        out = self.conv_block_1(out)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        outputs.append(out)

        out = self.conv_block_4(out)
        out = self.fc6(out)
        out = self.fc7(out)
        outputs.append(out)

        if not self.extra_block_filters:
            return out

        # extra block
        for extra_conv in self.extra_convs:
            out = extra_conv(out)
            outputs.append(out)

        for i, n in enumerate(self.normalizations):
            if n != -1:
                outputs[i] = self.norms[i](outputs[i])

        return outputs
