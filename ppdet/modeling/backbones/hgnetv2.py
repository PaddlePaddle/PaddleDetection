# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from paddle.nn.initializer import KaimingNormal, Constant
from paddle.nn import Conv2D, BatchNorm2D, AdaptiveAvgPool2D
from paddle.regularizer import L2Decay
from paddle import ParamAttr
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ["PPHGNetV2"]

kaiming_normal_ = KaimingNormal()
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def get_freeze_norm(ch_out):
    param_attr = ParamAttr(
        learning_rate=0., regularizer=L2Decay(0.), trainable=False)
    bias_attr = ParamAttr(
        learning_rate=0., regularizer=L2Decay(0.), trainable=False)
    global_stats = True

    norm = nn.BatchNorm2D(
        ch_out,
        weight_attr=param_attr,
        bias_attr=bias_attr,
        use_global_stats=global_stats)

    for param in norm.parameters():
        param.stop_gradient = True
    return norm


class ConvBNAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_act=True,
                 use_act1=True,
                 use_act2=True,
                 dw_kernel_size=3,
                 freeze_norm=False,
                 lr=1.0,
                 act="ReLU"):
        super().__init__()
        self.use_act = use_act
        self.conv = Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=False)

        if freeze_norm:
            self.bn = get_freeze_norm(out_channels)
        else:
            self.bn = BatchNorm2D(
                out_channels,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        if self.use_act:
            self.act = eval("nn." + act)()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class ConvBNActDW(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_act=True,
                 use_act1=True,
                 use_act2=True,
                 dw_kernel_size=3,
                 freeze_norm=False,
                 lr=1.0,
                 act="ReLU"):
        super().__init__()
        self.use_act = use_act
        self.use_act1 = use_act1
        self.use_act2 = use_act2
        self.conv1 = Conv2D(
            in_channels,
            out_channels,
            1,
            stride,
            padding=0,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=False)
        self.conv2 = Conv2D(
            out_channels,
            out_channels,
            dw_kernel_size,
            stride,
            padding=(dw_kernel_size - 1) // 2,
            groups=out_channels,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=False)

        if freeze_norm:
            self.bn1 = get_freeze_norm(out_channels)
            self.bn2 = get_freeze_norm(out_channels)
        else:
            self.bn1 = BatchNorm2D(
                out_channels,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
            self.bn2 = BatchNorm2D(
                out_channels,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        if self.use_act:
            self.act = eval("nn." + act)()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.use_act1:
            x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act2:
            x = self.act(x)
        return x


class _StemBlock(nn.Layer):
    def __init__(self,
                 num_input_channels,
                 num_init_features,
                 freeze_norm=False,
                 lr=1.0,
                 out_channel=48):
        super().__init__()
        num_stem_features = int(num_init_features / 2)
        self.stem1 = BasicConv2D(
            num_input_channels,
            num_init_features,
            freeze_norm=freeze_norm,
            lr=lr,
            kernel_size=3,
            stride=2,
            padding=1)
        self.stem2a = BasicConv2D(
            num_init_features,
            num_stem_features,
            freeze_norm=freeze_norm,
            lr=lr,
            kernel_size=2,
            stride=1,
            padding="SAME")
        self.stem2b = BasicConv2D(
            num_stem_features,
            num_init_features,
            freeze_norm=freeze_norm,
            lr=lr,
            kernel_size=2,
            stride=1,
            padding="SAME")
        self.stem3 = BasicConv2D(
            2 * num_init_features,
            num_init_features,
            freeze_norm=freeze_norm,
            lr=lr,
            kernel_size=3,
            stride=2,
            padding=1)
        self.stem4 = BasicConv2D(
            num_init_features,
            out_channel,
            freeze_norm=freeze_norm,
            lr=lr,
            kernel_size=1,
            stride=1,
            padding=0)
        self.pool = nn.MaxPool2D(
            kernel_size=2, stride=1, ceil_mode=True, padding="SAME")

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)
        out = paddle.concat([branch1, branch2], 1)
        out = self.stem3(out)
        out = self.stem4(out)

        return out


class BasicConv2D(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=True,
                 freeze_norm=False,
                 lr=1.0,
                 **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=False,
            **kwargs)
        self.activation = activation
        if freeze_norm:
            self.norm = get_freeze_norm(out_channels)
        else:
            self.norm = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x)
        else:
            return x


class ESEModule(nn.Layer):
    def __init__(self, channels, lr=1.0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv = Conv2D(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=ParamAttr(
                learning_rate=lr, regularizer=L2Decay(0.0)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return paddle.multiply(x=identity, y=x)


class HG_Block(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 layer_num,
                 identity=False,
                 dw_block=True,
                 freeze_norm=False,
                 lr=1.0,
                 dw_act=[True, True],
                 dw_kernel_size=3,
                 act="ReLU"):
        super().__init__()
        self.identity = identity

        self.layers = nn.LayerList()
        block_type = "ConvBNActDW" if dw_block else "ConvBNAct"

        self.layers.append(
            eval(block_type)(in_channels=in_channels,
                             out_channels=mid_channels,
                             kernel_size=3,
                             stride=1,
                             use_act1=dw_act[0],
                             use_act2=dw_act[1],
                             dw_kernel_size=dw_kernel_size,
                             freeze_norm=freeze_norm,
                             lr=lr,
                             act=act))
        for _ in range(layer_num - 1):
            self.layers.append(
                eval(block_type)(in_channels=mid_channels,
                                 out_channels=mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 use_act1=dw_act[0],
                                 use_act2=dw_act[1],
                                 dw_kernel_size=dw_kernel_size,
                                 freeze_norm=freeze_norm,
                                 lr=lr,
                                 act=act))

        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_conv1 = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            freeze_norm=freeze_norm,
            lr=lr,
            stride=1)
        self.aggregation_conv2 = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            freeze_norm=freeze_norm,
            lr=lr,
            stride=1)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = paddle.concat(output, axis=1)
        x = self.aggregation_conv1(x)
        x = self.aggregation_conv2(x)
        if self.identity:
            x += identity
        return x


class HG_Stage(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num,
                 downsample=True,
                 dw_block=True,
                 dw_act=[True, True],
                 dw_kernel_size=3,
                 freeze_norm=False,
                 act="ReLU",
                 lr=1.0):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                freeze_norm=freeze_norm,
                lr=lr,
                use_act=False)

        blocks_list = []
        blocks_list.append(
            HG_Block(
                in_channels,
                mid_channels,
                out_channels,
                layer_num,
                identity=False,
                dw_block=dw_block,
                dw_act=dw_act,
                dw_kernel_size=dw_kernel_size,
                freeze_norm=freeze_norm,
                lr=lr,
                act=act))
        for _ in range(block_num - 1):
            blocks_list.append(
                HG_Block(
                    out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    identity=True,
                    dw_block=dw_block,
                    dw_act=dw_act,
                    dw_kernel_size=dw_kernel_size,
                    freeze_norm=freeze_norm,
                    lr=lr,
                    act=act))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


@register
@serializable
class PPHGNetV2(nn.Layer):
    """
    PPHGNet
    Args:
        stem_channels: list. Stem channel list of PPHGNet.
        stage_config: dict. The configuration of each stage of PPHGNet. [in_channels, mid_channels, out_channels, blocks, downsample, dw_block, dw_kernel_size, act]
        layer_num: int. Number of layers of HG_Block.
    Returns:
        model: nn.Layer. Specific PPHGNet model depends on args.
    """

    def __init__(self,
                 stem_channels,
                 stage_config,
                 layer_num=6,
                 depth_mult=1.0,
                 width_mult=1.0,
                 dw_act=[False, True],
                 return_idx=[0, 1, 2, 3],
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 freeze_stem_only=False,
                 freeze_norm=False,
                 freeze_at=-1,
                 dw_kernel_size=3):
        super().__init__()
        self.return_idx = return_idx
        layer_num = max(round(layer_num * depth_mult), 1)
        self._out_channels = [
            max(round(stage_config[k][2] * width_mult), 1) for k in stage_config
        ]
        self._out_strides = [4, 8, 16, 32]
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        assert len(lr_mult_list) == 4, \
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))

        # stem
        self.stem = _StemBlock(
            num_input_channels=3,
            num_init_features=32,
            freeze_norm=freeze_norm,
            lr=1.0,
            out_channel=stem_channels[-1])

        # stages
        self.stages = nn.LayerList()
        pre_out = -1
        i = 0
        for k in stage_config:
            in_channels, mid_channels, out_channels, block_num, downsample, \
            dw_block, dw_kernel_size, act = stage_config[k]
            in_channels = pre_out if pre_out >= 0 else in_channels
            out_channels = max(round(out_channels * width_mult), 1)
            pre_out = out_channels
            lr_mul = lr_mult_list[i]
            i += 1
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    dw_block,
                    dw_act,
                    dw_kernel_size,
                    freeze_norm=freeze_norm,
                    act=act,
                    lr=lr_mul))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        self._init_weights()

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.stop_gradient = True

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2D)):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                zeros_(m.bias)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
