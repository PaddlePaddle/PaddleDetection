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
"""
This code is based on
https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from numbers import Integral
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant
from ppdet.core.workspace import register
from ppdet.modeling.shape_spec import ShapeSpec
from ppdet.modeling.ops import channel_shuffle
from .. import layers as L

__all__ = ['LiteHRNet']


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride=1,
                 groups=1,
                 norm_type=None,
                 norm_groups=32,
                 norm_decay=0.,
                 freeze_norm=False,
                 act=None):
        super(ConvNormLayer, self).__init__()
        self.act = act
        norm_lr = 0. if freeze_norm else 1.
        if norm_type is not None:
            assert norm_type in ['bn', 'sync_bn', 'gn'], \
                "norm_type should be one of ['bn', 'sync_bn', 'gn'], but got {}".format(norm_type)
            param_attr = ParamAttr(
                initializer=Constant(1.0),
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay), )
            bias_attr = ParamAttr(
                learning_rate=norm_lr, regularizer=L2Decay(norm_decay))
            global_stats = True if freeze_norm else None
            if norm_type in ['bn', 'sync_bn']:
                self.norm = nn.BatchNorm2D(
                    ch_out,
                    weight_attr=param_attr,
                    bias_attr=bias_attr,
                    use_global_stats=global_stats, )
            elif norm_type == 'gn':
                self.norm = nn.GroupNorm(
                    num_groups=norm_groups,
                    num_channels=ch_out,
                    weight_attr=param_attr,
                    bias_attr=bias_attr)
            norm_params = self.norm.parameters()
            if freeze_norm:
                for param in norm_params:
                    param.stop_gradient = True
            conv_bias_attr = False
        else:
            conv_bias_attr = True
            self.norm = None

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.001)),
            bias_attr=conv_bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)

        if self.act == 'relu':
            out = F.relu(out)
        elif self.act == 'sigmoid':
            out = F.sigmoid(out)
        return out


class DepthWiseSeparableConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride=1,
                 dw_norm_type=None,
                 pw_norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 dw_act=None,
                 pw_act=None):
        super(DepthWiseSeparableConvNormLayer, self).__init__()
        self.depthwise_conv = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_in,
            filter_size=filter_size,
            stride=stride,
            groups=ch_in,
            norm_type=dw_norm_type,
            act=dw_act,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm, )
        self.pointwise_conv = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            norm_type=pw_norm_type,
            act=pw_act,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm, )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class CrossResolutionWeightingModule(nn.Layer):
    def __init__(self,
                 channels,
                 ratio=16,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.):
        super(CrossResolutionWeightingModule, self).__init__()
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvNormLayer(
            ch_in=total_channel,
            ch_out=total_channel // ratio,
            filter_size=1,
            stride=1,
            norm_type=norm_type,
            act='relu',
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)
        self.conv2 = ConvNormLayer(
            ch_in=total_channel // ratio,
            ch_out=total_channel,
            filter_size=1,
            stride=1,
            norm_type=norm_type,
            act='sigmoid',
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)

    def forward(self, x):
        mini_size = x[-1].shape[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = paddle.concat(out, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = paddle.split(out, self.channels, 1)
        out = [
            s * F.interpolate(
                a, s.shape[-2:], mode='nearest') for s, a in zip(x, out)
        ]
        return out


class SpatialWeightingModule(nn.Layer):
    def __init__(self, in_channel, ratio=16, freeze_norm=False, norm_decay=0.):
        super(SpatialWeightingModule, self).__init__()
        self.global_avgpooling = nn.AdaptiveAvgPool2D(1)
        self.conv1 = ConvNormLayer(
            ch_in=in_channel,
            ch_out=in_channel // ratio,
            filter_size=1,
            stride=1,
            act='relu',
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)
        self.conv2 = ConvNormLayer(
            ch_in=in_channel // ratio,
            ch_out=in_channel,
            filter_size=1,
            stride=1,
            act='sigmoid',
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)

    def forward(self, x):
        out = self.global_avgpooling(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class ConditionalChannelWeightingBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.):
        super(ConditionalChannelWeightingBlock, self).__init__()
        assert stride in [1, 2]
        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeightingModule(
            branch_channels,
            ratio=reduce_ratio,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)
        self.depthwise_convs = nn.LayerList([
            ConvNormLayer(
                channel,
                channel,
                filter_size=3,
                stride=stride,
                groups=channel,
                norm_type=norm_type,
                freeze_norm=freeze_norm,
                norm_decay=norm_decay) for channel in branch_channels
        ])

        self.spatial_weighting = nn.LayerList([
            SpatialWeightingModule(
                channel,
                ratio=4,
                freeze_norm=freeze_norm,
                norm_decay=norm_decay) for channel in branch_channels
        ])

    def forward(self, x):
        x = [s.chunk(2, axis=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        out = [paddle.concat([s1, s2], axis=1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, groups=2) for s in out]
        return out


class ShuffleUnit(nn.Layer):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.):
        super(ShuffleUnit, self).__init__()
        branch_channel = out_channel // 2
        self.stride = stride
        if self.stride == 1:
            assert in_channel == branch_channel * 2, \
                "when stride=1, in_channel {} should equal to branch_channel*2 {}".format(in_channel, branch_channel * 2)
        if stride > 1:
            self.branch1 = nn.Sequential(
                ConvNormLayer(
                    ch_in=in_channel,
                    ch_out=in_channel,
                    filter_size=3,
                    stride=self.stride,
                    groups=in_channel,
                    norm_type=norm_type,
                    freeze_norm=freeze_norm,
                    norm_decay=norm_decay),
                ConvNormLayer(
                    ch_in=in_channel,
                    ch_out=branch_channel,
                    filter_size=1,
                    stride=1,
                    norm_type=norm_type,
                    act='relu',
                    freeze_norm=freeze_norm,
                    norm_decay=norm_decay), )
        self.branch2 = nn.Sequential(
            ConvNormLayer(
                ch_in=branch_channel if stride == 1 else in_channel,
                ch_out=branch_channel,
                filter_size=1,
                stride=1,
                norm_type=norm_type,
                act='relu',
                freeze_norm=freeze_norm,
                norm_decay=norm_decay),
            ConvNormLayer(
                ch_in=branch_channel,
                ch_out=branch_channel,
                filter_size=3,
                stride=self.stride,
                groups=branch_channel,
                norm_type=norm_type,
                freeze_norm=freeze_norm,
                norm_decay=norm_decay),
            ConvNormLayer(
                ch_in=branch_channel,
                ch_out=branch_channel,
                filter_size=1,
                stride=1,
                norm_type=norm_type,
                act='relu',
                freeze_norm=freeze_norm,
                norm_decay=norm_decay), )

    def forward(self, x):
        if self.stride > 1:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
        else:
            x1, x2 = x.chunk(2, axis=1)
            x2 = self.branch2(x2)
        out = paddle.concat([x1, x2], axis=1)
        out = channel_shuffle(out, groups=2)
        return out


class IterativeHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.):
        super(IterativeHead, self).__init__()
        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]

        projects = []
        for i in range(num_branches):
            if i != num_branches - 1:
                projects.append(
                    DepthWiseSeparableConvNormLayer(
                        ch_in=self.in_channels[i],
                        ch_out=self.in_channels[i + 1],
                        filter_size=3,
                        stride=1,
                        dw_act=None,
                        pw_act='relu',
                        dw_norm_type=norm_type,
                        pw_norm_type=norm_type,
                        freeze_norm=freeze_norm,
                        norm_decay=norm_decay))
            else:
                projects.append(
                    DepthWiseSeparableConvNormLayer(
                        ch_in=self.in_channels[i],
                        ch_out=self.in_channels[i],
                        filter_size=3,
                        stride=1,
                        dw_act=None,
                        pw_act='relu',
                        dw_norm_type=norm_type,
                        pw_norm_type=norm_type,
                        freeze_norm=freeze_norm,
                        norm_decay=norm_decay))
        self.projects = nn.LayerList(projects)

    def forward(self, x):
        x = x[::-1]
        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class Stem(nn.Layer):
    def __init__(self,
                 in_channel,
                 stem_channel,
                 out_channel,
                 expand_ratio,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.):
        super(Stem, self).__init__()
        self.conv1 = ConvNormLayer(
            in_channel,
            stem_channel,
            filter_size=3,
            stride=2,
            norm_type=norm_type,
            act='relu',
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)
        mid_channel = int(round(stem_channel * expand_ratio))
        branch_channel = stem_channel // 2
        if stem_channel == out_channel:
            inc_channel = out_channel - branch_channel
        else:
            inc_channel = out_channel - stem_channel
        self.branch1 = nn.Sequential(
            ConvNormLayer(
                ch_in=branch_channel,
                ch_out=branch_channel,
                filter_size=3,
                stride=2,
                groups=branch_channel,
                norm_type=norm_type,
                freeze_norm=freeze_norm,
                norm_decay=norm_decay),
            ConvNormLayer(
                ch_in=branch_channel,
                ch_out=inc_channel,
                filter_size=1,
                stride=1,
                norm_type=norm_type,
                act='relu',
                freeze_norm=freeze_norm,
                norm_decay=norm_decay), )
        self.expand_conv = ConvNormLayer(
            ch_in=branch_channel,
            ch_out=mid_channel,
            filter_size=1,
            stride=1,
            norm_type=norm_type,
            act='relu',
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)
        self.depthwise_conv = ConvNormLayer(
            ch_in=mid_channel,
            ch_out=mid_channel,
            filter_size=3,
            stride=2,
            groups=mid_channel,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)
        self.linear_conv = ConvNormLayer(
            ch_in=mid_channel,
            ch_out=branch_channel
            if stem_channel == out_channel else stem_channel,
            filter_size=1,
            stride=1,
            norm_type=norm_type,
            act='relu',
            freeze_norm=freeze_norm,
            norm_decay=norm_decay)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, axis=1)
        x1 = self.branch1(x1)
        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)
        out = paddle.concat([x1, x2], axis=1)
        out = channel_shuffle(out, groups=2)

        return out


class LiteHRNetModule(nn.Layer):
    def __init__(self,
                 num_branches,
                 num_blocks,
                 in_channels,
                 reduce_ratio,
                 module_type,
                 multiscale_output=False,
                 with_fuse=True,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.):
        super(LiteHRNetModule, self).__init__()
        assert num_branches == len(in_channels),\
            "num_branches {} should equal to num_in_channels {}".format(num_branches, len(in_channels))
        assert module_type in [
            'LITE', 'NAIVE'
        ], "module_type should be one of ['LITE', 'NAIVE']"
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_type = 'bn'
        self.module_type = module_type

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(
                num_blocks,
                reduce_ratio,
                freeze_norm=freeze_norm,
                norm_decay=norm_decay)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(
                num_branches,
                num_blocks,
                freeze_norm=freeze_norm,
                norm_decay=norm_decay)

        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers(
                freeze_norm=freeze_norm, norm_decay=norm_decay)
            self.relu = nn.ReLU()

    def _make_weighting_blocks(self,
                               num_blocks,
                               reduce_ratio,
                               stride=1,
                               freeze_norm=False,
                               norm_decay=0.):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeightingBlock(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    norm_type=self.norm_type,
                    freeze_norm=freeze_norm,
                    norm_decay=norm_decay))
        return nn.Sequential(*layers)

    def _make_naive_branches(self,
                             num_branches,
                             num_blocks,
                             freeze_norm=False,
                             norm_decay=0.):
        branches = []
        for branch_idx in range(num_branches):
            layers = []
            for i in range(num_blocks):
                layers.append(
                    ShuffleUnit(
                        self.in_channels[branch_idx],
                        self.in_channels[branch_idx],
                        stride=1,
                        norm_type=self.norm_type,
                        freeze_norm=freeze_norm,
                        norm_decay=norm_decay))
            branches.append(nn.Sequential(*layers))
        return nn.LayerList(branches)

    def _make_fuse_layers(self, freeze_norm=False, norm_decay=0.):
        if self.num_branches == 1:
            return None
        fuse_layers = []
        num_out_branches = self.num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            L.Conv2d(
                                self.in_channels[j],
                                self.in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False, ),
                            nn.BatchNorm2D(self.in_channels[i]),
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    L.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=self.in_channels[j],
                                        bias=False, ),
                                    nn.BatchNorm2D(self.in_channels[j]),
                                    L.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False, ),
                                    nn.BatchNorm2D(self.in_channels[i])))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    L.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=self.in_channels[j],
                                        bias=False, ),
                                    nn.BatchNorm2D(self.in_channels[j]),
                                    L.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False, ),
                                    nn.BatchNorm2D(self.in_channels[j]),
                                    nn.ReLU()))

                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.LayerList(fuse_layer))

        return nn.LayerList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.layers[0](x[0])]
        if self.module_type == 'LITE':
            out = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x
        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if j == 0:
                        y += y
                    elif i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                    if i == 0:
                        out[i] = y
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out


@register
class LiteHRNet(nn.Layer):
    """
    @inproceedings{Yulitehrnet21,
    title={Lite-HRNet: A Lightweight High-Resolution Network},
        author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
        booktitle={CVPR},year={2021}
    }
    Args:
        network_type (str): the network_type should be one of ["lite_18", "lite_30", "naive", "wider_naive"],
            "naive": Simply combining the shuffle block in ShuffleNet and the highresolution design pattern in HRNet.
            "wider_naive": Naive network with wider channels in each block.
            "lite_18": Lite-HRNet-18, which replaces the pointwise convolution in a shuffle block by conditional channel weighting.
            "lite_30": Lite-HRNet-30, with more blocks compared with Lite-HRNet-18.
        freeze_at (int): the stage to freeze
        freeze_norm (bool): whether to freeze norm in HRNet
        norm_decay (float): weight decay for normalization layer weights
        return_idx (List): the stage to return
    """

    def __init__(self,
                 network_type,
                 freeze_at=0,
                 freeze_norm=True,
                 norm_decay=0.,
                 return_idx=[0, 1, 2, 3]):
        super(LiteHRNet, self).__init__()
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert network_type in ["lite_18", "lite_30", "naive", "wider_naive"], \
            "the network_type should be one of [lite_18, lite_30, naive, wider_naive]"
        assert len(return_idx) > 0, "need one or more return index"
        self.freeze_at = freeze_at
        self.freeze_norm = freeze_norm
        self.norm_decay = norm_decay
        self.return_idx = return_idx
        self.norm_type = 'bn'

        self.module_configs = {
            "lite_18": {
                "num_modules": [2, 4, 2],
                "num_branches": [2, 3, 4],
                "num_blocks": [2, 2, 2],
                "module_type": ["LITE", "LITE", "LITE"],
                "reduce_ratios": [8, 8, 8],
                "num_channels": [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            },
            "lite_30": {
                "num_modules": [3, 8, 3],
                "num_branches": [2, 3, 4],
                "num_blocks": [2, 2, 2],
                "module_type": ["LITE", "LITE", "LITE"],
                "reduce_ratios": [8, 8, 8],
                "num_channels": [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            },
            "naive": {
                "num_modules": [2, 4, 2],
                "num_branches": [2, 3, 4],
                "num_blocks": [2, 2, 2],
                "module_type": ["NAIVE", "NAIVE", "NAIVE"],
                "reduce_ratios": [1, 1, 1],
                "num_channels": [[30, 60], [30, 60, 120], [30, 60, 120, 240]],
            },
            "wider_naive": {
                "num_modules": [2, 4, 2],
                "num_branches": [2, 3, 4],
                "num_blocks": [2, 2, 2],
                "module_type": ["NAIVE", "NAIVE", "NAIVE"],
                "reduce_ratios": [1, 1, 1],
                "num_channels": [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            },
        }

        self.stages_config = self.module_configs[network_type]

        self.stem = Stem(3, 32, 32, 1)
        num_channels_pre_layer = [32]
        for stage_idx in range(3):
            num_channels = self.stages_config["num_channels"][stage_idx]
            setattr(self, 'transition{}'.format(stage_idx),
                    self._make_transition_layer(num_channels_pre_layer,
                                                num_channels, self.freeze_norm,
                                                self.norm_decay))
            stage, num_channels_pre_layer = self._make_stage(
                self.stages_config, stage_idx, num_channels, True,
                self.freeze_norm, self.norm_decay)
            setattr(self, 'stage{}'.format(stage_idx), stage)
        self.head_layer = IterativeHead(num_channels_pre_layer, 'bn',
                                        self.freeze_norm, self.norm_decay)

    def _make_transition_layer(self,
                               num_channels_pre_layer,
                               num_channels_cur_layer,
                               freeze_norm=False,
                               norm_decay=0.):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            L.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            nn.BatchNorm2D(num_channels_pre_layer[i]),
                            L.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False, ),
                            nn.BatchNorm2D(num_channels_cur_layer[i]),
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    conv_downsamples.append(
                        nn.Sequential(
                            L.Conv2d(
                                num_channels_pre_layer[-1],
                                num_channels_pre_layer[-1],
                                groups=num_channels_pre_layer[-1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False, ),
                            nn.BatchNorm2D(num_channels_pre_layer[-1]),
                            L.Conv2d(
                                num_channels_pre_layer[-1],
                                num_channels_cur_layer[i]
                                if j == i - num_branches_pre else
                                num_channels_pre_layer[-1],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False, ),
                            nn.BatchNorm2D(num_channels_cur_layer[i]
                                           if j == i - num_branches_pre else
                                           num_channels_pre_layer[-1]),
                            nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.LayerList(transition_layers)

    def _make_stage(self,
                    stages_config,
                    stage_idx,
                    in_channels,
                    multiscale_output,
                    freeze_norm=False,
                    norm_decay=0.):
        num_modules = stages_config["num_modules"][stage_idx]
        num_branches = stages_config["num_branches"][stage_idx]
        num_blocks = stages_config["num_blocks"][stage_idx]
        reduce_ratio = stages_config['reduce_ratios'][stage_idx]
        module_type = stages_config['module_type'][stage_idx]

        modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            modules.append(
                LiteHRNetModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=True,
                    freeze_norm=freeze_norm,
                    norm_decay=norm_decay))
            in_channels = modules[-1].in_channels
        return nn.Sequential(*modules), in_channels

    def forward(self, inputs):
        x = inputs['image']
        dims = x.shape
        if len(dims) == 5:
            x = paddle.reshape(x, (dims[0] * dims[1], dims[2], dims[3],
                                   dims[4]))  # [6, 3, 128, 96]

        x = self.stem(x)
        y_list = [x]
        for stage_idx in range(3):
            x_list = []
            transition = getattr(self, 'transition{}'.format(stage_idx))
            for j in range(self.stages_config["num_branches"][stage_idx]):
                if transition[j] is not None:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(stage_idx))(x_list)
        x = self.head_layer(y_list)
        res = []
        for i, layer in enumerate(x):
            if i == self.freeze_at:
                layer.stop_gradient = True
            if i in self.return_idx:
                res.append(layer)
        return res

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]
