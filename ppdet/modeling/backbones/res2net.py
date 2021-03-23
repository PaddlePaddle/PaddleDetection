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

import math
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec
from .resnet import ConvNormLayer

__all__ = ['Res2Net', 'Res2NetC5']

Res2Net_cfg = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 12, 48, 3]
}


class BottleNeck(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 name_adapter,
                 name,
                 width,
                 scales=4,
                 groups=1,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False):
        super(BottleNeck, self).__init__()

        conv_name1, conv_name2, conv_name3, \
            shortcut_name = name_adapter.fix_bottleneck_name(name)

        self.shortcut = shortcut
        self.scales = scales
        self.stride = stride
        if not shortcut:
            self.branch1 = ConvNormLayer(
                ch_in=ch_in,
                ch_out=ch_out,
                filter_size=1,
                stride=stride,
                name_adapter=name_adapter,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                lr=lr,
                name=shortcut_name)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=width * scales,
            filter_size=1,
            stride=1,
            name_adapter=name_adapter,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            name=conv_name1)

        self.branch2b = nn.LayerList([
            ConvNormLayer(
                ch_in=width,
                ch_out=width,
                filter_size=3,
                stride=stride,
                name_adapter=name_adapter,
                groups=groups,
                act='relu',
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                lr=lr,
                dcn_v2=dcn_v2,
                name=conv_name2 + '_' + str(i + 1))
            for i in range(self.scales - 1)
        ])

        self.branch2c = ConvNormLayer(
            ch_in=width * scales,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            name_adapter=name_adapter,
            groups=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            name=conv_name3)

    def forward(self, inputs):

        out = self.branch2a(inputs)
        feature_split = paddle.split(out, self.scales, 1)
        out_split = []
        for i in range(self.scales - 1):
            if i == 0 or self.stride == 2:
                out_split.append(self.branch2b[i](feature_split[i]))
            else:
                out_split.append(self.branch2b[i](paddle.add(feature_split[i],
                                                             out_split[-1])))
        if self.stride == 1:
            out_split.append(feature_split[-1])
        else:
            out_split.append(F.avg_pool2d(feature_split[-1], 3, self.stride, 1))
        out = self.branch2c(paddle.concat(out_split, 1))
        if self.shortcut:
            short = inputs
        else:
            short = self.branch1(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)

        return out


class Blocks(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 count,
                 name_adapter,
                 stage_num,
                 width,
                 scales=4,
                 groups=1,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False):
        super(Blocks, self).__init__()

        self.blocks = nn.Sequential()
        for i in range(count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            self.blocks.add_sublayer(
                conv_name,
                BottleNeck(
                    ch_in=ch_in if i == 0 else ch_out,
                    ch_out=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    name_adapter=name_adapter,
                    name=conv_name,
                    width=width * (2**(stage_num - 2)),
                    scales=scales,
                    groups=groups,
                    lr=lr,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=dcn_v2))

    def forward(self, inputs):
        return self.blocks(inputs)


@register
@serializable
class Res2Net(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 width=26,
                 scales=4,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 num_stages=4,
                 in_channels=3):
        super(Res2Net, self).__init__()

        self._model_type = 'Res2Net' if groups == 1 else 'Res2NeXt'

        assert num_stages >= 1 and num_stages <= 4
        assert depth >= 50, "just support depth>=50 in res2net, but got depth=".format(
            depth)

        self.depth = depth
        self.variant = variant
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages
        assert len(lr_mult_list) == 4, \
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages

        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = Res2Net_cfg[depth]
        na = NameAdapter(self)

        # C1 stage
        conv1_name = na.fix_c1_stage_name()
        self.conv1 = ConvNormLayer(
            ch_in=in_channels,
            ch_out=64,
            filter_size=7,
            stride=2,
            name_adapter=na,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=1.0,
            name=conv1_name)

        self._in_channels = [64, 256, 512, 1024]
        self._out_channels = [256, 512, 1024, 2048]
        self._out_strides = [4, 8, 16, 32]

        # C2-C5 stages
        self.res_layers = nn.LayerList()
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            self.res_layers.append(
                Blocks(
                    self._in_channels[i],
                    self._out_channels[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    width=width,
                    scales=scales,
                    groups=groups,
                    lr=lr_mult,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=(i in self.dcn_v2_stages)))

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == self.freeze_at:
                x.stop_gradient = True
            if idx in self.return_idx:
                outs.append(x)
        return outs


@register
class Res2NetC5(nn.Layer):
    def __init__(self, depth=50, width=26, scales=4):
        super(Res2NetC5, self).__init__()
        feat_in, feat_out = [1024, 2048]
        na = NameAdapter(self)
        self.res5 = Blocks(
            feat_in,
            feat_out,
            count=3,
            name_adapter=na,
            stage_num=5,
            width=width,
            scales=scales)
        self.feat_out = feat_out

    @property
    def out_shape(self):
        return [ShapeSpec(
            channels=self.feat_out,
            stride=32, )]

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y
