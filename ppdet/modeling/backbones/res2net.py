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

from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
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
                 width,
                 scales=4,
                 variant='b',
                 groups=1,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False):
        super(BottleNeck, self).__init__()

        self.shortcut = shortcut
        self.scales = scales
        self.stride = stride
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.branch1 = nn.Sequential()
                self.branch1.add_sublayer(
                    'pool',
                    nn.AvgPool2D(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.branch1.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.branch1 = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=width * scales,
            filter_size=1,
            stride=stride if variant == 'a' else 1,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = nn.LayerList([
            ConvNormLayer(
                ch_in=width,
                ch_out=width,
                filter_size=3,
                stride=1 if variant == 'a' else stride,
                groups=groups,
                act='relu',
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                lr=lr,
                dcn_v2=dcn_v2) for _ in range(self.scales - 1)
        ])

        self.branch2c = ConvNormLayer(
            ch_in=width * scales,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            groups=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

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

        out = paddle.add(out, short)
        out = F.relu(out)

        return out


class Blocks(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 count,
                 stage_num,
                 width,
                 scales=4,
                 variant='b',
                 groups=1,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False):
        super(Blocks, self).__init__()

        self.blocks = nn.Sequential()
        for i in range(count):
            self.blocks.add_sublayer(
                str(i),
                BottleNeck(
                    ch_in=ch_in if i == 0 else ch_out,
                    ch_out=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    width=width * (2**(stage_num - 2)),
                    scales=scales,
                    variant=variant,
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
    """
    Res2Net, see https://arxiv.org/abs/1904.01169
    Args:
        depth (int): Res2Net depth, should be 50, 101, 152, 200.
        width (int): Res2Net width
        scales (int): Res2Net scale
        variant (str): Res2Net variant, supports 'a', 'b', 'c', 'd' currently
        lr_mult_list (list): learning rate ratio of different resnet stages(2,3,4,5),
                             lower learning rate ratio is need for pretrained model
                             got using distillation(default as [1.0, 1.0, 1.0, 1.0]).
        groups (int): The groups number of the Conv Layer.
        norm_type (str): normalization type, 'bn' or 'sync_bn'
        norm_decay (float): weight decay for normalization layer weights
        freeze_norm (bool): freeze normalization layers
        freeze_at (int): freeze the backbone at which stage
        return_idx (list): index of stages whose feature maps are returned,
                           index 0 stands for res2
        dcn_v2_stages (list): index of stages who select deformable conv v2
        num_stages (int): number of stages created

    """
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 width=26,
                 scales=4,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 num_stages=4):
        super(Res2Net, self).__init__()

        self._model_type = 'Res2Net' if groups == 1 else 'Res2NeXt'

        assert depth in [50, 101, 152, 200], \
            "depth {} not in [50, 101, 152, 200]"
        assert variant in ['a', 'b', 'c', 'd'], "invalid Res2Net variant"
        assert num_stages >= 1 and num_stages <= 4

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
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = Res2Net_cfg[depth]

        # C1 stage
        if self.variant in ['c', 'd']:
            conv_def = [
                [3, 32, 3, 2, "conv1_1"],
                [32, 32, 3, 1, "conv1_2"],
                [32, 64, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, 64, 7, 2, "conv1"]]
        self.res1 = nn.Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.res1.add_sublayer(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    groups=1,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0))

        self._in_channels = [64, 256, 512, 1024]
        self._out_channels = [256, 512, 1024, 2048]
        self._out_strides = [4, 8, 16, 32]

        # C2-C5 stages
        self.res_layers = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            self.res_layers.append(
                self.add_sublayer(
                    "res{}".format(stage_num),
                    Blocks(
                        self._in_channels[i],
                        self._out_channels[i],
                        count=block_nums[i],
                        stage_num=stage_num,
                        width=width,
                        scales=scales,
                        groups=groups,
                        lr=lr_mult,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        dcn_v2=(i in self.dcn_v2_stages))))

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        res1 = self.res1(x)
        x = F.max_pool2d(res1, kernel_size=3, stride=2, padding=1)
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
    def __init__(self, depth=50, width=26, scales=4, variant='b'):
        super(Res2NetC5, self).__init__()
        feat_in, feat_out = [1024, 2048]
        self.res5 = Blocks(
            feat_in,
            feat_out,
            count=3,
            stage_num=5,
            width=width,
            scales=scales,
            variant=variant)
        self.feat_out = feat_out

    @property
    def out_shape(self):
        return [ShapeSpec(
            channels=self.feat_out,
            stride=32, )]

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y
