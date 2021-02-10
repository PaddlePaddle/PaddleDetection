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
from paddle.regularizer import L2Decay
from ppdet.modeling.layers import DeformableConvV2
from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec

__all__ = ['ResNet', 'Res5Head']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 name_adapter,
                 groups=1,
                 act=None,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False,
                 name=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.act = act

        if not dcn_v2:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=paddle.ParamAttr(
                    learning_rate=lr, name=name + "_weights"),
                bias_attr=False)
        else:
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=paddle.ParamAttr(
                    learning_rate=lr, name=name + '_weights'),
                bias_attr=False,
                name=name)

        bn_name = name_adapter.fix_conv_norm_name(name)
        norm_lr = 0. if freeze_norm else lr
        param_attr = paddle.ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            name=bn_name + "_scale",
            trainable=False if freeze_norm else True)
        bias_attr = paddle.ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            name=bn_name + "_offset",
            trainable=False if freeze_norm else True)

        global_stats = True if freeze_norm else False
        if norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        else:
            self.norm = nn.BatchNorm(
                ch_out,
                act=None,
                param_attr=param_attr,
                bias_attr=bias_attr,
                use_global_stats=global_stats,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
        norm_params = self.norm.parameters()

        if freeze_norm:
            for param in norm_params:
                param.stop_gradient = True

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm_type in ['bn', 'sync_bn']:
            out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out


class BasicBlock(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 name_adapter,
                 name,
                 variant='b',
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False):
        super(BasicBlock, self).__init__()
        assert dcn_v2 is False, "Not implemented yet."
        conv_name1, conv_name2, shortcut_name = name_adapter.fix_basicblock_name(
            name)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_sublayer(
                    'pool',
                    nn.AvgPool2D(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        name_adapter=name_adapter,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr,
                        name=shortcut_name))
            else:
                self.short = ConvNormLayer(
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
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            name_adapter=name_adapter,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            name=conv_name1)

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            name_adapter=name_adapter,
            act=None,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            name=conv_name2)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)

        return out


class BottleNeck(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 name_adapter,
                 name,
                 variant='b',
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False):
        super(BottleNeck, self).__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        # ResNeXt
        if groups == 1:
            width = ch_out
        else:
            width = int(
                math.floor(ch_out * (base_width * 1.0 / base_channels)) *
                groups)

        conv_name1, conv_name2, conv_name3, \
            shortcut_name = name_adapter.fix_bottleneck_name(name)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_sublayer(
                    'pool',
                    nn.AvgPool2D(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out * 4,
                        filter_size=1,
                        stride=1,
                        name_adapter=name_adapter,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr,
                        name=shortcut_name))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out * 4,
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
            ch_out=width,
            filter_size=1,
            stride=stride1,
            name_adapter=name_adapter,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            name=conv_name1)

        self.branch2b = ConvNormLayer(
            ch_in=width,
            ch_out=width,
            filter_size=3,
            stride=stride2,
            name_adapter=name_adapter,
            groups=groups,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2,
            name=conv_name2)

        self.branch2c = ConvNormLayer(
            ch_in=width,
            ch_out=ch_out * 4,
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
        out = self.branch2b(out)
        out = self.branch2c(out)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)

        return out


class Blocks(nn.Layer):
    def __init__(self,
                 depth,
                 ch_in,
                 ch_out,
                 count,
                 name_adapter,
                 stage_num,
                 variant='b',
                 groups=1,
                 base_width=-1,
                 base_channels=-1,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False):
        super(Blocks, self).__init__()

        self.blocks = []
        for i in range(count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            if depth >= 50:
                block = self.add_sublayer(
                    conv_name,
                    BottleNeck(
                        ch_in=ch_in if i == 0 else ch_out * 4,
                        ch_out=ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        name_adapter=name_adapter,
                        name=conv_name,
                        variant=variant,
                        groups=groups,
                        base_width=base_width,
                        base_channels=base_channels,
                        lr=lr,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        dcn_v2=dcn_v2))
            else:
                ch_in = ch_in // 4 if i > 0 else ch_in
                block = self.add_sublayer(
                    conv_name,
                    BasicBlock(
                        ch_in=ch_in if i == 0 else ch_out,
                        ch_out=ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        name_adapter=name_adapter,
                        name=conv_name,
                        variant=variant,
                        lr=lr,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        dcn_v2=dcn_v2))
            self.blocks.append(block)

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out


@register
@serializable
class ResNet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=-1,
                 base_channels=-1,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 num_stages=4):
        super(ResNet, self).__init__()
        self._model_type = 'ResNet' if groups == 1 else 'ResNeXt'
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

        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = ResNet_cfg[depth]
        na = NameAdapter(self)

        conv1_name = na.fix_c1_stage_name()
        if variant in ['c', 'd']:
            conv_def = [
                [3, 32, 3, 2, "conv1_1"],
                [32, 32, 3, 1, "conv1_2"],
                [32, 64, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, 64, 7, 2, conv1_name]]
        self.conv1 = nn.Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.conv1.add_sublayer(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    name_adapter=na,
                    groups=1,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0,
                    name=_name))

        ch_in_list = [64, 256, 512, 1024]
        ch_out_list = [64, 128, 256, 512]
        self.expansion = 4 if depth >= 50 else 1

        self._out_channels = [self.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]

        self.res_layers = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            res_layer = self.add_sublayer(
                res_name,
                Blocks(
                    depth,
                    ch_in_list[i] // 4
                    if i > 0 and depth < 50 else ch_in_list[i],
                    ch_out_list[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    base_channels=base_channels,
                    lr=lr_mult,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=(i in self.dcn_v2_stages)))
            self.res_layers.append(res_layer)

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
class Res5Head(nn.Layer):
    def __init__(self, depth=50):
        super(Res5Head, self).__init__()
        feat_in, feat_out = [1024, 512]
        if depth < 50:
            feat_in = 256
        na = NameAdapter(self)
        self.res5 = Blocks(
            depth, feat_in, feat_out, count=3, name_adapter=na, stage_num=5)
        self.feat_out = feat_out if depth < 50 else feat_out * 4

    @property
    def out_shape(self):
        return [ShapeSpec(
            channels=self.feat_out,
            stride=16, )]

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y
