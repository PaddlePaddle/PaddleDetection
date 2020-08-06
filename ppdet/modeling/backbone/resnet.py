import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer, Sequential
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from ppdet.core.workspace import register, serializable
from paddle.fluid.regularizer import L2Decay
from .name_adapter import NameAdapter
from numbers import Integral


class ConvNormLayer(Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 name_adapter,
                 act=None,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 name=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'affine_channel']
        self.norm_type = norm_type
        self.act = act

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=1,
            act=None,
            param_attr=ParamAttr(
                learning_rate=lr, name=name + "_weights"),
            bias_attr=False)

        bn_name = name_adapter.fix_conv_norm_name(name)
        norm_lr = 0. if freeze_norm else lr
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            name=bn_name + "_scale")
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            name=bn_name + "_offset")

        if norm_type == 'bn':
            global_stats = True if freeze_norm else False
            self.norm = BatchNorm(
                num_channels=ch_out,
                act=act,
                param_attr=param_attr,
                bias_attr=bias_attr,
                use_global_stats=global_stats,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
            norm_params = self.norm.parameters()
        elif norm_type == 'affine_channel':
            self.scale = fluid.layers.create_parameter(
                shape=[ch_out],
                dtype='float32',
                attr=param_attr,
                default_initializer=Constant(1.))

            self.offset = fluid.layers.create_parameter(
                shape=[ch_out],
                dtype='float32',
                attr=bias_attr,
                default_initializer=Constant(0.))
            norm_params = [self.scale, self.offset]

        if freeze_norm:
            for param in norm_params:
                param.stop_gradient = True

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm_type == 'bn':
            out = self.norm(out)
        elif self.norm_type == 'affine_channel':
            out = fluid.layers.affine_channel(
                out, scale=self.scale, bias=self.offset, act=self.act)
        return out


class BottleNeck(Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 name_adapter,
                 name,
                 variant='a',
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True):
        super(BottleNeck, self).__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        conv_name1, conv_name2, conv_name3, \
            shortcut_name = name_adapter.fix_bottleneck_name(name)

        self.shortcut = shortcut
        if not shortcut:
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
            ch_out=ch_out,
            filter_size=1,
            stride=stride1,
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
            stride=stride2,
            name_adapter=name_adapter,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            name=conv_name2)

        self.branch2c = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out * 4,
            filter_size=1,
            stride=1,
            name_adapter=name_adapter,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            name=conv_name3)

    def forward(self, inputs):
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        out = fluid.layers.elementwise_add(x=short, y=out, act='relu')

        return out


class Blocks(Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 count,
                 name_adapter,
                 stage_num,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True):
        super(Blocks, self).__init__()

        self.blocks = []
        for i in range(count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)

            block = self.add_sublayer(
                conv_name,
                BottleNeck(
                    ch_in=ch_in if i == 0 else ch_out * 4,
                    ch_out=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    name_adapter=name_adapter,
                    name=conv_name,
                    variant=name_adapter.variant,
                    lr=lr,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm))
            self.blocks.append(block)

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out


ResNet_cfg = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}


@register
@serializable
class ResNet(Layer):
    def __init__(self,
                 depth=50,
                 variant='a',
                 lr_mult=1.,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 num_stages=4):
        super(ResNet, self).__init__()
        self.depth = depth
        self.norm_type = norm_type
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages

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
        self.conv1 = Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.conv1.add_sublayer(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    name_adapter=na,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr_mult,
                    name=_name))

        self.pool = Pool2D(
            pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)

        ch_in_list = [64, 256, 512, 1024]
        ch_out_list = [64, 128, 256, 512]

        self.res_layers = []
        for i in range(num_stages):
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            res_layer = self.add_sublayer(
                res_name,
                Blocks(
                    ch_in_list[i],
                    ch_out_list[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    lr=lr_mult,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm))
            self.res_layers.append(res_layer)

    def forward(self, inputs):
        x = inputs['image']
        conv1 = self.conv1(x)
        x = self.pool(conv1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == self.freeze_at:
                x.stop_gradient = True
            if idx in self.return_idx:
                outs.append(x)
        return outs
