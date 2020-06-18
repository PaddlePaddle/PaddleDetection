import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

from ppdet.core.workspace import register, serializable


class ConvBNLayer(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='relu',
                 learning_rate=1.0):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=1,
            act=act,
            param_attr=ParamAttr(
                name=name_scope + "_weights", learning_rate=learning_rate),
            bias_attr=ParamAttr(name=name_scope + "_bias"))

        if name_scope == "conv1":
            bn_name = "bn_" + name_scope
        else:
            bn_name = "bn" + name_scope[3:]

        self._bn = BatchNorm(
            num_channels=ch_out,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            is_test=True)

    def forward(self, inputs):
        x = self._conv(inputs)
        out = self._bn(x)
        return out


class ConvAffineLayer(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 learning_rate=1.0,
                 act='relu'):
        super(ConvAffineLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(
                name=name_scope + "_weights", learning_rate=learning_rate),
            bias_attr=False)

        if name_scope == "conv1":
            bn_name = "bn_" + name_scope
        else:
            bn_name = "bn" + name_scope[3:]
        self.name_scope = name_scope

        self.scale = fluid.Layer.create_parameter(
            shape=[ch_out],
            dtype='float32',
            attr=ParamAttr(
                name=bn_name + '_scale', learning_rate=0.),
            default_initializer=Constant(1.))
        self.bias = fluid.layers.create_parameter(
            shape=[ch_out],
            dtype='float32',
            attr=ParamAttr(
                bn_name + '_offset', learning_rate=0.),
            default_initializer=Constant(0.))

        self.act = act

    def forward(self, inputs):
        conv = self._conv(inputs)
        out = fluid.layers.affine_channel(
            x=conv, scale=self.scale, bias=self.bias)
        if self.act == 'relu':
            out = fluid.layers.relu(x=out)
        return out


class BottleNeck(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut=True,
                 learning_rate=1.0):
        super(BottleNeck, self).__init__()

        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvBNLayer(
                name_scope + "_branch1",
                ch_in=ch_in,
                ch_out=ch_out * 4,
                filter_size=1,
                stride=stride,
                padding=0,
                act=None,
                learning_rate=learning_rate)

        self.conv1 = ConvBNLayer(
            name_scope + "_branch2a",
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=stride,
            padding=0,
            learning_rate=learning_rate, )

        self.conv2 = ConvBNLayer(
            name_scope + "_branch2b",
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1,
            learning_rate=learning_rate)

        self.conv3 = ConvBNLayer(
            name_scope + "_branch2c",
            ch_in=ch_out,
            ch_out=ch_out * 4,
            filter_size=1,
            stride=1,
            padding=0,
            learning_rate=learning_rate,
            act=None)
        self.name_scope = name_scope

    def forward(self, inputs):
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        out = fluid.layers.elementwise_add(
            x=short,
            y=conv3,
            act='relu',
            name=self.name_scope + ".add.output.5")

        return out


class Blocks(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 count,
                 stride,
                 learning_rate=1.0):
        super(Blocks, self).__init__()

        self.blocks = []
        for i in range(count):
            if i == 0:
                name = name_scope + "a"
                self.stride = stride
                self.shortcut = False
            else:
                name = name_scope + chr(ord("a") + i)
                self.stride = 1
                self.shortcut = True

            block = self.add_sublayer(
                name,
                BottleNeck(
                    name,
                    ch_in=ch_in if i == 0 else ch_out * 4,
                    ch_out=ch_out,
                    stride=self.stride,
                    shortcut=self.shortcut,
                    learning_rate=learning_rate))
            self.blocks.append(block)
            shortcut = True

    def forward(self, inputs):
        res_out = self.blocks[0](inputs)
        for block in self.blocks[1:]:
            res_out = block(res_out)
        return res_out


@register
@serializable
class ResNet(Layer):
    def __init__(
            self,
            norm_type='bn',
            depth=50,
            feature_maps=4,
            freeze_at=2, ):
        super(ResNet, self).__init__()

        if depth == 50:
            blocks = [3, 4, 6, 3]
        elif depth == 101:
            blocks = [3, 4, 23, 3]
        elif depth == 152:
            blocks = [3, 8, 36, 3]

        self.conv = ConvBNLayer(
            "conv1",
            ch_in=3,
            ch_out=64,
            filter_size=7,
            stride=2,
            padding=3,
            learning_rate=0.)

        self.pool2d_max = Pool2D(
            pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)

        self.stage2 = Blocks(
            "res2",
            ch_in=64,
            ch_out=64,
            count=blocks[0],
            stride=1,
            learning_rate=0.)

        self.stage3 = Blocks(
            "res3", ch_in=256, ch_out=128, count=blocks[1], stride=2)

        self.stage4 = Blocks(
            "res4", ch_in=512, ch_out=256, count=blocks[2], stride=2)

    def forward(self, inputs):
        x = inputs['image']

        conv1 = self.conv(x)
        poo1 = self.pool2d_max(conv1)

        res2 = self.stage2(poo1)
        res2.stop_gradient = True

        res3 = self.stage3(res2)

        res4 = self.stage4(res3)

        outs = {'res2': res2, 'res3': res3, 'res4': res4}
        return outs
