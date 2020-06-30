import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from ppdet.core.workspace import register, serializable

__all__ = ['DarkNet', 'ConvBNLayer']


class ConvBNLayer(Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act="leaky"):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False,
            act=None)
        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.)))

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out


class DownSample(Layer):
    def __init__(self, ch_in, ch_out, filter_size=3, stride=2, padding=1):

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(Layer):
    def __init__(self, ch_in, ch_out):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in, ch_out=ch_out, filter_size=1, stride=1, padding=0)
        self.conv2 = ConvBNLayer(
            ch_in=ch_out, ch_out=ch_out * 2, filter_size=3, stride=1, padding=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = fluid.layers.elementwise_add(x=inputs, y=conv2, act=None)
        return out


class Blocks(Layer):
    def __init__(self, ch_in, ch_out, count):
        super(Blocks, self).__init__()

        self.basicblock0 = BasicBlock(ch_in, ch_out)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer("basic_block_%d" % (i),
                                        BasicBlock(ch_out * 2, ch_out))
            self.res_out_list.append(res_out)
        self.ch_out = ch_out

    def forward(self, inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


@register
@serializable
class DarkNet(Layer):
    def __init__(self, depth=53, mode='train'):
        super(DarkNet, self).__init__()
        self.depth = depth
        self.mode = mode
        self.stages = DarkNet_cfg[self.depth][0:5]

        self.conv0 = ConvBNLayer(
            ch_in=3, ch_out=32, filter_size=3, stride=1, padding=1)

        self.downsample0 = DownSample(ch_in=32, ch_out=32 * 2)

        self.darknet53_conv_block_list = []
        self.downsample_list = []
        ch_in = [64, 128, 256, 512, 1024]
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer("stage_%d" % (i),
                                           Blocks(
                                               int(ch_in[i]), 32 * (2**i),
                                               stage))
            self.darknet53_conv_block_list.append(conv_block)
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(
                    ch_in=32 * (2**(i + 1)), ch_out=32 * (2**(i + 2))))
            self.downsample_list.append(downsample)

    def forward(self, inputs):
        x = inputs['image']

        out = self.conv0(x)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        outs = {'darknet_outs': blocks[-1:-4:-1]}
        return outs
