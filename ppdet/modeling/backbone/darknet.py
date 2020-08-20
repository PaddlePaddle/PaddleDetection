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
                 act="leaky",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(name=name + '.conv.weights'),
            bias_attr=False,
            act=None)
        bn_name = name + '.bn'
        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            param_attr=ParamAttr(
                name=bn_name + '.scale', regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                name=bn_name + '.offset', regularizer=L2Decay(0.)),
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out


class DownSample(Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=2,
                 padding=1,
                 name=None):

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            name=name)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(Layer):
    def __init__(self, ch_in, ch_out, name=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            name=name + '.0')
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            name=name + '.1')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = fluid.layers.elementwise_add(x=inputs, y=conv2, act=None)
        return out


class Blocks(Layer):
    def __init__(self, ch_in, ch_out, count, name=None):
        super(Blocks, self).__init__()

        self.basicblock0 = BasicBlock(ch_in, ch_out, name=name + '.0')
        self.res_out_list = []
        for i in range(1, count):
            block_name = '{}.{}'.format(name, i)
            res_out = self.add_sublayer(
                block_name, BasicBlock(
                    ch_out * 2, ch_out, name=block_name))
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
    def __init__(self,
                 depth=53,
                 freeze_at=-1,
                 return_idx=[2, 3, 4],
                 num_stages=5):
        super(DarkNet, self).__init__()
        self.depth = depth
        self.freeze_at = freeze_at
        self.return_idx = return_idx
        self.num_stages = num_stages
        self.stages = DarkNet_cfg[self.depth][0:num_stages]

        self.conv0 = ConvBNLayer(
            ch_in=3,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            name='yolo_input')

        self.downsample0 = DownSample(
            ch_in=32, ch_out=32 * 2, name='yolo_input.downsample')

        self.darknet_conv_block_list = []
        self.downsample_list = []
        ch_in = [64, 128, 256, 512, 1024]
        for i, stage in enumerate(self.stages):
            name = 'stage.{}'.format(i)
            conv_block = self.add_sublayer(
                name, Blocks(
                    int(ch_in[i]), 32 * (2**i), stage, name=name))
            self.darknet_conv_block_list.append(conv_block)
        for i in range(num_stages - 1):
            down_name = 'stage.{}.downsample'.format(i)
            downsample = self.add_sublayer(
                down_name,
                DownSample(
                    ch_in=32 * (2**(i + 1)),
                    ch_out=32 * (2**(i + 2)),
                    name=down_name))
            self.downsample_list.append(downsample)

    def forward(self, inputs):
        x = inputs['image']

        out = self.conv0(x)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet_conv_block_list):
            out = conv_block_i(out)
            if i == self.freeze_at:
                out.stop_gradient = True
            if i in self.return_idx:
                blocks.append(out)
            if i < self.num_stages - 1:
                out = self.downsample_list[i](out)
        return blocks
