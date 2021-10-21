import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import batch_norm
from ..shape_spec import ShapeSpec

class Mish(nn.Layer):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = x * (self.tanh(self.softplus(x)))
        return x

class Conv_Bn_Activation(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False, norm_type='sync_bn', name=''):
        super().__init__()
        pad = (kernel_size - 1) // 2

        #self.conv = nn.ModuleList()
        self.conv = nn.Sequential()
        #self.op_list = []
        if bias:
            self.conv.add_sublayer(name+'.conv', nn.Conv2D(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.add_sublayer(name+'.conv', nn.Conv2D(in_channels, out_channels, kernel_size, stride, pad, bias_attr=False))
        if bn:
            self.conv.add_sublayer(name+'.bn', batch_norm(out_channels, norm_type=norm_type))
        if activation == "mish":
            self.conv.add_sublayer(name+'.act', Mish())
        elif activation == "relu":
            self.conv.add_sublayer(name+'.act', nn.ReLU())
        elif activation == "leaky":
            self.conv.add_sublayer(name+'.act', nn.LeakyReLU(0.1))
        elif activation == "linear":
            pass
        else:
            print("activate error !!!")
        #self.op_list.append(self.conv)
    def forward(self, x):
        x = self.conv(x)
        return x


class ResBlock(nn.Layer):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True, name=''):
        super().__init__()
        self.shortcut = shortcut
        #self.module_list = nn.ModuleList()
        #self.module_list = nn.Sequential()
        self.module_list = []
        for i in range(nblocks):
            # resblock_one = nn.Sequential()
            # resblock_one.add_sublayer(name+'.'+str(i)+'.0', Conv_Bn_Activation(ch, ch, 1, 1, 'mish', name=name+'.'+str(i)+'.0'))
            # resblock_one.add_sublayer(name+'.'+str(i)+'.1', Conv_Bn_Activation(ch, ch, 3, 1, 'mish', name=name+'.'+str(i)+'.1'))
            # #self.module_list.add_sublayer(name+'.'+str(i), resblock_one)
            # self.module_list.append(resblock_one)
            self.module_list.append([Conv_Bn_Activation(ch, ch, 1, 1, 'mish', name=name+'.'+str(i)+'.0'),
                                    Conv_Bn_Activation(ch, ch, 3, 1, 'mish', name=name+'.'+str(i)+'.1')])

    def forward(self, x):
        #print(self.module_list)
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish', name='conv')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish', name='stage0.downsample')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish', name='stage0.route_in.right')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish', name='stage0.neck')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish', name='stage0.0.0')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish', name='stage0.0.1')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish', name='stage0.route_in.left')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish', name='stage0.conv_layer')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = paddle.concat([x7, x3], axis=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish', name='stage1.downsample')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish', name='stage1.route_in.right')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish', name='stage1.neck')

        #self.resblock = ResBlock(ch=64, nblocks=2, name='stage1.res')
        self.res0_conv0 = Conv_Bn_Activation(64, 64, 1, 1, 'mish', name='stage1.res.0.0')
        self.res0_conv1 = Conv_Bn_Activation(64, 64, 3, 1, 'mish', name='stage1.res.0.1')
        self.res1_conv0 = Conv_Bn_Activation(64, 64, 1, 1, 'mish', name='stage1.res.1.0')
        self.res1_conv1 = Conv_Bn_Activation(64, 64, 3, 1, 'mish', name='stage1.res.1.1')

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish', name='stage1.route_in.left')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage1.conv_layer')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        #r = self.resblock(x3)
        r = x3
        r = self.res0_conv0(r)
        r = self.res0_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res1_conv0(r)
        r = self.res1_conv1(r)
        x3 = x3 + r

        x4 = self.conv4(x3)

        x4 = paddle.concat([x4, x2], axis=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish', name='stage2.downsample')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish', name='stage2.route_in.right')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish', name='stage2.neck')

        #self.resblock = ResBlock(ch=128, nblocks=8, name='stage2.res')
        self.res0_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.0.0')
        self.res0_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.0.1')
        self.res1_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.1.0')
        self.res1_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.1.1')
        self.res2_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.2.0')
        self.res2_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.2.1')
        self.res3_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.3.0')
        self.res3_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.3.1')
        self.res4_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.4.0')
        self.res4_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.4.1')
        self.res5_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.5.0')
        self.res5_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.5.1')
        self.res6_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.6.0')
        self.res6_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.6.1')
        self.res7_conv0 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.res.7.0')
        self.res7_conv1 = Conv_Bn_Activation(128, 128, 3, 1, 'mish', name='stage2.res.7.1')

        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish', name='stage2.route_in.left')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage2.conv_layer')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        #r = self.resblock(x3)
        r = x3
        r = self.res0_conv0(r)
        r = self.res0_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res1_conv0(r)
        r = self.res1_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res2_conv0(r)
        r = self.res2_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res3_conv0(r)
        r = self.res3_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res4_conv0(r)
        r = self.res4_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res5_conv0(r)
        r = self.res5_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res6_conv0(r)
        r = self.res6_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res7_conv0(r)
        r = self.res7_conv1(r)
        x3 = x3 + r

        x4 = self.conv4(x3)

        x4 = paddle.concat([x4, x2], axis=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish', name='stage3.downsample')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish', name='stage3.route_in.right')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish', name='stage3.neck')

        #self.resblock = ResBlock(ch=256, nblocks=8, name='stage3.res')
        self.res0_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.0.0')
        self.res0_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.0.1')
        self.res1_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.1.0')
        self.res1_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.1.1')
        self.res2_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.2.0')
        self.res2_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.2.1')
        self.res3_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.3.0')
        self.res3_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.3.1')
        self.res4_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.4.0')
        self.res4_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.4.1')
        self.res5_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.5.0')
        self.res5_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.5.1')
        self.res6_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.6.0')
        self.res6_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.6.1')
        self.res7_conv0 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.res.7.0')
        self.res7_conv1 = Conv_Bn_Activation(256, 256, 3, 1, 'mish', name='stage3.res.7.1')

        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish', name='stage3.route_in.left')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish', name='stage3.conv_layer')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        #r = self.resblock(x3)
        r = x3
        r = self.res0_conv0(r)
        r = self.res0_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res1_conv0(r)
        r = self.res1_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res2_conv0(r)
        r = self.res2_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res3_conv0(r)
        r = self.res3_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res4_conv0(r)
        r = self.res4_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res5_conv0(r)
        r = self.res5_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res6_conv0(r)
        r = self.res6_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res7_conv0(r)
        r = self.res7_conv1(r)
        x3 = x3 + r

        x4 = self.conv4(x3)

        x4 = paddle.concat([x4, x2], axis=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish', name='stage4.downsample')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish', name='stage4.route_in.right')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish', name='stage4.neck')

        #self.resblock = ResBlock(ch=512, nblocks=4, name='stage4.res')
        self.res0_conv0 = Conv_Bn_Activation(512, 512, 1, 1, 'mish', name='stage4.res.0.0')
        self.res0_conv1 = Conv_Bn_Activation(512, 512, 3, 1, 'mish', name='stage4.res.0.1')
        self.res1_conv0 = Conv_Bn_Activation(512, 512, 1, 1, 'mish', name='stage4.res.1.0')
        self.res1_conv1 = Conv_Bn_Activation(512, 512, 3, 1, 'mish', name='stage4.res.1.1')
        self.res2_conv0 = Conv_Bn_Activation(512, 512, 1, 1, 'mish', name='stage4.res.2.0')
        self.res2_conv1 = Conv_Bn_Activation(512, 512, 3, 1, 'mish', name='stage4.res.2.1')
        self.res3_conv0 = Conv_Bn_Activation(512, 512, 1, 1, 'mish', name='stage4.res.3.0')
        self.res3_conv1 = Conv_Bn_Activation(512, 512, 3, 1, 'mish', name='stage4.res.3.1')

        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish', name='stage4.route_in.left')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish', name='stage4.conv_layer')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        #r = self.resblock(x3)
        r = x3
        r = self.res0_conv0(r)
        r = self.res0_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res1_conv0(r)
        r = self.res1_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res2_conv0(r)
        r = self.res2_conv1(r)
        x3 = x3 + r
        r = x3
        r = self.res3_conv0(r)
        r = self.res3_conv1(r)
        x3 = x3 + r

        x4 = self.conv4(x3)

        x4 = paddle.concat([x4, x2], axis=1)
        x5 = self.conv5(x4)
        return x5


@register
@serializable
class CSPDarkNet(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self, norm_type='sync_bn', data_format='NCHW'):
        super().__init__()
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
    
    def forward(self, inputs):
        x = inputs['image']
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        return [d3, d4, d5]
    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in [256, 512, 1024]]