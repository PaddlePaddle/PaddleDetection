import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from ..backbones.CSPDarkNet import Conv_Bn_Activation


class Upsample(nn.Layer):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        
        # _, _, tH, tW = target_size

        return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

class Neck(nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky', name='neck1')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky', name='neck2')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky', name='neck3')
        # SPP
        self.maxpool1 = nn.MaxPool2D(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2D(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2D(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky', name='neck4')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky', name='neck5')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky', name='neck6')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='neck7')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='neck8')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='neck9')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky', name='neck10')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='neck11')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky', name='neck12')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='neck13')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky', name='neck14')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky', name='neck15')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky', name='neck16')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky', name='neck17')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky', name='neck18')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky', name='neck19')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky', name='neck20')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = paddle.concat([m3, m2, m1, x3], axis=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.shape)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = paddle.concat([x8, up], axis=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.shape)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = paddle.concat([x15, up], axis=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6

class Yolov4Head(nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky', name='head1')

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky', name='head3')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='head4')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky', name='head5')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='head6')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky', name='head7')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky', name='head8')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky', name='head9')

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky', name='head10')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky', name='head12')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky', name='head13')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky', name='head14')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky', name='head15')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky', name='head16')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky', name='head17')

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = paddle.concat([x3, input2], axis=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = paddle.concat([x11, input3], axis=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        
        #return [x1, x9, x17]
        return [x17, x9, x1]


@register
@serializable
class YOLOv4_Neck(nn.Layer):
    def __init__(self, norm_type='sync_bn', data_format='NCHW'):
        super(YOLOv4_Neck, self).__init__()
        self.neck = Neck()
        self.head = Yolov4Head()
    
    def forward(self, inputs, for_mot=False):
        d3, d4, d5 = inputs
        x20, x13, x6 = self.neck(d5, d4, d3)
        output = self.head(x20, x13, x6)
        return output

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in [1024, 512, 256]]