# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from ppdet.core.workspace import register
from ..shape_spec import ShapeSpec

__all__ = ['HarDNet']


def ConvLayer(in_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              bias_attr=False):
    layer = nn.Sequential(
        ('conv', nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=1,
            bias_attr=bias_attr)), ('norm', nn.BatchNorm2D(out_channels)),
        ('relu', nn.ReLU6()))
    return layer


def DWConvLayer(in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                bias_attr=False):
    layer = nn.Sequential(
        ('dwconv', nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=out_channels,
            bias_attr=bias_attr)), ('norm', nn.BatchNorm2D(out_channels)))
    return layer


def CombConvLayer(in_channels, out_channels, kernel_size=1, stride=1):
    layer = nn.Sequential(
        ('layer1', ConvLayer(
            in_channels, out_channels, kernel_size=kernel_size)),
        ('layer2', DWConvLayer(
            out_channels, out_channels, stride=stride)))
    return layer


class HarDBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 growth_rate,
                 grmul,
                 n_layers,
                 keepBase=False,
                 residual_out=False,
                 dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate,
                                              grmul)
            self.links.append(link)
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.LayerList(layers_)

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate

        link = []
        for i in range(10):
            dv = 2**i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul

        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch

        return out_channels, in_channels, link

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = paddle.concat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = paddle.concat(out_, 1)

        return out


@register
class HarDNet(nn.Layer):
    def __init__(self,
                 depth_wise=False,
                 arch=85):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        #drop_rate = 0.1

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            #drop_rate = 0.2

        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            #drop_rate = 0.05

        self._out_channels = ch_list

        blks = len(n_layers)
        self.base = nn.LayerList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            ConvLayer(
                in_channels=3,
                out_channels=first_ch[0],
                kernel_size=3,
                stride=2,
                bias_attr=False))

        # Second Layer
        self.base.append(
            ConvLayer(
                first_ch[0], first_ch[1], kernel_size=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.out_channels
            self.base.append(blk)

            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel_size=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2D(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))
        '''
        ch = ch_list[blks - 1]
        layers = []
        if with_pool:
            layers.append(nn.AdaptiveAvgPool2D((1, 1)))
        if class_num > 0:
            layers.append(nn.Flatten())
            layers.append(nn.Dropout(drop_rate))
            layers.append(nn.Linear(ch, class_num))
        self.base.append(nn.Sequential(*layers))
        '''

    def forward(self, inputs):
        x = inputs['image']
        for layer in self.base:
            x = layer(x)
        return x

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self._out_channels[-1])]
