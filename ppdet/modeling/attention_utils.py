# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant
from ppdet.modeling.ops import get_act_fn

__all__ = [
    'GAMAttention',
    'CBAM',
    'CoordAtt',
    'NAMAttention',
    'SEAttention',
    'ShuffleAttention',
    'SimAM',
    'SKAttention',
]


def channel_shuffle(x, groups=2):
    # channel_shuffle of GAMAttention
    B, C, H, W = x.shape[:]
    out = x.reshape([B, groups, C // groups, H, W]).transpose([0, 2, 1, 3, 4])
    out = out.reshape([B, C, H, W])
    return out


class GAMAttention(nn.Layer):
    """
    Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions
    """

    def __init__(self, c1, group=True, rate=4):
        super(GAMAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(), nn.Linear(int(c1 / rate), c1))
        self.spatial_attention = nn.Sequential(
            nn.Conv2D(
                c1, c1 // rate, kernel_size=7, padding=3, groups=rate)
            if group else nn.Conv2D(
                c1, int(c1 / rate), kernel_size=7, padding=3),
            nn.BatchNorm2D(int(c1 / rate)),
            nn.ReLU(),
            nn.Conv2D(
                c1 // rate, c1, kernel_size=7, padding=3, groups=rate)
            if group else nn.Conv2D(
                int(c1 / rate), c1, kernel_size=7, padding=3),
            nn.BatchNorm2D(c1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_transpose = x.transpose([0, 2, 3, 1]).reshape([b, -1, c])
        x_att_transpose = self.channel_attention(x_transpose).reshape(
            [b, h, w, c])
        x_channel_att = x_att_transpose.transpose([0, 3, 1, 2])
        x = x * x_channel_att

        x_spatial_att = F.sigmoid(self.spatial_attention(x))
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  #last shuffle 
        out = x * x_spatial_att
        return out


class CAM(nn.Layer):
    """Channel Attention Module of CBAM"""

    def __init__(self, in_channels, ratio=16):
        super(CAM, self).__init__()
        mid_channels = in_channels // ratio
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveAvgPool2D(1)
        self.shared_MLP = nn.Sequential(
            #nn.Linear(in_channels, mid_channels),
            nn.Conv2D(
                in_channels, mid_channels, 1, bias_attr=False),
            nn.ReLU(),
            #nn.Linear(mid_channels, in_channels),
            nn.Conv2D(
                mid_channels, in_channels, 1, bias_attr=False), )

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return F.sigmoid(avg_out + max_out)


class SAM(nn.Layer):
    """Spatial Attention Module of CBAM"""

    def __init__(self):
        super(SAM, self).__init__()
        self.conv2d = nn.Conv2D(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        out = paddle.concat([avg_out, max_out], axis=1)
        out = F.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Layer):
    """CBAM: Convolutional Block Attention Module"""

    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = CAM(in_channels)
        self.spatial_attention = SAM()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class CoordAtt(nn.Layer):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2D(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.conv_h = nn.Conv2D(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2D(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        h, w = x.shape[2:]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose([0, 1, 3, 2])

        y = paddle.concat([x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.hardswish(y)

        x_h, x_w = paddle.split(y, [h, w], axis=2)
        x_w = x_w.transpose([0, 1, 3, 2])

        a_h = F.sigmoid(self.conv_h(x_h))
        a_w = F.sigmoid(self.conv_w(x_w))
        out = identity * a_w * a_h
        return out


class Channel_Att(nn.Layer):
    # Channel_Att of NAMAttention
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.bn = nn.BatchNorm2D(channels)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        weight_bn = self.bn.weight.abs() / paddle.sum(self.bn.weight.abs())
        x = x.transpose([0, 2, 3, 1])
        x = paddle.multiply(weight_bn, x)
        x = x.transpose([0, 3, 1, 2])
        x = F.sigmoid(x) * residual
        return x


class NAMAttention(nn.Layer):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.channel_att = Channel_Att(channels)

    def forward(self, x):
        return self.channel_att(x)


class SEAttention(nn.Layer):
    # see https://arxiv.org/abs/1709.01507
    def __init__(self, ch, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(ch)
        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(
            ch,
            c_,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(
            c_,
            ch,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

    def forward(self, inputs):
        out = self.pool(inputs)
        out = paddle.squeeze(out, axis=[2, 3])
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = F.sigmoid(out)
        out = paddle.unsqueeze(out, axis=[2, 3])
        scale = out * inputs
        return scale


class ShuffleAttention(nn.Layer):
    # see https://arxiv.org/abs/2102.00240
    def __init__(self, channel, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = self.create_parameter(
            shape=[1, channel // (2 * G), 1, 1],
            default_initializer=Constant(0.))
        self.cbias = self.create_parameter(
            shape=[1, channel // (2 * G), 1, 1],
            default_initializer=Constant(1.))
        self.sweight = self.create_parameter(
            shape=[1, channel // (2 * G), 1, 1],
            default_initializer=Constant(0.))
        self.sbias = self.create_parameter(
            shape=[1, channel // (2 * G), 1, 1],
            default_initializer=Constant(1.))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2D):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape([b, groups, -1, h, w])
        x = x.transpose([0, 2, 1, 3, 4])
        x = x.reshape([b, -1, h, w])
        return x

    def forward(self, x):
        b, c, h, w = x.shape[:]
        x = x.reshape([b * self.G, -1, h, w])  #bs*G,c//G,h,w

        #channel_split
        x_0, x_1 = x.chunk(2, axis=1)  #bs*G,c//(2*G),h,w

        #channel attention
        x_channel = self.avg_pool(x_0)  #bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  #bs*G,c//(2*G),1,1
        x_channel = x_0 * F.sigmoid(x_channel)

        #spatial attention
        x_spatial = self.gn(x_1)  #bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  #bs*G,c//(2*G),h,w
        x_spatial = x_1 * F.sigmoid(x_spatial)  #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = paddle.concat([x_channel, x_spatial], 1)  #bs*G,c//G,h,w
        out = out.reshape([b, -1, h, w])

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class SimAM(nn.Layer):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape[:]
        n = w * h - 1
        x_minus_mu_square = (x - x.mean((2, 3), keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(
            (2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * F.sigmoid(y)


class SKAttention(nn.Layer):
    def __init__(self,
                 channel,
                 kernels=[1, 3, 5, 7],
                 reduction=16,
                 group=1,
                 L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.LayerList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2D(
                        channel,
                        channel,
                        kernel_size=k,
                        padding=k // 2,
                        groups=group),
                    nn.BatchNorm2D(channel),
                    nn.ReLU()))
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.LayerList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        bs, c, _, _ = x.shape[:]
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = paddle.stack(conv_outs, 0)

        U = sum(conv_outs)

        S = U.mean(-1).mean(-1)
        Z = self.fc(S)

        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.reshape([bs, c, 1, 1]))
        attention_weughts = paddle.stack(weights, 0)
        attention_weughts = self.softmax(attention_weughts)

        V = (attention_weughts * feats).sum(0)
        return V
