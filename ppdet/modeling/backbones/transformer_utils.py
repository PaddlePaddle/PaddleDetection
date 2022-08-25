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
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant
from ppdet.modeling.ops import get_act_fn

from paddle.nn.initializer import TruncatedNormal, Constant, Assign

__all__ = ['TransformerBlock', 'BoTBlock', 'CoTBlock']

# Common initializations
ones_ = Constant(value=1.)
zeros_ = Constant(value=0.)
trunc_normal_ = TruncatedNormal(std=.02)


# Common Layers
def drop_path(x, drop_prob=0., training=False):
    """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# common funcs
def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return tuple([x] * 2)


def add_parameter(layer, datas, name=None):
    parameter = layer.create_parameter(
        shape=(datas.shape), default_initializer=Assign(datas))
    if name:
        layer.add_parameter(name, parameter)
    return parameter


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class TransformerLayer(nn.Layer):
    def __init__(self, c, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias_attr=False)
        self.k = nn.Linear(c, c, bias_attr=False)
        self.v = nn.Linear(c, c, bias_attr=False)
        self.ma = nn.MultiHeadAttention(embed_dim=c, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4 * c, bias_attr=False)
        self.fc2 = nn.Linear(4 * c, c, bias_attr=False)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x_ = self.ln1(x)
        x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))[0]) + x
        x_ = self.ln2(x)
        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
        x = x + self.dropout(x_)
        return x


class TransformerBlock(nn.Layer):
    # Vision Transformer
    def __init__(self, c1, c2, num_heads=4, n=1):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = nn.Conv2D(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads)
                                  for _ in range(n)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape[:]
        p = x.flatten(2).unsqueeze(0).transpose([3, 1, 2, 0]).squeeze(3)
        out = self.tr(p + self.linear(p))
        out = out.unsqueeze(3).transpose([3, 1, 2, 0]).reshape(
            [b, self.c2, w, h])
        return out


class BoTBlock(nn.Layer):
    # BoT: Bottleneck Transformer Block
    def __init__(self, c1, c2, shortcut=False):
        super(BoTBlock, self).__init__()
        self.tr = BottleneckTransformer(c1, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.tr(x) if self.add else self.tr(x)


class BottleneckTransformer(nn.Layer):
    # BoT: BoTNet(Bottleneck Transformers), see https://arxiv.org/abs/2101.11605
    def __init__(self,
                 c1,
                 c2,
                 stride=1,
                 heads=4,
                 mhsa=True,
                 resolution=(20, 20),
                 expansion=1):
        super(BottleneckTransformer, self).__init__()
        c_ = int(c2 * expansion)
        self.cv1 = ConvBNLayer(c1, c_, 1, 1)
        if not mhsa:
            self.cv2 = ConvBNLayer(c_, c2, 3, 1)
        else:
            self.cv2 = nn.LayerList()
            self.cv2.append(
                MHSA(
                    c2,
                    width=int(resolution[0]),
                    height=int(resolution[1]),
                    heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2D(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        self.shortcut = c1 == c2
        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2D(
                    c1, expansion * c2, kernel_size=1, stride=stride),
                nn.BatchNorm2D(expansion * c2))
        self.fc1 = nn.Linear(c2, c2)

    def forward(self, x):
        out = x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(
            self.cv1(x))
        return out


class MHSA(nn.Layer):
    def __init__(self, n_dims, width=20, height=20, heads=4, pos_emb=False):
        super(MHSA, self).__init__()
        self.heads = heads
        self.query = nn.Conv2D(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2D(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2D(n_dims, n_dims, kernel_size=1)
        self.pos_emb = pos_emb
        if self.pos_emb:
            self.rel_h_weight = self.create_parameter(
                shape=([1, heads, (n_dims) // heads, 1, int(height)]))
            self.rel_w_weight = self.create_parameter(
                shape=([1, heads, (n_dims) // heads, int(width), 1]))
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        B, C, W, H = x.shape[:]
        q = self.query(x).reshape([B, self.heads, C // self.heads, -1])
        k = self.key(x).reshape([B, self.heads, C // self.heads, -1])
        v = self.value(x).reshape([B, self.heads, C // self.heads, -1])
        content_content = paddle.matmul(q.transpose([0, 1, 3, 2]), k)
        _, _, c3, _ = content_content.shape[:]
        if self.pos_emb:
            content_position = (self.rel_h_weight + self.rel_w_weight).reshape(
                [1, self.heads, C // self.heads, -1]).transpose([0, 1, 3, 2])
            content_position = paddle.matmul(content_position, q)
            # TODO: in case inputsize > 640x640, here c3 > 20x20
            content_position = content_position if (
                content_content.shape == content_position.shape
            ) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = paddle.matmul(v, attention.transpose([0, 1, 3, 2]))
        out = out.reshape([B, C, W, H])
        return out


class CoTBlock(nn.Layer):
    # CoT: Contextual Transformer Block
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.tr = ContextualTransformer(c2, kernel_size=3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.tr(x) if self.add else self.tr(x)


class ContextualTransformer(nn.Layer):
    # Contextual Transformer Networks, see https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        factor = 4

        self.key_embed = nn.Sequential(
            nn.Conv2D(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=4,
                bias_attr=False),
            nn.BatchNorm2D(dim),
            nn.ReLU())
        self.value_embed = nn.Sequential(
            nn.Conv2D(
                dim, dim, 1, bias_attr=False), nn.BatchNorm2D(dim))

        self.attention_embed = nn.Sequential(
            nn.Conv2D(
                2 * dim, 2 * dim // factor, 1, bias_attr=False),
            nn.BatchNorm2D(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2D(2 * dim // factor, kernel_size * kernel_size * dim, 1))

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).reshape([bs, c, -1])
        y = paddle.concat([k1, x], 1)
        att = self.attention_embed(y)
        att = att.reshape([bs, c, self.kernel_size * self.kernel_size, h, w])
        att = att.mean(2, keepdim=False).reshape([bs, c, -1])
        k2 = F.softmax(att, -1) * v
        k2 = k2.reshape([bs, c, h, w])
        return k1 + k2
