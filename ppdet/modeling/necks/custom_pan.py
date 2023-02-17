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
import math
import copy
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import DropBlock, MultiHeadAttention
from ppdet.modeling.ops import get_act_fn
from ..backbones.cspresnet import ConvBNLayer, BasicBlock
from ..shape_spec import ShapeSpec
from ..initializer import linear_init_

__all__ = ['CustomCSPPAN']


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class SPP(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 data_format='NCHW'):
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            pool = self.add_sublayer(
                'pool{}'.format(i),
                nn.MaxPool2D(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    data_format=data_format,
                    ceil_mode=False))
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == 'NCHW':
            y = paddle.concat(outs, axis=1)
        else:
            y = paddle.concat(outs, axis=-1)

        y = self.conv(y)
        return y


class CSPStage(nn.Layer):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 act='swish',
                 spp=False,
                 use_alpha=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_sublayer(
                str(i),
                eval(block_fn)(next_ch_in,
                               ch_mid,
                               act=act,
                               shortcut=False,
                               use_alpha=use_alpha))
            if i == (n - 1) // 2 and spp:
                self.convs.add_sublayer(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = paddle.concat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
@serializable
class CustomCSPPAN(nn.Layer):
    __shared__ = [
        'norm_type', 'data_format', 'width_mult', 'depth_mult', 'trt',
        'eval_size'
    ]

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='leaky',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 data_format='NCHW',
                 width_mult=1.0,
                 depth_mult=1.0,
                 use_alpha=False,
                 trt=False,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='gelu',
                 nhead=4,
                 num_layers=4,
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 use_trans=False,
                 eval_size=None):

        super(CustomCSPPAN, self).__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.num_blocks = len(in_channels)
        self.data_format = data_format
        self._out_channels = out_channels

        self.hidden_dim = in_channels[-1]
        in_channels = in_channels[::-1]

        self.use_trans = use_trans
        self.eval_size = eval_size
        if use_trans:
            if eval_size is not None:
                self.pos_embed = self.build_2d_sincos_position_embedding(
                    eval_size[1] // 32,
                    eval_size[0] // 32,
                    embed_dim=self.hidden_dim)
            else:
                self.pos_embed = None

            encoder_layer = TransformerEncoderLayer(
                self.hidden_dim, nhead, dim_feedforward, dropout, activation,
                attn_dropout, act_dropout, normalize_before)
            encoder_norm = nn.LayerNorm(
                self.hidden_dim) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        fpn_stages = []
        fpn_routes = []
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_sublayer(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   spp=(spp and i == 0),
                                   use_alpha=use_alpha))

            if drop_block:
                stage.add_sublayer('drop', DropBlock(block_size, keep_prob))

            fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act))

            ch_pre = ch_out

        self.fpn_stages = nn.LayerList(fpn_stages)
        self.fpn_routes = nn.LayerList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act))

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_sublayer(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   spp=False,
                                   use_alpha=use_alpha))
            if drop_block:
                stage.add_sublayer('drop', DropBlock(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.LayerList(pan_stages[::-1])
        self.pan_routes = nn.LayerList(pan_routes[::-1])

    def build_2d_sincos_position_embedding(
            self,
            w,
            h,
            embed_dim=1024,
            temperature=10000., ):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        pos_emb = paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

        return pos_emb

    def forward(self, blocks, for_mot=False):
        if self.use_trans:
            last_feat = blocks[-1]
            n, c, h, w = last_feat.shape

            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = last_feat.flatten(2).transpose([0, 2, 1])
            if self.eval_size is not None and not self.training:
                pos_embed = self.pos_embed
            else:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w=w, h=h, embed_dim=self.hidden_dim)

            memory = self.encoder(src_flatten, pos_embed=pos_embed)
            last_feat_encode = memory.transpose([0, 2, 1]).reshape([n, c, h, w])
            blocks[-1] = last_feat_encode

        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2., data_format=self.data_format)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = paddle.concat([route, block], axis=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
