# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.distributed.fleet.utils import recompute

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from .utils import _get_clones, get_valid_ratio
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

__all__ = ['DeformableTransformer']


class MSDeformableAttention(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        try:
            # use cuda op
            print('use cuda ms_deformable_attn op!')
            from deformable_detr_ops import ms_deformable_attn
        except:
            # use paddle func
            from .utils import deformable_attention_core_func as ms_deformable_attn
        self.ms_deformable_attn_core = ms_deformable_attn

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1])
        scaling = paddle.arange(
            1, self.num_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (Tensor(int64)): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer.astype(sampling_offsets.dtype)
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


@register
class DeformableTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 lr_mult=0.1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                               n_points, lr_mult)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                reference_points,
                spatial_shapes,
                level_start_index,
                src_mask=None,
                query_pos_embed=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, query_pos_embed), reference_points, src,
            spatial_shapes, level_start_index, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src


@register
class DeformableTransformerEncoder(nn.Layer):
    __inject__ = ["encoder_layer"]
    def __init__(self, encoder_layer, num_layers, with_rp=-1):
        super(DeformableTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        assert with_rp <= num_layers
        self.with_rp = with_rp

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, offset=0.5):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = paddle.meshgrid(
                paddle.arange(end=H) + offset, paddle.arange(end=W) + offset)
            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] *
                                                    H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] *
                                                    W)
            reference_points.append(paddle.stack((ref_x, ref_y), axis=-1))
        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(self,
                feat,
                spatial_shapes,
                level_start_index,
                feat_mask=None,
                query_pos_embed=None,
                valid_ratios=None):
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [feat.shape[0], spatial_shapes.shape[0], 2])
        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios)
        for i, layer in enumerate(self.layers):
            if self.training and i < self.with_rp:
                feat = recompute(layer, feat, reference_points, spatial_shapes,
                             level_start_index, feat_mask, query_pos_embed,
                             **{"preserve_rng_state": True, "use_reentrant": False})
            else:
                feat = layer(feat, reference_points, spatial_shapes,
                                 level_start_index, feat_mask, query_pos_embed)

        return feat


@register
class DeformableTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 lr_mult=0.1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points, lr_mult)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2 = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Layer):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super(DeformableTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask=None,
                query_pos_embed=None):
        output = tgt
        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, reference_points, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           memory_mask, query_pos_embed)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


@register
class DeformableTransformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 num_queries=300,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 in_feats_channel=[512, 1024, 2048],
                 num_feature_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 lr_mult=0.1,
                 pe_temperature=10000,
                 pe_offset=-0.5):
        super(DeformableTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(in_feats_channel) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_encoder_points, lr_mult)
        self.encoder = DeformableTransformerEncoder(encoder_layer,
                                                    num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_decoder_points)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.reference_points = nn.Linear(
            hidden_dim,
            2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.input_proj = nn.LayerList()
        for in_channels in in_feats_channel:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim)))
        in_channels = in_feats_channel[-1]
        for _ in range(num_feature_levels - len(in_feats_channel)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    nn.GroupNorm(32, hidden_dim)))
            in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            temperature=pe_temperature,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=pe_offset,
            eps=1e-4)

        self._reset_parameters()

    def _reset_parameters(self):
        normal_(self.level_embed.weight)
        normal_(self.tgt_embed.weight)
        normal_(self.query_pos_embed.weight)
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_feats_channel': [i.channels for i in input_shape], }

    def forward(self, src_feats, src_mask=None, *args, **kwargs):
        srcs = []
        for i in range(len(src_feats)):
            srcs.append(self.input_proj[i](src_feats[i]))
        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for i in range(len_srcs, self.num_feature_levels):
                if i == len_srcs:
                    srcs.append(self.input_proj[i](src_feats[-1]))
                else:
                    srcs.append(self.input_proj[i](srcs[-1]))
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(srcs):
            src_shape = paddle.shape(src)
            bs = src_shape[0:1]
            h = src_shape[2:3]
            w = src_shape[3:4]
            spatial_shapes.append(paddle.concat([h, w]))
            src = src.flatten(2).transpose([0, 2, 1])
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(src_mask.unsqueeze(0), size=(h, w))[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(get_valid_ratio(mask))
            pos_embed = self.position_embedding(mask).flatten(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[level]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = None if src_mask is None else paddle.concat(mask_flatten,
                                                                   1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        # [l, 2]
        spatial_shapes = paddle.to_tensor(
            paddle.stack(spatial_shapes).astype('int64'))
        # [l], 每一个level的起始index
        level_start_index = paddle.concat([
            paddle.zeros(
                [1], dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        # [b, l, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              mask_flatten, lvl_pos_embed_flatten, valid_ratios)

        # prepare input for decoder
        bs, _, c = memory.shape
        query_embed = self.query_pos_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        tgt = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        reference_points = F.sigmoid(self.reference_points(query_embed))
        reference_points_input = reference_points.unsqueeze(
            2) * valid_ratios.unsqueeze(1)

        # decoder
        hs = self.decoder(tgt, reference_points_input, memory, spatial_shapes,
                          level_start_index, mask_flatten, query_embed)

        return (hs, memory, reference_points)


class QRDeformableTransformerDecoder(DeformableTransformerDecoder):
    def __init__(self, decoder_layer, num_layers,
                 start_q=None, end_q=None, return_intermediate=False):
        super(QRDeformableTransformerDecoder, self).__init__(
            decoder_layer, num_layers, return_intermediate=return_intermediate)
        self.start_q = start_q
        self.end_q = end_q

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask=None,
                query_pos_embed=None):

        if not self.training:
            return super(QRDeformableTransformerDecoder, self).forward(
                tgt, reference_points,
                memory, memory_spatial_shapes,
                memory_level_start_index,
                memory_mask=memory_mask,
                query_pos_embed=query_pos_embed)

        batchsize = tgt.shape[0]
        query_list_reserve = [tgt]
        intermediate = []
        for lid, layer in enumerate(self.layers):

            start_q = self.start_q[lid]
            end_q = self.end_q[lid]
            query_list = query_list_reserve.copy()[start_q:end_q]

            # prepare for parallel process
            output = paddle.concat(query_list, axis=0)
            fakesetsize = int(output.shape[0] / batchsize)
            reference_points_tiled = reference_points.tile([fakesetsize, 1, 1, 1])

            memory_tiled = memory.tile([fakesetsize, 1, 1])
            query_pos_embed_tiled = query_pos_embed.tile([fakesetsize, 1, 1])
            memory_mask_tiled = memory_mask.tile([fakesetsize, 1])

            output = layer(output, reference_points_tiled, memory_tiled,
                           memory_spatial_shapes, memory_level_start_index,
                           memory_mask_tiled, query_pos_embed_tiled)

            for i in range(fakesetsize):
                query_list_reserve.append(output[batchsize*i:batchsize*(i+1)])

            if self.return_intermediate:
                for i in range(fakesetsize):
                    intermediate.append(output[batchsize*i:batchsize*(i+1)])

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


@register
class QRDeformableTransformer(DeformableTransformer):

    def __init__(self,
                 num_queries=300,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 in_feats_channel=[512, 1024, 2048],
                 num_feature_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 lr_mult=0.1,
                 pe_temperature=10000,
                 pe_offset=-0.5,
                 start_q=None,
                 end_q=None):
        super(QRDeformableTransformer, self).__init__(
                 num_queries=num_queries,
                 position_embed_type=position_embed_type,
                 return_intermediate_dec=return_intermediate_dec,
                 in_feats_channel=in_feats_channel,
                 num_feature_levels=num_feature_levels,
                 num_encoder_points=num_encoder_points,
                 num_decoder_points=num_decoder_points,
                 hidden_dim=hidden_dim,
                 nhead=nhead,
                 num_encoder_layers=num_encoder_layers,
                 num_decoder_layers=num_decoder_layers,
                 dim_feedforward=dim_feedforward,
                 dropout=dropout,
                 activation=activation,
                 lr_mult=lr_mult,
                 pe_temperature=pe_temperature,
                 pe_offset=pe_offset)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_decoder_points)
        self.decoder = QRDeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, start_q, end_q, return_intermediate_dec)
