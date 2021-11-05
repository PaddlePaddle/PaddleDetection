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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention, _convert_attention_mask
from .position_encoding import PositionEmbedding
from .utils import _get_clones
from ..initializer import linear_init_, conv_init_, xavier_uniform_, normal_

__all__ = ['DETRTransformer']


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
        src_mask = _convert_attention_mask(src_mask, src.dtype)

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
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        q = self.with_pos_embed(tgt, query_pos_embed)
        k = self.with_pos_embed(memory, pos_embed)
        tgt = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Layer):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                pos_embed=pos_embed,
                query_pos_embed=query_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


@register
class DETRTransformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 num_queries=100,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=2048,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(DETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'],\
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        self.input_proj = nn.Conv2D(
            backbone_num_channels, hidden_dim, kernel_size=1)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        conv_init_(self.input_proj)
        normal_(self.query_pos_embed.weight)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'backbone_num_channels': [i.channels for i in input_shape][-1],
        }

    def forward(self, src, src_mask=None):
        r"""
        Applies a Transformer model on the inputs.

        Parameters:
            src (List(Tensor)): Backbone feature maps with shape [[bs, c, h, w]].
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                [bs, H, W]`. When the data type is bool, the unwanted positions
                have `False` values and the others have `True` values. When the
                data type is int, the unwanted positions have 0 values and the
                others have 1 values. When the data type is float, the unwanted
                positions have `-INF` values and the others have 0 values. It
                can be None when nothing wanted or needed to be prevented
                attention to. Default None.

        Returns:
            output (Tensor): [num_levels, batch_size, num_queries, hidden_dim]
            memory (Tensor): [batch_size, hidden_dim, h, w]
        """
        # use last level feature map
        src_proj = self.input_proj(src[-1])
        bs, c, h, w = src_proj.shape
        # flatten [B, C, H, W] to [B, HxW, C]
        src_flatten = src_proj.flatten(2).transpose([0, 2, 1])
        if src_mask is not None:
            src_mask = F.interpolate(
                src_mask.unsqueeze(0).astype(src_flatten.dtype),
                size=(h, w))[0].astype('bool')
        else:
            src_mask = paddle.ones([bs, h, w], dtype='bool')
        pos_embed = self.position_embedding(src_mask).flatten(2).transpose(
            [0, 2, 1])

        src_mask = _convert_attention_mask(src_mask, src_flatten.dtype)
        src_mask = src_mask.reshape([bs, 1, 1, -1])

        memory = self.encoder(
            src_flatten, src_mask=src_mask, pos_embed=pos_embed)

        query_pos_embed = self.query_pos_embed.weight.unsqueeze(0).tile(
            [bs, 1, 1])
        tgt = paddle.zeros_like(query_pos_embed)
        output = self.decoder(
            tgt,
            memory,
            memory_mask=src_mask,
            pos_embed=pos_embed,
            query_pos_embed=query_pos_embed)

        return (output, memory.transpose([0, 2, 1]).reshape([bs, c, h, w]),
                src_proj, src_mask.reshape([bs, 1, 1, h, w]))
