# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from .utils import _get_clones, get_valid_ratio, inverse_sigmoid
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

__all__ = ['OVDeformableTransformer']


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


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
            value = masked_fill(value, value_mask, 0.0)

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
            ]) + sampling_offsets / offset_normalizer
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


class DeformableTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super(DeformableTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

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

        for layer in self.layers:
            feat = layer(feat, reference_points, spatial_shapes,
                         level_start_index, feat_mask, query_pos_embed)

        return feat


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

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points, lr_mult)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
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
                query_pos_embed=None,
                tgt_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
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
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 return_intermediate=False,
                 no_sine_embed=False):
        super(DeformableTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.no_sine_embed = no_sine_embed

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask=None,
                memory_valid_ratios=None,
                query_pos_embed=None,
                tgt_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] *
                    paddle.concat([memory_valid_ratios, memory_valid_ratios],
                                  -1)[:, None])
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :,
                                                          None] * memory_valid_ratios[:,
                                                                                      None]

            output = layer(
                output,
                reference_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask,
                query_pos_embed,
                tgt_mask=tgt_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = F.sigmoid(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = F.sigmoid(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(
                intermediate_reference_points)

        return output.unsqueeze(0), reference_points


@register
class OVDeformableTransformer(nn.Layer):
    def __init__(self,
                 num_queries=300,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=[512, 1024, 2048],
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
                 two_stage=False,
                 two_stage_num_proposals=300):

        super(OVDeformableTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_num_channels) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

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

        #self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.level_embed = self.create_parameter(
            shape=(num_feature_levels, hidden_dim),
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        if self.two_stage:
            self.enc_output = paddle.nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = paddle.nn.LayerNorm(hidden_dim)
            self.pos_trans = paddle.nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.pos_trans_norm = paddle.nn.LayerNorm(hidden_dim * 2)
        else:
            self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)
            self.reference_points = nn.Linear(
                hidden_dim,
                2,
                weight_attr=ParamAttr(learning_rate=lr_mult),
                bias_attr=ParamAttr(learning_rate=lr_mult))
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.LayerList()
        for in_channels in backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim)))
        in_channels = backbone_num_channels[-1]
        for _ in range(num_feature_levels - len(backbone_num_channels)):
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
            offset=pe_offset)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
                if hasattr(p, 'bias') and p.bias is not None:
                    constant_(p.bais)
        for m in self.sublayers():
            if isinstance(m, MSDeformableAttention):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight)
            constant_(self.reference_points.bias)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = paddle.arange(start=num_pos_feats).astype('int32')
        dim_t = temperature**(2 * (dim_t // 2).astype('float32') /
                              num_pos_feats)
        proposals = F.sigmoid(proposals) * scale
        pos = proposals[:, :, :, (None)] / dim_t
        pos = paddle.stack(
            x=(pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
            axis=4).flatten(start_axis=2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor): The output of encoder, has shape
                (bs, num_key, embed_dim). num_key is equal the number of points
                on feature map from all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder, has shape
                    (bs, num_key, embed_dim). num_key is equal the number of
                    points on feature map from all levels.
                - output_proposals (Tensor): The normalized proposal
                    after a inverse sigmoid, has shape (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0

        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].reshape(
                [N, H, W, 1])
            valid_H = paddle.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = paddle.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = paddle.meshgrid(
                paddle.linspace(
                    0, H - 1, H, dtype="float32"),
                paddle.linspace(
                    0, W - 1, W, dtype="float32"))
            grid = paddle.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)],
                                 -1)

            scale = paddle.concat(
                [valid_W.unsqueeze(-1),
                 valid_H.unsqueeze(-1)], 1).reshape([N, 1, 1, 2])
            grid = (grid.unsqueeze(0).expand((N, -1, -1, -1)) + 0.5) / scale
            wh = paddle.ones_like(x=grid) * 0.05 * 2.0**lvl
            proposal = paddle.concat(x=(grid, wh), axis=-1).reshape((N, -1, 4))
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = paddle.concat(proposals, 1)

        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True).astype("bool")
        output_proposals = paddle.log(output_proposals / (1 - output_proposals))
        output_proposals = masked_fill(
            output_proposals,
            memory_padding_mask.astype("bool").unsqueeze(-1), float('inf'))
        output_proposals = masked_fill(output_proposals,
                                       ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = masked_fill(
            output_memory,
            memory_padding_mask.astype("bool").unsqueeze(-1), float(0))
        output_memory = masked_fill(output_memory, ~output_proposals_valid,
                                    float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward(self, src_feats, src_mask=None, text_query=None):
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
            bs, _, h, w = paddle.shape(src)
            spatial_shapes.append(paddle.concat([h, w]))
            src = src.flatten(2).transpose([0, 2, 1])
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(
                    src_mask.unsqueeze(0).astype('int32'),
                    size=(h, w)).astype('bool')[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(get_valid_ratio(mask))
            pos_embed = self.position_embedding((mask).astype(
                'float32')).flatten(1, 2)
            lvl_pos_embed = pos_embed + paddle.reshape(
                self.level_embed[level], shape=(1, 1, -1))
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.astype('int32').flatten(1).astype('bool')
            mask_flatten.append(~mask)

        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)

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
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[
                self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[
                self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = paddle.topk(
                x=enc_outputs_class[..., 0], k=topk, axis=1)[1]
            topk_coords_unact = paddle.take_along_axis(
                arr=enc_outputs_coord_unact,
                axis=1,
                indices=topk_proposals.unsqueeze(axis=-1).tile(
                    repeat_times=[1, 1, 4]))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = F.sigmoid(topk_coords_unact)
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = paddle.split(pos_trans_out, [c, c], axis=2)
            num_queries = query_embed.shape[1]
            num_patch = len(text_query)
            query_embed = query_embed.tile(repeat_times=[1, num_patch, 1])

            tgt = tgt.tile(repeat_times=[1, num_patch, 1])
            text_query = paddle.repeat_interleave(
                text_query, num_queries, axis=0)
            text_query = paddle.expand(
                text_query.unsqueeze(axis=0), [bs, -1, -1])
            tgt = tgt + text_query
            reference_points = reference_points.tile(
                repeat_times=[1, num_patch, 1])
            init_reference_out = init_reference_out.tile(
                repeat_times=[1, num_patch, 1])

        else:
            query_embed = self.query_pos_embed.weight
            num_queries = len(num_queries)
            num_patch = len(text_query)
            query_embed = query_embed.unsqueeze(0).tile([bs, 1, 1])
            tgt = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])

            text_query = paddle.repeat_interleave(
                text_query, num_queries, axis=0)
            text_query = text_query.unsqueeze(axis=0).expand(bs, -1, -1)

            tgt = tgt + text_query
            reference_points = F.sigmoid(self.reference_points(query_embed))
            init_reference_out = reference_points

        decoder_mask = (paddle.ones([
            num_queries * num_patch,
            num_queries * num_patch,
        ]) * float("-inf"))

        for i in range(num_patch):
            decoder_mask[i * num_queries:(i + 1) * num_queries, i * num_queries:
                         (i + 1) * num_queries, ] = 0

        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_mask=mask_flatten,
            memory_valid_ratios=valid_ratios,
            query_pos_embed=query_embed,
            tgt_mask=decoder_mask, )

        memory_features = []
        spatial_index = 0
        for lvl in range(len(spatial_shapes)):
            h, w = spatial_shapes[lvl]
            memory_lvl = (paddle.transpose(
                paddle.reshape(
                    memory[:, spatial_index:spatial_index + h * w, :],
                    (bs, h, w, c)),
                perm=(0, 3, 1, 2)))
            memory_features.append(memory_lvl)
            spatial_index += h * w

        inter_references_out = inter_references
        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact, ), memory_features
        return (hs, init_reference_out, inter_references_out, None,
                None), memory_features
