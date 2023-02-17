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
"""
this code is base on https://github.com/hikvision-research/opera/blob/main/opera/models/utils/transformer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention, _convert_attention_mask
from .utils import _get_clones
from ..initializer import linear_init_, normal_, constant_, xavier_uniform_

__all__ = [
    'PETRTransformer', 'MultiScaleDeformablePoseAttention',
    'PETR_TransformerDecoderLayer', 'PETR_TransformerDecoder',
    'PETR_DeformableDetrTransformerDecoder',
    'PETR_DeformableTransformerDecoder', 'TransformerEncoderLayer',
    'TransformerEncoder', 'MSDeformableAttention'
]


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


@register
class TransformerEncoderLayer(nn.Layer):
    __inject__ = ['attn']

    def __init__(self,
                 d_model,
                 attn=None,
                 nhead=8,
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
        self.embed_dims = d_model

        if attn is None:
            self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        else:
            self.self_attn = attn
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

    def forward(self, src, src_mask=None, pos_embed=None, **kwargs):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask, **kwargs)

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


@register
class TransformerEncoder(nn.Layer):
    __inject__ = ['encoder_layer']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.embed_dims = encoder_layer.embed_dims

    def forward(self, src, src_mask=None, pos_embed=None, **kwargs):
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=src_mask, pos_embed=pos_embed, **kwargs)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
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
            print("use deformable_detr_ops in ms_deformable_attn")
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
                key,
                value,
                reference_points,
                value_spatial_shapes,
                value_level_start_index,
                attn_mask=None,
                **kwargs):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (Tensor(int64)): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            attn_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if attn_mask is not None:
            attn_mask = attn_mask.astype(value.dtype).unsqueeze(-1)
            value *= attn_mask
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


@register
class MultiScaleDeformablePoseAttention(nn.Layer):
    """An attention module used in PETR. `End-to-End Multi-Person
    Pose Estimation with Transformers`.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 17.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0.1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=17,
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False,
                 lr_mult=0.1):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn("You'd better set embed_dims in "
                          'MultiScaleDeformAttention to make '
                          'the dimension of each attention head a power of 2 '
                          'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        try:
            # use cuda op
            from deformable_detr_ops import ms_deformable_attn
        except:
            # use paddle func
            from .utils import deformable_attention_core_func as ms_deformable_attn
        self.ms_deformable_attn_core = ms_deformable_attn

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_(self.sampling_offsets.weight)
        constant_(self.sampling_offsets.bias)
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                attn_mask=None,
                reference_points=None,
                value_spatial_shapes=None,
                value_level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape (num_key, bs, embed_dims).
            value (Tensor): The value tensor with shape
                (num_key, bs, embed_dims).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            reference_points (Tensor):  The normalized reference points with
                shape (bs, num_query, num_levels, K*2), all elements is range
                in [0, 1], top-left (0,0), bottom-right (1, 1), including
                padding area.
            attn_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            value_spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            value_level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        bs, num_query, _ = query.shape
        bs, num_key, _ = value.shape
        assert (value_spatial_shapes[:, 0].numpy() *
                value_spatial_shapes[:, 1].numpy()).sum() == num_key

        value = self.value_proj(value)
        if attn_mask is not None:
            # value = value.masked_fill(attn_mask[..., None], 0.0)
            value *= attn_mask.unsqueeze(-1)
        value = value.reshape([bs, num_key, self.num_heads, -1])
        sampling_offsets = self.sampling_offsets(query).reshape([
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        ])
        attention_weights = self.attention_weights(query).reshape(
            [bs, num_query, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights, axis=-1)

        attention_weights = attention_weights.reshape(
            [bs, num_query, self.num_heads, self.num_levels, self.num_points])
        if reference_points.shape[-1] == self.num_points * 2:
            reference_points_reshape = reference_points.reshape(
                (bs, num_query, self.num_levels, -1, 2)).unsqueeze(2)
            x1 = reference_points[:, :, :, 0::2].min(axis=-1, keepdim=True)
            y1 = reference_points[:, :, :, 1::2].min(axis=-1, keepdim=True)
            x2 = reference_points[:, :, :, 0::2].max(axis=-1, keepdim=True)
            y2 = reference_points[:, :, :, 1::2].max(axis=-1, keepdim=True)
            w = paddle.clip(x2 - x1, min=1e-4)
            h = paddle.clip(y2 - y1, min=1e-4)
            wh = paddle.concat([w, h], axis=-1)[:, :, None, :, None, :]

            sampling_locations = reference_points_reshape \
                                 + sampling_offsets * wh * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2K, but get {reference_points.shape[-1]} instead.')

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output


@register
class PETR_TransformerDecoderLayer(nn.Layer):
    __inject__ = ['self_attn', 'cross_attn']

    def __init__(self,
                 d_model,
                 nhead=8,
                 self_attn=None,
                 cross_attn=None,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(PETR_TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        if self_attn is None:
            self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        else:
            self.self_attn = self_attn
        if cross_attn is None:
            self.cross_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        else:
            self.cross_attn = cross_attn
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
                query_pos_embed=None,
                **kwargs):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

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
        key_tmp = tgt
        # k = self.with_pos_embed(memory, pos_embed)
        tgt = self.cross_attn(
            q, key=key_tmp, value=memory, attn_mask=memory_mask, **kwargs)
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


@register
class PETR_TransformerDecoder(nn.Layer):
    """Implements the decoder in PETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """
    __inject__ = ['decoder_layer']

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False,
                 num_keypoints=17,
                 **kwargs):
        super(PETR_TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.num_keypoints = num_keypoints

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                kpt_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims).
            reference_points (Tensor): The reference points of offset,
                has shape (bs, num_query, K*2).
            valid_ratios (Tensor): The radios of valid points on the feature
                map, has shape (bs, num_levels, 2).
            kpt_branches: (obj:`nn.LayerList`): Used for refining the
                regression results. Only would be passed when `with_box_refine`
                is True, otherwise would be passed a `None`.

        Returns:
            tuple (Tensor): Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims] and
                [num_layers, bs, num_query, K*2].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == self.num_keypoints * 2:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios.tile((1, 1, self.num_keypoints))[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                                         valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)

            if kpt_branches is not None:
                tmp = kpt_branches[lid](output)
                if reference_points.shape[-1] == self.num_keypoints * 2:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = F.sigmoid(new_reference_points)
                else:
                    raise NotImplementedError
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(
                intermediate_reference_points)

        return output, reference_points


@register
class PETR_DeformableTransformerDecoder(nn.Layer):
    __inject__ = ['decoder_layer']

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super(PETR_DeformableTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_mask=None,
                query_pos_embed=None):
        output = tgt
        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, reference_points, memory,
                           memory_spatial_shapes, memory_mask, query_pos_embed)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


@register
class PETR_DeformableDetrTransformerDecoder(PETR_DeformableTransformerDecoder):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(PETR_DeformableDetrTransformerDecoder, self).__init__(*args,
                                                                    **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.LayerList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    paddle.concat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
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

        return output, reference_points


@register
class PETRTransformer(nn.Layer):
    """Implements the PETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """
    __inject__ = ["encoder", "decoder", "hm_encoder", "refine_decoder"]

    def __init__(self,
                 encoder="",
                 decoder="",
                 hm_encoder="",
                 refine_decoder="",
                 as_two_stage=True,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 num_keypoints=17,
                 **kwargs):
        super(PETRTransformer, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_keypoints = num_keypoints
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = self.encoder.embed_dims
        self.hm_encoder = hm_encoder
        self.refine_decoder = refine_decoder
        self.init_layers()
        self.init_weights()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        #paddle.create_parameter
        self.level_embeds = paddle.create_parameter(
            (self.num_feature_levels, self.embed_dims), dtype="float32")

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.refine_query_embedding = nn.Embedding(self.num_keypoints,
                                                       self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims,
                                              2 * self.num_keypoints)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.rank() > 1:
                xavier_uniform_(p)
                if hasattr(p, 'bias') and p.bias is not None:
                    constant_(p.bais)
        for m in self.sublayers():
            if isinstance(m, MSDeformableAttention):
                m._reset_parameters()
        for m in self.sublayers():
            if isinstance(m, MultiScaleDeformablePoseAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_uniform_(self.reference_points.weight)
            constant_(self.reference_points.bias)
        normal_(self.level_embeds)
        normal_(self.refine_query_embedding.weight)

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
            valid_H = paddle.sum(mask_flatten_[:, :, 0, 0], 1)
            valid_W = paddle.sum(mask_flatten_[:, 0, :, 0], 1)

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
            proposal = grid.reshape([N, -1, 2])
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = paddle.concat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True).astype("bool")
        output_proposals = paddle.log(output_proposals / (1 - output_proposals))
        output_proposals = masked_fill(
            output_proposals, ~memory_padding_mask.astype("bool").unsqueeze(-1),
            float('inf'))
        output_proposals = masked_fill(output_proposals,
                                       ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = masked_fill(
            output_memory, ~memory_padding_mask.astype("bool").unsqueeze(-1),
            float(0))
        output_memory = masked_fill(output_memory, ~output_proposals_valid,
                                    float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all feature maps,
                has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid points on the
                feature map, has shape (bs, num_levels, 2).

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(
                    0.5, H - 0.5, H, dtype="float32"),
                paddle.linspace(
                    0.5, W - 0.5, W, dtype="float32"))
            ref_y = ref_y.reshape(
                (-1, ))[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(
                (-1, ))[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = paddle.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = paddle.concat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all level."""
        _, H, W = mask.shape
        valid_H = paddle.sum(mask[:, :, 0].astype('float'), 1)
        valid_W = paddle.sum(mask[:, 0, :].astype('float'), 1)
        valid_ratio_h = valid_H.astype('float') / H
        valid_ratio_w = valid_W.astype('float') / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = paddle.arange(num_pos_feats, dtype="float32")
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = F.sigmoid(proposals) * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = paddle.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
            axis=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                kpt_branches=None,
                cls_branches=None):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from different level.
                Each element has shape [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from different
                level used for encoder and decoder, each element has shape
                [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            kpt_branches (obj:`nn.LayerList`): Keypoint Regression heads for
                feature maps from each decoder layer. Only would be passed when
                `with_box_refine` is Ture. Default to None.
            cls_branches (obj:`nn.LayerList`): Classification heads for
                feature maps from each decoder layer. Only would be passed when
                `as_two_stage` is Ture. Default to None.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    `return_intermediate_dec` is True output has shape \
                    (num_dec_layers, bs, num_query, embed_dims), else has \
                    shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of proposals \
                    generated from encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_kpt_unact: The regression results generated from \
                    encoder's feature maps., has shape (batch, h*w, K*2).
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed
                  ) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose((0, 2, 1))
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose((0, 2, 1))
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].reshape(
                [1, 1, -1])
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = paddle.concat(feat_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        spatial_shapes_cumsum = paddle.to_tensor(
            np.array(spatial_shapes).prod(1).cumsum(0))
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype="int64")
        level_start_index = paddle.concat((paddle.zeros(
            (1, ), dtype=spatial_shapes.dtype), spatial_shapes_cumsum[:-1]))
        valid_ratios = paddle.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios)

        memory = self.encoder(
            src=feat_flatten,
            pos_embed=lvl_pos_embed_flatten,
            src_mask=mask_flatten,
            value_spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            value_level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        bs, _, c = memory.shape

        hm_proto = None
        if self.training:
            hm_memory = paddle.slice(
                memory,
                starts=level_start_index[0],
                ends=level_start_index[1],
                axes=[1])
            hm_pos_embed = paddle.slice(
                lvl_pos_embed_flatten,
                starts=level_start_index[0],
                ends=level_start_index[1],
                axes=[1])
            hm_mask = paddle.slice(
                mask_flatten,
                starts=level_start_index[0],
                ends=level_start_index[1],
                axes=[1])
            hm_reference_points = paddle.slice(
                reference_points,
                starts=level_start_index[0],
                ends=level_start_index[1],
                axes=[1])[:, :, :1, :]

            # official code make a mistake of pos_embed to pose_embed, which disable pos_embed
            hm_memory = self.hm_encoder(
                src=hm_memory,
                pose_embed=hm_pos_embed,
                src_mask=hm_mask,
                value_spatial_shapes=spatial_shapes[[0]],
                reference_points=hm_reference_points,
                value_level_start_index=level_start_index[0],
                valid_ratios=valid_ratios[:, :1, :])
            hm_memory = hm_memory.reshape((bs, spatial_shapes[0, 0],
                                           spatial_shapes[0, 1], -1))
            hm_proto = (hm_memory, mlvl_masks[0])

        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_kpt_unact = \
                kpt_branches[self.decoder.num_layers](output_memory)
            enc_outputs_kpt_unact[..., 0::2] += output_proposals[..., 0:1]
            enc_outputs_kpt_unact[..., 1::2] += output_proposals[..., 1:2]

            topk = self.two_stage_num_proposals
            topk_proposals = paddle.topk(
                enc_outputs_class[..., 0], topk, axis=1)[1].unsqueeze(-1)

            #paddle.take_along_axis 对应torch.gather
            topk_kpts_unact = paddle.take_along_axis(enc_outputs_kpt_unact,
                                                     topk_proposals, 1)
            topk_kpts_unact = topk_kpts_unact.detach()

            reference_points = F.sigmoid(topk_kpts_unact)
            init_reference_out = reference_points
            # learnable query and query_pos
            query_pos, query = paddle.split(
                query_embed, query_embed.shape[1] // c, axis=1)
            query_pos = query_pos.unsqueeze(0).expand((bs, -1, -1))
            query = query.unsqueeze(0).expand((bs, -1, -1))
        else:
            query_pos, query = paddle.split(
                query_embed, query_embed.shape[1] // c, axis=1)
            query_pos = query_pos.unsqueeze(0).expand((bs, -1, -1))
            query = query.unsqueeze(0).expand((bs, -1, -1))
            reference_points = F.sigmoid(self.reference_points(query_pos))
            init_reference_out = reference_points

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,
            memory=memory,
            query_pos_embed=query_pos,
            memory_mask=mask_flatten,
            reference_points=reference_points,
            value_spatial_shapes=spatial_shapes,
            value_level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            kpt_branches=kpt_branches)

        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, \
                   inter_references_out, enc_outputs_class, \
                   enc_outputs_kpt_unact, hm_proto, memory
        return inter_states, init_reference_out, \
               inter_references_out, None, None, None, None, None, hm_proto

    def forward_refine(self,
                       mlvl_masks,
                       memory,
                       reference_points_pose,
                       img_inds,
                       kpt_branches=None,
                       **kwargs):
        mask_flatten = []
        spatial_shapes = []
        for lvl, mask in enumerate(mlvl_masks):
            bs, h, w = mask.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        mask_flatten = paddle.concat(mask_flatten, 1)
        spatial_shapes_cumsum = paddle.to_tensor(
            np.array(
                spatial_shapes, dtype='int64').prod(1).cumsum(0))
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype="int64")
        level_start_index = paddle.concat((paddle.zeros(
            (1, ), dtype=spatial_shapes.dtype), spatial_shapes_cumsum[:-1]))
        valid_ratios = paddle.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # pose refinement (17 queries corresponding to 17 keypoints)
        # learnable query and query_pos
        refine_query_embedding = self.refine_query_embedding.weight
        query_pos, query = paddle.split(refine_query_embedding, 2, axis=1)
        pos_num = reference_points_pose.shape[0]
        query_pos = query_pos.unsqueeze(0).expand((pos_num, -1, -1))
        query = query.unsqueeze(0).expand((pos_num, -1, -1))
        reference_points = reference_points_pose.reshape(
            (pos_num, reference_points_pose.shape[1] // 2, 2))
        pos_memory = memory[img_inds]
        mask_flatten = mask_flatten[img_inds]
        valid_ratios = valid_ratios[img_inds]
        if img_inds.size == 1:
            pos_memory = pos_memory.unsqueeze(0)
            mask_flatten = mask_flatten.unsqueeze(0)
            valid_ratios = valid_ratios.unsqueeze(0)
        inter_states, inter_references = self.refine_decoder(
            query=query,
            memory=pos_memory,
            query_pos_embed=query_pos,
            memory_mask=mask_flatten,
            reference_points=reference_points,
            value_spatial_shapes=spatial_shapes,
            value_level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=kpt_branches,
            **kwargs)
        # [num_decoder, num_query, bs, embed_dim]

        init_reference_out = reference_points
        return inter_states, init_reference_out, inter_references
