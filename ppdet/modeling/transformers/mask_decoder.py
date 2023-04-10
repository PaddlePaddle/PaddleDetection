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
'''
Modified from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''
import math
from IPython import embed
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable

from typing import List, Tuple, Type
from ..backbones.vit_image_encoder import LayerNorm2d, MLPBlock

__all__ = ['MaskDecoder']


class TwoWayTransformer(nn.Layer):
    def __init__(self,
                 depth=2,
                 embedding_dim=256,
                 num_heads=8,
                 mlp_dim=2048,
                 activation=nn.ReLU,
                 attention_downsample_rate=2):
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Layer): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.LayerList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=i == 0))
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(
            normalized_shape=embedding_dim,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None)

    def forward(self, image_embedding, image_pe, point_embedding):
        """
        Args:
          image_embedding (paddle.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (paddle.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (paddle.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          paddle.Tensor: the processed point_embedding
          paddle.Tensor: the processed image_embedding
        """
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(start_axis=2).transpose(
            perm=[0, 2, 1])
        image_pe = image_pe.flatten(start_axis=2).transpose(perm=[0, 2, 1])
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TwoWayAttentionBlock(nn.Layer):
    def __init__(self,
                 embedding_dim,
                 num_heads,
                 mlp_dim=2048,
                 activation=nn.ReLU,
                 attention_downsample_rate=2,
                 skip_first_layer_pe=False):
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Layer): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(
            normalized_shape=embedding_dim,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(
            normalized_shape=embedding_dim,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(
            normalized_shape=embedding_dim,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None)
        self.norm4 = nn.LayerNorm(
            normalized_shape=embedding_dim,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class Attention(nn.Layer):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 downsample_rate: int=1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, 'num_heads must divide embedding_dim.'
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape([b, n, num_heads, c // num_heads])
        # x = x
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        return x.transpose(perm=perm_0)

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        x = x.transpose(perm=perm_1)
        return x.reshape([b, n_tokens, n_heads * c_per_head])

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @k.transpose(perm=[0, 1, 3, 2])
        attn = attn / math.sqrt(c_per_head)
        attn = F.softmax(attn, axis=-1)
        out = attn @v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


@register
@serializable
class MaskDecoder(nn.Layer):
    def __init__(self,
                 num_multimask_outputs=3,
                 transformer_dim=256,
                 activation=nn.GELU,
                 iou_head_depth=3,
                 iou_head_hidden_dim=256):
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Layer): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Layer): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            num_heads=8,
            mlp_dim=2048,
            activation=nn.GELU)

        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Linear(1, transformer_dim)  # Embedding
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Linear(self.num_mask_tokens,
                                     transformer_dim)  # Embedding
        self.output_upscaling = nn.Sequential(
            nn.Conv2DTranspose(
                in_channels=transformer_dim,
                out_channels=transformer_dim // 4,
                kernel_size=2,
                stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2DTranspose(
                in_channels=transformer_dim // 4,
                out_channels=transformer_dim // 8,
                kernel_size=2,
                stride=2),
            activation())
        self.output_hypernetworks_mlps = nn.LayerList(sublayers=[
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

    def forward(self,
                image_embeddings: paddle.Tensor,
                image_pe: paddle.Tensor,
                sparse_prompt_embeddings: paddle.Tensor,
                dense_prompt_embeddings: paddle.Tensor,
                multimask_output: bool) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (paddle.Tensor): the embeddings from the image encoder
          image_pe (paddle.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (paddle.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (paddle.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          paddle.Tensor: batched predicted masks
          paddle.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, (mask_slice), :, :]
        iou_pred = iou_pred[:, (mask_slice)]
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: paddle.Tensor,  # [1, 256, 64, 64]
            image_pe: paddle.Tensor,  # [1, 256, 64, 64]
            sparse_prompt_embeddings: paddle.Tensor,  # [1, 0, 256]
            dense_prompt_embeddings: paddle.Tensor):  # [1, 256, 64, 64]
        """Predicts masks. See 'forward' for more details."""
        output_tokens = paddle.concat(
            x=[self.iou_token.weight, self.mask_tokens.weight], axis=0)
        output_tokens = output_tokens.unsqueeze(axis=0).expand(
            [sparse_prompt_embeddings.shape[0], -1, -1])
        # [1, 5, 256]
        tokens = paddle.concat(
            x=(output_tokens, sparse_prompt_embeddings), axis=1)
        src = paddle.repeat_interleave(
            image_embeddings, tokens.shape[0], axis=0)
        src = src + dense_prompt_embeddings
        pos_src = paddle.repeat_interleave(image_pe, tokens.shape[0], axis=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, (0), :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        src = src.transpose([0, 2, 1]).reshape([b, c, h, w])
        # [1, 256, 64, 64]
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[paddle.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](
                mask_tokens_out[:, (i), :]))
        hyper_in = paddle.stack(x=hyper_in_list, axis=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @upscaled_embedding.reshape([b, c, h * w])).reshape(
            [b, -1, h, w])
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class MLP(nn.Layer):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 sigmoid_output: bool=False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(sublayers=(nn.Linear(
            in_features=n,
            out_features=k) for n, k in zip([input_dim] + h, h + [output_dim])))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(x=layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x=x)
        return x
