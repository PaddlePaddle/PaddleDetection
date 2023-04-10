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

import paddle
import paddle.nn as nn
from ppdet.core.workspace import register, serializable

import numpy as np
from typing import Any, Optional, Tuple, Type
from .vit_image_encoder import LayerNorm2d

__all__ = ['PromptEncoder']


@register
@serializable
class PromptEncoder(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 image_embedding_size=[64, 64],
                 input_image_size=[1024, 1024],
                 mask_in_chans=16,
                 activation=nn.GELU):
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Layer): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4
        point_embeddings = [
            nn.Linear(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.LayerList(sublayers=point_embeddings)
        self.not_a_point_embed = nn.Linear(1, embed_dim)
        self.mask_input_size = 4 * image_embedding_size[
            0], 4 * image_embedding_size[1]
        self.mask_downscaling = nn.Sequential(
            nn.Conv2D(
                in_channels=1,
                out_channels=mask_in_chans // 4,
                kernel_size=2,
                stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2D(
                in_channels=mask_in_chans // 4,
                out_channels=mask_in_chans,
                kernel_size=2,
                stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2D(
                in_channels=mask_in_chans,
                out_channels=embed_dim,
                kernel_size=1))
        self.no_mask_embed = nn.Linear(1, embed_dim)

    def get_dense_pe(self) -> paddle.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          paddle.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(axis=0)

    def _embed_points(self,
                      points: paddle.Tensor,
                      labels: paddle.Tensor,
                      pad: bool) -> paddle.Tensor:
        """Embeds point prompts."""
        points = points + 0.5
        if pad:
            padding_point = paddle.zeros(shape=(points.shape[0], 1, 2))
            padding_label = -paddle.ones(shape=(labels.shape[0], 1))
            points = paddle.concat(x=[points, padding_point], axis=1)
            labels = paddle.concat(x=[labels, padding_label], axis=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: paddle.Tensor) -> paddle.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5
        coords = boxes.reshape([-1, 2, 2])
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size)
        corner_embedding[:, (0), :] += self.point_embeddings[2].weight
        corner_embedding[:, (1), :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: paddle.Tensor) -> paddle.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self,
                        points: Optional[Tuple[paddle.Tensor, paddle.Tensor]],
                        boxes: Optional[paddle.Tensor],
                        masks: Optional[paddle.Tensor]) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(self,
                points: Optional[Tuple[paddle.Tensor, paddle.Tensor]],
                boxes: Optional[paddle.Tensor],
                masks: Optional[paddle.Tensor]) -> Tuple[paddle.Tensor,
                                                         paddle.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(paddle.Tensor, paddle.Tensor) or none): point coordinates
            and labels to embed.
          boxes (paddle.Tensor or none): boxes to embed
          masks (paddle.Tensor or none): masks to embed

        Returns:
          paddle.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          paddle.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = paddle.empty(shape=(bs, 0, self.embed_dim))
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(
                coords, labels, pad=boxes is None)
            sparse_embeddings = paddle.concat(
                x=[sparse_embeddings, point_embeddings], axis=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = paddle.concat(
                x=[sparse_embeddings, box_embeddings], axis=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                [1, -1, 1, 1]).expand([
                    bs, -1, self.image_embedding_size[0],
                    self.image_embedding_size[1]
                ])
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Layer):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int=64,
                 scale: Optional[float]=None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            'positional_encoding_gaussian_matrix',
            scale * paddle.randn(shape=(2, num_pos_feats)))

    def _pe_encoding(self, coords: paddle.Tensor) -> paddle.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return paddle.concat(
            x=[paddle.sin(x=coords), paddle.cos(x=coords)], axis=-1)

    def forward(self, size: Tuple[int, int]) -> paddle.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = paddle.ones(shape=(h, w), dtype='float32')
        y_embed = grid.cumsum(axis=0) - 0.5
        x_embed = grid.cumsum(axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(paddle.stack(x=[x_embed, y_embed], axis=-1))
        return pe.transpose(perm=[2, 0, 1])

    def forward_with_coords(self,
                            coords_input: paddle.Tensor,
                            image_size: Tuple[int, int]) -> paddle.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, (0)] = coords[:, :, (0)] / image_size[1]
        coords[:, :, (1)] = coords[:, :, (1)] / image_size[0]
        coords = paddle.cast('float32')
        return self._pe_encoding(coords)
