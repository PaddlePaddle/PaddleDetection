# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
this code is base on https://github.com/Sense-X/Co-DETR/blob/main/projects/models/transformer.py
"""

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppdet.core.workspace import register
from ..transformers.deformable_transformer import MSDeformableAttention
from .utils import _get_clones
from ..initializer import linear_init_, normal_, constant_, xavier_uniform_
from ..shape_spec import ShapeSpec

__all__ = [
    "CoDeformableDetrTransformerDecoder",
    "CoDeformableDetrTransformer",
]


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
class CoDeformableDetrTransformerDecoder(nn.Layer):
    __inject__ = ["decoder_layer"]

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        look_forward_twice=False,
        **kwargs
    ):
        super(CoDeformableDetrTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice

    def forward(
        self,
        query,
        reference_points=None,
        valid_ratios=None,
        reg_branches=None,
        **kwargs
    ):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims).
            reference_points (Tensor): The reference points of offset,
                has shape (bs, num_query, K*2).
            valid_ratios (Tensor): The radios of valid points on the feature
                map, has shape (bs, num_levels, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for refining the regression results.
                Only would be passed when with_box_refine is True,otherwise would be
                passed a `None`.

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
                reference_points_input = (
                    reference_points[:, :, None]
                    * paddle.concat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )
            output = layer(
                output, reference_points=reference_points_input, **kwargs
            )

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = F.sigmoid(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(
                intermediate_reference_points
            )

        return output, reference_points


@register
class CoDeformableDetrTransformer(nn.Layer):
    __inject__ = ["encoder", "decoder"]

    def __init__(
        self,
        encoder,
        decoder,
        embed_dims=256,
        mixed_selection=True,
        with_pos_coord=True,
        with_coord_feat=True,
        num_co_heads=1,
        as_two_stage=False,
        two_stage_num_proposals=300,
        num_feature_levels=4,
        **kwargs
    ):
        super(CoDeformableDetrTransformer, self).__init__(**kwargs)

        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.mixed_selection = mixed_selection
        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads
        self.num_feature_levels = num_feature_levels
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the CoDeformableDetrTransformer."""
        self.level_embeds = paddle.create_parameter(
            (self.num_feature_levels, self.embed_dims), dtype="float32"
        )

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)

        if self.with_pos_coord:
            if self.num_co_heads > 0:
                # bug: this code should be 'self.head_pos_embed = nn.Embedding(self.num_co_heads, self.embed_dims)', we keep this bug for reproducing our results with ResNet-50.
                # You can fix this bug when reproducing results with swin transformer.
                self.head_pos_embed = nn.Embedding(
                    self.num_co_heads, 1, 1, self.embed_dims
                )
                self.aux_pos_trans = nn.LayerList()
                self.aux_pos_trans_norm = nn.LayerList()
                self.pos_feats_trans = nn.LayerList()
                self.pos_feats_norm = nn.LayerList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(
                        nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
                    )
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims * 2))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(
                            nn.Linear(self.embed_dims, self.embed_dims)
                        )
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.rank() > 1:
                xavier_uniform_(p)
        for m in self.sublayers():
            if isinstance(m, MSDeformableAttention):
                m._reset_parameters()
        if not self.as_two_stage:
            xavier_uniform_(self.reference_points.weight)
            constant_(self.reference_points.bias)
        normal_(self.level_embeds)

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all level."""
        _, H, W = mask.shape
        valid_H = paddle.sum(mask[:, :, 0], 1)
        valid_W = paddle.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H.astype("float32") / H
        valid_ratio_w = valid_W.astype("float32") / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = paddle.arange(num_pos_feats, dtype="float32")
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = F.sigmoid(proposals) * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = paddle.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), axis=4
        ).flatten(2)
        return pos

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
                paddle.linspace(0.5, H - 0.5, H, dtype="float32"),
                paddle.linspace(0.5, W - 0.5, W, dtype="float32"),
            )
            ref_y = ref_y.reshape((-1,))[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape((-1,))[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = paddle.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = paddle.concat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
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
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].reshape(
                [N, H, W, 1]
            )

            valid_H = paddle.sum(mask_flatten_[:, :, 0, 0].astype("float32"), 1)
            valid_W = paddle.sum(mask_flatten_[:, 0, :, 0].astype("float32"), 1)

            grid_y, grid_x = paddle.meshgrid(
                paddle.linspace(0, H - 1, H.astype(paddle.int32), dtype="float32"),
                paddle.linspace(0, W - 1, W.astype(paddle.int32), dtype="float32"),
            )
            grid = paddle.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = paddle.concat(
                [valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1
            ).reshape([N, 1, 1, 2])
            grid = (grid.unsqueeze(0).expand((N, -1, -1, -1)) + 0.5) / scale
            wh = paddle.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = paddle.concat((grid, wh), -1).reshape([N, -1, 4])
            proposals.append(proposal)
            _cur += H * W
        output_proposals = paddle.concat(proposals, 1)
        output_proposals_valid = (
            ((output_proposals > 0.01) & (output_proposals < 0.99))
            .all(-1, keepdim=True)
        )
        output_proposals = paddle.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            ~memory_padding_mask.astype('bool').unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(
            ~memory_padding_mask.astype('bool').unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward(
        self,
        mlvl_feats,
        mlvl_masks,
        query_embed,
        mlvl_pos_embeds,
        reg_branches=None,
        cls_branches=None,
        return_encoder_output=False,
        **kwargs
    ):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)
        ):
            src_shape = paddle.shape(feat)
            h = src_shape[2:3]
            w = src_shape[3:4]
            spatial_shapes.append(paddle.concat([h, w]))
            feat = feat.flatten(2).transpose((0, 2, 1))
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose((0, 2, 1))
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].reshape((1, 1, -1))            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = paddle.concat(feat_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)

        spatial_shapes = paddle.to_tensor(
            paddle.stack(spatial_shapes).astype(paddle.int64))
        # [l], 每一个level的起始index
        level_start_index = paddle.concat(
            [paddle.zeros([1], dtype="int64"), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        valid_ratios = paddle.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        
        memory = self.encoder(
            feat_flatten,
            query_pos_embed=lvl_pos_embed_flatten,
            feat_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            enc_outputs_class = cls_branches[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                reg_branches[self.decoder.num_layers](output_memory) + output_proposals
            )
            topk = self.two_stage_num_proposals
            # We only use the first channel in enc_outputs_class as foreground,
            # the other (num_classes - 1) channels are actually not used.
            # Its targets are set to be 0s, which indicates the first
            # class (foreground) because we use [0, num_classes - 1] to
            # indicate class labels, background class is indicated by
            # num_classes (similar convention in RPN).
            topk_proposals = paddle.topk(enc_outputs_class[..., 0], topk, axis=1)[1]
            # paddle.take_along_axis 对应torch.gather
            topk_coords_unact = paddle.take_along_axis(
                enc_outputs_coord_unact, topk_proposals.unsqueeze(-1).tile([1, 1, 4]),axis=1
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            if not self.mixed_selection:
                query_pos, query = paddle.split(pos_trans_out, pos_trans_out.shape[2]//c, axis=2)
            else:
                # query_embed here is the content embed for deformable DETR
                query = query_embed.unsqueeze(0).expand([bs, -1, -1])
                query_pos, _ = paddle.split(pos_trans_out, pos_trans_out.shape[2]//c, axis=2)
        else:
            query_pos, query = paddle.split(query_embed, query_embed.shape[1]//c, axis=1)
            query_pos = query_pos.unsqueeze(0).expand([bs, -1, -1])
            query = query.unsqueeze(0).expand([bs, -1, -1])
            reference_points = F.sigmoid(self.reference_points(query_pos))
            init_reference_out = reference_points

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,
            memory=memory,
            query_pos_embed=query_pos,
            memory_mask=mask_flatten,
            reference_points=reference_points,
            memory_spatial_shapes=spatial_shapes,
            memory_level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs
        )
        inter_references_out = inter_references
        if self.as_two_stage:
            if return_encoder_output:
                return (
                    inter_states,
                    init_reference_out,
                    inter_references_out,
                    enc_outputs_class,
                    enc_outputs_coord_unact,
                    memory,
                )
            return (
                inter_states,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        if return_encoder_output:
            return (
                inter_states,
                init_reference_out,
                inter_references_out,
                None,
                None,
                memory,
            )
        return inter_states, init_reference_out, inter_references_out, None, None

    def forward_aux(
        self,
        mlvl_feats,
        mlvl_masks,
        query_embed,
        mlvl_pos_embeds,
        pos_anchors,
        pos_feats=None,
        reg_branches=None,
        cls_branches=None,
        return_encoder_output=False,
        attn_masks=None,
        head_idx=0,
        **kwargs
    ):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose((0,2,1))
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = paddle.concat(feat_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        spatial_shapes = paddle.to_tensor(spatial_shapes,dtype=paddle.int64)
        level_start_index = paddle.concat(
            [paddle.zeros([1], dtype="int64"), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )
        valid_ratios = paddle.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        memory = feat_flatten
        bs, _, c = memory.shape
        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = pos_anchors
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))            
            query_pos, query = paddle.split(pos_trans_out, pos_trans_out.shape[2]//c, axis=2)
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))
                query_pos = query_pos + self.head_pos_embed.weight[head_idx]

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,
            memory=memory,
            query_pos_embed=query_pos,  # error
            memory_mask=mask_flatten,
            reference_points=reference_points, # error
            memory_spatial_shapes=spatial_shapes,
            memory_level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs
        )

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out
