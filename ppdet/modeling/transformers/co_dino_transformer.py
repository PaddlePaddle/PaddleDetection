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

import paddle
import paddle.nn as nn

from ppdet.core.workspace import register
from ..heads.detr_head import MLP
from .co_deformable_detr_transformer import CoDeformableDetrTransformer
from .utils import inverse_sigmoid
from ..initializer import normal_

__all__ = [
    "CoDINOTransformer",
]


@register
class CoDINOTransformer(CoDeformableDetrTransformer):

    def __init__(self, *args, **kwargs):
        super(CoDINOTransformer, self).__init__(*args, **kwargs)

    def init_layers(self):
        """Initialize layers of the CoDinoTransformer."""
        self.level_embeds = paddle.create_parameter(
            (self.num_feature_levels, self.embed_dims), dtype=paddle.float32)
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)
        self.query_pos_head = MLP(self.embed_dims*2, self.embed_dims, 
                                  self.embed_dims, num_layers=2)
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.LayerList()
                self.aux_pos_trans_norm = nn.LayerList()
                self.pos_feats_trans = nn.LayerList()
                self.pos_feats_norm = nn.LayerList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims*2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        super().init_weights()
        normal_(self.query_embed.weight)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                reg_branches=None,
                cls_branches=None,
                attn_mask=None):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

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
        level_start_index = paddle.concat((paddle.zeros((1), 
            dtype=paddle.int64), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = paddle.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        
        memory = self.encoder(
            feat_flatten,
            query_pos_embed=lvl_pos_embed_flatten,
            feat_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        cls_out_features = cls_branches[self.decoder.num_layers].weight.shape[-1]
        topk = self.two_stage_num_proposals
        topk_indices = paddle.topk(enc_outputs_class.max(-1), topk, axis=1)[1]

        topk_socre = paddle.take_along_axis(
            enc_outputs_class, 
            topk_indices.unsqueeze(-1).tile([1, 1, cls_out_features]), 1)
        topk_coords_unact = paddle.take_along_axis(
            enc_outputs_coord_unact, 
            topk_indices.unsqueeze(-1).tile([1, 1, 4]), 1)
        topk_anchor = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embed.weight[None, :, :].expand([bs, -1, -1])

        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        if dn_label_query is not None:
            query = paddle.concat([dn_label_query, query], axis=1)
        if dn_bbox_query is not None:
            reference_points_unact = paddle.concat([dn_bbox_query, topk_coords_unact],
                                                   axis=1)
        else:
            reference_points_unact = topk_coords_unact

        inter_references_list = [reference_points_unact.sigmoid().unsqueeze(0)]
        # decoder
        inter_states, inter_references = self.decoder(
            tgt=query,
            ref_points_unact=reference_points_unact,
            memory=memory,
            memory_spatial_shapes=spatial_shapes,
            memory_level_start_index=level_start_index,
            bbox_head=reg_branches,
            query_pos_head=self.query_pos_head,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
            memory_mask=mask_flatten,
        )
        inter_references_list.append(inter_references)
        inter_references_out = paddle.concat(inter_references_list)

        return inter_states, inter_references_out, topk_socre, topk_anchor, memory

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
            feat = feat.flatten(2).transpose((0, 2, 1))
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = paddle.concat(feat_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype=paddle.int64)
        level_start_index = paddle.concat((paddle.zeros(
            (1, ), dtype=paddle.int64), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = paddle.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        memory = feat_flatten
        topk_coords_unact = inverse_sigmoid(pos_anchors)
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))            
            query = pos_trans_out
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))

        inter_references_list = [pos_anchors.unsqueeze(0)]
        inter_states, inter_references = self.decoder(
            query,
            ref_points_unact=topk_coords_unact,
            memory=memory,
            memory_spatial_shapes=spatial_shapes,
            memory_level_start_index=level_start_index,
            bbox_head=reg_branches,
            query_pos_head=self.query_pos_head,
            valid_ratios=valid_ratios,
            attn_mask=None,
            memory_mask=mask_flatten)

        inter_references_list.append(inter_references)
        inter_references_out = paddle.concat(inter_references_list)

        return inter_states, inter_references_out
