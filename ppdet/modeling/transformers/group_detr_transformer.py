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
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from ..heads.detr_head import MLP
from .deformable_transformer import MSDeformableAttention
from ..initializer import (linear_init_, constant_, xavier_uniform_, normal_,
                           bias_init_with_prob)
from .utils import (_get_clones, get_valid_ratio,
                    get_contrastive_denoising_training_group,
                    get_sine_pos_embed, inverse_sigmoid)

__all__ = ['GroupDINOTransformer']


class DINOTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(DINOTransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                               n_points, 1.0)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
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


class DINOTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super(DINOTransformerEncoder, self).__init__()
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


class DINOTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 dual_queries=False,
                 dual_groups=0,
                 weight_attr=None,
                 bias_attr=None):
        super(DINOTransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # for dual groups 
        self.dual_queries = dual_queries
        self.dual_groups = dual_groups
        self.n_head = n_head

        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if self.dual_queries:
            dual_groups = self.dual_groups
            bs, num_queries, n_model = q.shape
            q = paddle.concat(q.split(dual_groups + 1, axis=1), axis=0)
            k = paddle.concat(k.split(dual_groups + 1, axis=1), axis=0)
            tgt = paddle.concat(tgt.split(dual_groups + 1, axis=1), axis=0)

            g_num_queries = num_queries // (dual_groups + 1)
            if attn_mask is None or attn_mask[0] is None:
                attn_mask = None
            else:
                # [(dual_groups + 1), g_num_queries, g_num_queries]
                attn_mask = paddle.concat(
                    [sa_mask.unsqueeze(0) for sa_mask in attn_mask], axis=0)
                # [1, (dual_groups + 1), 1, g_num_queries, g_num_queries]
                # --> [bs, (dual_groups + 1), nhead, g_num_queries, g_num_queries]
                # --> [bs * (dual_groups + 1), nhead, g_num_queries, g_num_queries]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(2).tile(
                    [bs, 1, self.n_head, 1, 1])
                attn_mask = attn_mask.reshape([
                    bs * (dual_groups + 1), self.n_head, g_num_queries,
                    g_num_queries
                ])

        if attn_mask is not None:
            attn_mask = attn_mask.astype('bool')

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)

        # trace back
        if self.dual_queries:
            tgt = paddle.concat(tgt.split(dual_groups + 1, axis=0), axis=1)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DINOTransformerDecoder(nn.Layer):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 return_intermediate=True):
        super(DINOTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.norm = nn.LayerNorm(
            hidden_dim,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                query_pos_head,
                valid_ratios=None,
                attn_mask=None,
                memory_mask=None):
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [memory.shape[0], memory_spatial_shapes.shape[0], 2])

        output = tgt
        intermediate = []
        inter_ref_bboxes = []
        for i, layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(
                2) * valid_ratios.tile([1, 1, 2]).unsqueeze(1)
            query_pos_embed = get_sine_pos_embed(
                reference_points_input[..., 0, :], self.hidden_dim // 2)
            query_pos_embed = query_pos_head(query_pos_embed)

            output = layer(output, reference_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                reference_points))

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                inter_ref_bboxes.append(inter_ref_bbox)

            reference_points = inter_ref_bbox.detach()

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(inter_ref_bboxes)

        return output, reference_points


@register
class GroupDINOTransformer(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=900,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_feat_channels=[512, 1024, 2048],
                 num_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 pe_temperature=10000,
                 pe_offset=-0.5,
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 use_input_proj=True,
                 dual_queries=False,
                 dual_groups=0,
                 eps=1e-2):
        super(GroupDINOTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.use_input_proj = use_input_proj

        if use_input_proj:
            # backbone feature projection
            self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        encoder_layer = DINOTransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_encoder_points)
        self.encoder = DINOTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DINOTransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_decoder_points,
            dual_queries=dual_queries,
            dual_groups=dual_groups)
        self.decoder = DINOTransformerDecoder(hidden_dim, decoder_layer,
                                              num_decoder_layers,
                                              return_intermediate_dec)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # for dual group
        self.dual_queries = dual_queries
        self.dual_groups = dual_groups
        if self.dual_queries:
            self.denoising_class_embed_groups = nn.LayerList([
                nn.Embedding(
                    num_classes,
                    hidden_dim,
                    weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
                for _ in range(self.dual_groups)
            ])

        # position embedding
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            temperature=pe_temperature,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=pe_offset)
        self.level_embed = nn.Embedding(num_levels, hidden_dim)
        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
            normal_(self.tgt_embed.weight)
            if self.dual_queries:
                self.tgt_embed_dual = nn.LayerList([
                    nn.Embedding(num_queries, hidden_dim)
                    for _ in range(self.dual_groups)
                ])
                for dual_tgt_module in self.tgt_embed_dual:
                    normal_(dual_tgt_module.weight)
        self.query_pos_head = MLP(2 * hidden_dim,
                                  hidden_dim,
                                  hidden_dim,
                                  num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        if self.dual_queries:
            self.enc_output = _get_clones(self.enc_output, self.dual_groups + 1)
        else:
            self.enc_output = _get_clones(self.enc_output, 1)

        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        if self.dual_queries:
            self.enc_bbox_head_dq = nn.LayerList([
                MLP(hidden_dim, hidden_dim, 4, num_layers=3)
                for i in range(self.dual_groups)
            ])
            self.enc_score_head_dq = nn.LayerList([
                nn.Linear(hidden_dim, num_classes)
                for i in range(self.dual_groups)
            ])

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        for enc_output in self.enc_output:
            linear_init_(enc_output[0])
            xavier_uniform_(enc_output[0].weight)
        normal_(self.level_embed.weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        normal_(self.denoising_class_embed.weight)
        if self.use_input_proj:
            for l in self.input_proj:
                xavier_uniform_(l[0].weight)
                constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape], }

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels, self.hidden_dim, kernel_size=1)),
                    ('norm', nn.GroupNorm(
                        32,
                        self.hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1)), ('norm', nn.GroupNorm(
                            32,
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats, pad_mask=None):
        if self.use_input_proj:
            # get projection features
            proj_feats = [
                self.input_proj[i](feat) for i, feat in enumerate(feats)
            ]
            if self.num_levels > len(proj_feats):
                len_srcs = len(proj_feats)
                for i in range(len_srcs, self.num_levels):
                    if i == len_srcs:
                        proj_feats.append(self.input_proj[i](feats[-1]))
                    else:
                        proj_feats.append(self.input_proj[i](proj_feats[-1]))
        else:
            proj_feats = feats
        # get encoder inputs
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for i, feat in enumerate(proj_feats):
            bs, _, h, w = feat.shape
            spatial_shapes.append(paddle.concat([h, w]))
            # [b,c,h,w] -> [b,h*w,c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            if pad_mask is not None:
                mask = F.interpolate(pad_mask.unsqueeze(0), size=(h, w))[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(get_valid_ratio(mask))
            # [b, h*w, c]
            pos_embed = self.position_embedding(mask).flatten(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[i].reshape(
                [1, 1, -1])
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            if pad_mask is not None:
                # [b, h*w]
                mask_flatten.append(mask.flatten(1))

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        # [b, l]
        mask_flatten = None if pad_mask is None else paddle.concat(mask_flatten,
                                                                   1)
        # [b, l, c]
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        # [num_levels, 2]
        spatial_shapes = paddle.to_tensor(
            paddle.stack(spatial_shapes).astype('int64'))
        # [l] start index of each level
        level_start_index = paddle.concat([
            paddle.zeros(
                [1], dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        # [b, num_levels, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)
        return (feat_flatten, spatial_shapes, level_start_index, mask_flatten,
                lvl_pos_embed_flatten, valid_ratios)

    def forward(self, feats, pad_mask=None, gt_meta=None):
        # input projection and embedding
        (feat_flatten, spatial_shapes, level_start_index, mask_flatten,
         lvl_pos_embed_flatten,
         valid_ratios) = self._get_encoder_input(feats, pad_mask)

        # encoder
        memory = self.encoder(feat_flatten, spatial_shapes, level_start_index,
                              mask_flatten, lvl_pos_embed_flatten, valid_ratios)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
            if self.dual_queries:
                denoising_class_groups = []
                denoising_bbox_groups = []
                attn_mask_groups = []
                dn_meta_groups = []
                for g_id in range(self.dual_groups):
                    denoising_class_gid, denoising_bbox_gid, attn_mask_gid, dn_meta_gid = \
                        get_contrastive_denoising_training_group(gt_meta,
                                                    self.num_classes,
                                                    self.num_queries,
                                                    self.denoising_class_embed_groups[g_id].weight,
                                                    self.num_denoising,
                                                    self.label_noise_ratio,
                                                    self.box_noise_scale)
                    denoising_class_groups.append(denoising_class_gid)
                    denoising_bbox_groups.append(denoising_bbox_gid)
                    attn_mask_groups.append(attn_mask_gid)
                    dn_meta_groups.append(dn_meta_gid)

                # combine
                denoising_class = [denoising_class] + denoising_class_groups
                denoising_bbox = [denoising_bbox] + denoising_bbox_groups
                attn_mask = [attn_mask] + attn_mask_groups
                dn_meta = [dn_meta] + dn_meta_groups
        else:
            denoising_class, denoising_bbox, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, mask_flatten, denoising_class,
            denoising_bbox)

        # decoder
        inter_feats, inter_ref_bboxes = self.decoder(
            target, init_ref_points, memory, spatial_shapes, level_start_index,
            self.dec_bbox_head, self.query_pos_head, valid_ratios, attn_mask,
            mask_flatten)
        # solve hang during distributed training
        inter_feats[0] += self.denoising_class_embed.weight[0, 0] * 0.
        if self.dual_queries:
            for g_id in range(self.dual_groups):
                inter_feats[0] += self.denoising_class_embed_groups[
                    g_id].weight[0, 0] * 0.0

        out_bboxes = []
        out_logits = []
        for i in range(self.num_decoder_layers):
            out_logits.append(self.dec_score_head[i](inter_feats[i]))
            if i == 0:
                out_bboxes.append(
                    F.sigmoid(self.dec_bbox_head[i](inter_feats[i]) +
                              inverse_sigmoid(init_ref_points)))
            else:
                out_bboxes.append(
                    F.sigmoid(self.dec_bbox_head[i](inter_feats[i]) +
                              inverse_sigmoid(inter_ref_bboxes[i - 1])))

        out_bboxes = paddle.stack(out_bboxes)
        out_logits = paddle.stack(out_logits)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _get_encoder_output_anchors(self,
                                    memory,
                                    spatial_shapes,
                                    memory_mask=None,
                                    grid_size=0.05):
        output_anchors = []
        idx = 0
        for lvl, (h, w) in enumerate(spatial_shapes):
            if memory_mask is not None:
                mask_ = memory_mask[:, idx:idx + h * w].reshape([-1, h, w])
                valid_H = paddle.sum(mask_[:, :, 0], 1)
                valid_W = paddle.sum(mask_[:, 0, :], 1)
            else:
                valid_H, valid_W = h, w

            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=memory.dtype),
                paddle.arange(
                    end=w, dtype=memory.dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.stack([valid_W, valid_H], -1).reshape(
                [-1, 1, 1, 2]).astype(grid_xy.dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            output_anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))
            idx += h * w

        output_anchors = paddle.concat(output_anchors, 1)
        valid_mask = ((output_anchors > self.eps) *
                      (output_anchors < 1 - self.eps)).all(-1, keepdim=True)
        output_anchors = paddle.log(output_anchors / (1 - output_anchors))
        if memory_mask is not None:
            valid_mask = (valid_mask * (memory_mask.unsqueeze(-1) > 0)) > 0
        output_anchors = paddle.where(valid_mask, output_anchors,
                                      paddle.to_tensor(float("inf")))

        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        if self.dual_queries:
            output_memory = [
                self.enc_output[g_id](memory)
                for g_id in range(self.dual_groups + 1)
            ]
        else:
            output_memory = self.enc_output[0](memory)
        return output_memory, output_anchors

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           memory_mask=None,
                           denoising_class=None,
                           denoising_bbox=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        output_memory, output_anchors = self._get_encoder_output_anchors(
            memory, spatial_shapes, memory_mask)
        if self.dual_queries:
            enc_outputs_class = self.enc_score_head(output_memory[0])
            enc_outputs_coord_unact = self.enc_bbox_head(output_memory[
                0]) + output_anchors
        else:
            enc_outputs_class = self.enc_score_head(output_memory)
            enc_outputs_coord_unact = self.enc_bbox_head(
                output_memory) + output_anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)
        topk_coords_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                             topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(topk_coords_unact)
        reference_points = enc_topk_bboxes.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        if self.dual_queries:
            enc_topk_logits_groups = []
            enc_topk_bboxes_groups = []
            reference_points_groups = []
            topk_ind_groups = []
            for g_id in range(self.dual_groups):
                enc_outputs_class_gid = self.enc_score_head_dq[g_id](
                    output_memory[g_id + 1])
                enc_outputs_coord_unact_gid = self.enc_bbox_head_dq[g_id](
                    output_memory[g_id + 1]) + output_anchors
                _, topk_ind_gid = paddle.topk(
                    enc_outputs_class_gid.max(-1), self.num_queries, axis=1)
                # extract region proposal boxes
                batch_ind = paddle.arange(end=bs, dtype=topk_ind_gid.dtype)
                batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
                topk_ind_gid = paddle.stack([batch_ind, topk_ind_gid], axis=-1)
                topk_coords_unact_gid = paddle.gather_nd(
                    enc_outputs_coord_unact_gid, topk_ind_gid)  # unsigmoided.
                enc_topk_bboxes_gid = F.sigmoid(topk_coords_unact_gid)
                reference_points_gid = enc_topk_bboxes_gid.detach()
                enc_topk_logits_gid = paddle.gather_nd(enc_outputs_class_gid,
                                                       topk_ind_gid)

                # append and combine
                topk_ind_groups.append(topk_ind_gid)
                enc_topk_logits_groups.append(enc_topk_logits_gid)
                enc_topk_bboxes_groups.append(enc_topk_bboxes_gid)
                reference_points_groups.append(reference_points_gid)

            enc_topk_bboxes = paddle.concat(
                [enc_topk_bboxes] + enc_topk_bboxes_groups, 1)
            enc_topk_logits = paddle.concat(
                [enc_topk_logits] + enc_topk_logits_groups, 1)
            reference_points = paddle.concat(
                [reference_points] + reference_points_groups, 1)
            topk_ind = paddle.concat([topk_ind] + topk_ind_groups, 1)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
            if self.dual_queries:
                target = paddle.concat([target] + [
                    self.tgt_embed_dual[g_id].weight.unsqueeze(0).tile(
                        [bs, 1, 1]) for g_id in range(self.dual_groups)
                ], 1)
        else:
            if self.dual_queries:
                target = paddle.gather_nd(output_memory[0], topk_ind)
                target_groups = []
                for g_id in range(self.dual_groups):
                    target_gid = paddle.gather_nd(output_memory[g_id + 1],
                                                  topk_ind_groups[g_id])
                    target_groups.append(target_gid)
                target = paddle.concat([target] + target_groups, 1).detach()
            else:
                target = paddle.gather_nd(output_memory, topk_ind).detach()

        if denoising_bbox is not None:
            if isinstance(denoising_bbox, list) and isinstance(
                    denoising_class, list) and self.dual_queries:
                if denoising_bbox[0] is not None:
                    reference_points_list = paddle.split(
                        reference_points, self.dual_groups + 1, axis=1)
                    reference_points = paddle.concat(
                        [
                            paddle.concat(
                                [ref, ref_], axis=1)
                            for ref, ref_ in zip(denoising_bbox,
                                                 reference_points_list)
                        ],
                        axis=1)

                    target_list = paddle.split(
                        target, self.dual_groups + 1, axis=1)
                    target = paddle.concat(
                        [
                            paddle.concat(
                                [tgt, tgt_], axis=1)
                            for tgt, tgt_ in zip(denoising_class, target_list)
                        ],
                        axis=1)
                else:
                    reference_points, target = reference_points, target
            else:
                reference_points = paddle.concat(
                    [denoising_bbox, reference_points], 1)
                target = paddle.concat([denoising_class, target], 1)

        return target, reference_points, enc_topk_bboxes, enc_topk_logits
