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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from .position_encoding import PositionEmbedding
from ..heads.detr_head import MLP
from .deformable_transformer import (DeformableTransformerEncoderLayer,
                                     DeformableTransformerEncoder)
from .dino_transformer import (DINOTransformerDecoderLayer)
from ..initializer import (linear_init_, constant_, xavier_uniform_,
                           bias_init_with_prob)
from .utils import (_get_clones, get_valid_ratio, get_denoising_training_group,
                    get_sine_pos_embed, inverse_sigmoid, mask_to_box_coordinate)

__all__ = ['MaskDINO']


class ConvGNBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 num_groups=32,
                 bias=False,
                 act=None):
        super(ConvGNBlock, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias_attr=bias)
        self.norm = nn.GroupNorm(
            num_groups,
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = getattr(F, act) if act is not None else None

        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.norm(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x


class MaskDINOTransformerDecoder(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers):
        super(MaskDINOTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                query_pos_head,
                dec_norm,
                valid_ratios=None,
                attn_mask=None,
                memory_mask=None):
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [memory.shape[0], memory_spatial_shapes.shape[0], 2])

        output = tgt
        intermediate = []
        inter_bboxes = []
        ref_points = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            reference_points_input = ref_points.detach().unsqueeze(
                2) * valid_ratios.tile([1, 1, 2]).unsqueeze(1)
            query_pos_embed = get_sine_pos_embed(
                reference_points_input[..., 0, :], self.hidden_dim // 2)
            query_pos_embed = query_pos_head(query_pos_embed)

            output = layer(output, reference_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            ref_points = F.sigmoid(
                bbox_head(output) + inverse_sigmoid(ref_points.detach()))

            intermediate.append(dec_norm(output))
            inter_bboxes.append(ref_points)

        return paddle.stack(intermediate), paddle.stack(inter_bboxes)


@register
class MaskDINO(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 in_feats_channel=[256, 512, 1024, 2048],
                 num_levels=3,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=9,
                 enc_dim_feedforward=1024,
                 dec_dim_feedforward=2048,
                 dropout=0.,
                 activation="relu",
                 lr_mult=1.0,
                 pe_temperature=10000,
                 pe_offset=-0.5,
                 num_denoising=100,
                 label_noise_ratio=0.4,
                 box_noise_scale=0.4,
                 learnt_init_query=False,
                 mask_enhanced=True,
                 eps=1e-2):
        super(MaskDINO, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        feat0_dim = in_feats_channel.pop(0)
        assert len(in_feats_channel) <= num_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.mask_enhanced = mask_enhanced

        weight_attr = ParamAttr(regularizer=L2Decay(0.0))
        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        # backbone feature projection
        self._build_input_proj_layer(in_feats_channel, weight_attr, bias_attr)

        # Transformer module
        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, nhead, enc_dim_feedforward, dropout, activation,
            num_levels, num_encoder_points, lr_mult, weight_attr, bias_attr)
        self.encoder = DeformableTransformerEncoder(encoder_layer,
                                                    num_encoder_layers)
        decoder_layer = DINOTransformerDecoderLayer(
            hidden_dim, nhead, dec_dim_feedforward, dropout, activation,
            num_levels, num_decoder_points, lr_mult, weight_attr, bias_attr)
        self.decoder = MaskDINOTransformerDecoder(hidden_dim, decoder_layer,
                                                  num_decoder_layers)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # position embedding
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            temperature=pe_temperature,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=pe_offset)
        self.level_embed = nn.Embedding(
            num_levels,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(2 * hidden_dim,
                                  hidden_dim,
                                  hidden_dim,
                                  num_layers=2)
        # mask embedding
        self.mask_query_head = MLP(hidden_dim,
                                   hidden_dim,
                                   hidden_dim,
                                   num_layers=3)

        # encoder mask head
        self.enc_mask_lateral = ConvGNBlock(feat0_dim, hidden_dim, 1)
        self.enc_mask_output = nn.Sequential(
            ConvGNBlock(
                hidden_dim, hidden_dim, 3, act=activation),
            nn.Conv2D(hidden_dim, hidden_dim, 1))
        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim, weight_attr=weight_attr, bias_attr=bias_attr))
        # decoder norm layer
        self.dec_norm = nn.LayerNorm(
            hidden_dim, weight_attr=weight_attr, bias_attr=bias_attr)
        # shared prediction head
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.class_head)
        constant_(self.class_head.bias, bias_cls)
        constant_(self.bbox_head.layers[-1].weight)
        constant_(self.bbox_head.layers[-1].bias)

        xavier_uniform_(self.enc_mask_output[1].weight)
        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_feats_channel': [i.channels for i in input_shape], }

    def _build_input_proj_layer(self,
                                in_feats_channel,
                                weight_attr=None,
                                bias_attr=None):
        self.input_proj = nn.LayerList()
        for in_channels in in_feats_channel:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels, self.hidden_dim, kernel_size=1)), (
                            'norm', nn.GroupNorm(
                                32,
                                self.hidden_dim,
                                weight_attr=weight_attr,
                                bias_attr=bias_attr))))
        in_channels = in_feats_channel[-1]
        for _ in range(self.num_levels - len(in_feats_channel)):
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
                            weight_attr=weight_attr,
                            bias_attr=bias_attr))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats, pad_mask=None):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

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
            lvl_pos_embed = pos_embed + self.level_embed.weight[i]
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
        # [l], 每一个level的起始index
        level_start_index = paddle.concat([
            paddle.zeros(
                [1], dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        # [b, num_levels, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)
        return (feat_flatten, spatial_shapes, level_start_index, mask_flatten,
                lvl_pos_embed_flatten, valid_ratios)

    def forward(self, feats, pad_mask=None, gt_meta=None):
        feat0 = feats.pop(0)
        # input projection and embedding
        (feat_flatten, spatial_shapes, level_start_index, mask_flatten,
         lvl_pos_embed_flatten,
         valid_ratios) = self._get_encoder_input(feats, pad_mask)

        # encoder
        memory = self.encoder(feat_flatten, spatial_shapes, level_start_index,
                              mask_flatten, lvl_pos_embed_flatten, valid_ratios)

        mask_feat = self._get_encoder_mask_feature(feat0, memory,
                                                   spatial_shapes)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_out, init_out = \
            self._get_decoder_input(
            memory, mask_feat, spatial_shapes, mask_flatten, denoising_class,
            denoising_bbox_unact)

        # decoder
        inter_feats, inter_bboxes = self.decoder(
            target, init_ref_points_unact, memory, spatial_shapes,
            level_start_index, self.bbox_head, self.query_pos_head,
            self.dec_norm, valid_ratios, attn_mask, mask_flatten)

        out_logits = []
        out_bboxes = []
        out_masks = []
        for i in range(self.num_decoder_layers):
            if self.training or i == self.num_decoder_layers - 1:
                logits_, masks_ = self._get_pred_class_and_mask(inter_feats[i],
                                                                mask_feat)
            else:
                continue
            out_logits.append(logits_)
            out_masks.append(masks_)
            if i == 0:
                out_bboxes.append(
                    F.sigmoid(
                        self.bbox_head(inter_feats[i]) + init_ref_points_unact))
            else:
                out_bboxes.append(
                    F.sigmoid(
                        self.bbox_head(inter_feats[i]) + inverse_sigmoid(
                            inter_bboxes[i - 1])))
        out_bboxes = paddle.stack(out_bboxes)
        out_logits = paddle.stack(out_logits)
        out_masks = paddle.stack(out_masks)

        return (out_logits, out_bboxes, out_masks, enc_out, init_out, dn_meta)

    def _get_encoder_mask_feature(self, in_feat, memory, spatial_shapes):
        memory_feat0 = memory.split(
            spatial_shapes.prod(1).split(self.num_levels), axis=1)[0]
        h, w = spatial_shapes[0]
        memory_feat0 = memory_feat0.reshape(
            [0, h, w, self.hidden_dim]).transpose([0, 3, 1, 2])
        out = self.enc_mask_lateral(in_feat) + F.interpolate(
            memory_feat0,
            scale_factor=2.0,
            mode='bilinear',
            align_corners=False)
        return self.enc_mask_output(out)

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
                paddle.arange(end=h), paddle.arange(end=w))
            grid_xy = paddle.stack([grid_x, grid_y], -1).astype(memory.dtype)

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
        output_memory = self.enc_output(memory)
        return output_memory, output_anchors

    def _get_decoder_input(self,
                           memory,
                           mask_feat,
                           spatial_shapes,
                           memory_mask=None,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        # prepare input for decoder
        bs, _, _ = memory.shape
        output_memory, output_anchors = self._get_encoder_output_anchors(
            memory, spatial_shapes, memory_mask)
        enc_logits_unact = self.class_head(output_memory)
        enc_bboxes_unact = self.bbox_head(output_memory) + output_anchors

        # get topk index
        _, topk_ind = paddle.topk(
            enc_logits_unact.max(-1), self.num_queries, axis=1)
        batch_ind = paddle.arange(end=bs).astype(topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        # extract content and position query embedding
        target = paddle.gather_nd(output_memory, topk_ind)
        reference_points_unact = paddle.gather_nd(enc_bboxes_unact,
                                                  topk_ind)  # unsigmoided.
        # get encoder output: {logits, bboxes, masks}
        enc_out_logits, enc_out_masks = self._get_pred_class_and_mask(target,
                                                                      mask_feat)
        enc_out_bboxes = F.sigmoid(reference_points_unact)
        enc_out = (enc_out_logits, enc_out_bboxes, enc_out_masks)

        # concat denoising query
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)
        if self.mask_enhanced:
            # use mask-enhanced anchor box initialization
            reference_points = mask_to_box_coordinate(
                enc_out_masks > 0, normalize=True, format="xywh")
            reference_points_unact = inverse_sigmoid(reference_points)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)

        # direct prediction from the matching and denoising part in the begining
        if self.training and denoising_class is not None:
            init_out_logits, init_out_masks = self._get_pred_class_and_mask(
                target, mask_feat)
            init_out_bboxes = F.sigmoid(reference_points_unact)
            init_out = (init_out_logits, init_out_bboxes, init_out_masks)
        else:
            init_out = None

        return target, reference_points_unact.detach(), enc_out, init_out

    def _get_pred_class_and_mask(self, query_embed, mask_feat):
        out_query = self.dec_norm(query_embed)
        out_logits = self.class_head(out_query)
        mask_query_embed = self.mask_query_head(out_query)
        _, _, h, w = mask_feat.shape
        # [b, q, c] x [b, c, h, w] -> [b, q, h, w]
        out_mask = paddle.bmm(mask_query_embed, mask_feat.flatten(2)).reshape(
            [0, 0, h, w])
        return out_logits, out_mask
