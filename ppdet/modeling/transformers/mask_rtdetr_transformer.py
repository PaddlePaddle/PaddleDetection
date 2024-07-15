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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from .rtdetr_transformer import TransformerDecoderLayer
from .utils import (_get_clones, inverse_sigmoid, get_denoising_training_group,
                    mask_to_box_coordinate)
from ..heads.detr_head import MLP
from ..initializer import (linear_init_, constant_, xavier_uniform_, bias_init_with_prob)

__all__ = ['MaskRTDETR']


def _get_pred_class_and_mask(query_embed,
                             mask_feat,
                             dec_norm,
                             score_head,
                             mask_query_head):
    out_query = dec_norm(query_embed)
    out_logits = score_head(out_query)
    mask_query_embed = mask_query_head(out_query)
    batch_size, mask_dim, _ = mask_query_embed.shape
    _, _, mask_h, mask_w = mask_feat.shape
    out_mask = paddle.bmm(
        mask_query_embed, mask_feat.flatten(2)).reshape(
        [batch_size, mask_dim, mask_h, mask_w])
    return out_logits, out_mask


class MaskTransformerDecoder(nn.Layer):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 eval_idx=-1,
                 eval_topk=100):
        super(MaskTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 \
            else num_layers + eval_idx
        self.eval_topk = eval_topk

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                mask_feat,
                bbox_head,
                score_head,
                query_pos_head,
                mask_query_head,
                dec_norm,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_masks = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(
                    inverse_sigmoid(ref_points_detach))

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head(output) +
                                       inverse_sigmoid(ref_points_detach))

            if self.training:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm,
                    score_head, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head(output) +
                                  inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm,
                    score_head, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                dec_out_bboxes.append(inter_ref_bbox)
                return (paddle.stack(dec_out_bboxes),
                        paddle.stack(dec_out_logits),
                        paddle.stack(dec_out_masks))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return (paddle.stack(dec_out_bboxes),
                paddle.stack(dec_out_logits),
                paddle.stack(dec_out_masks))


@register
class MaskRTDETR(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size', 'num_prototypes']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_prototypes=32,
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.4,
                 box_noise_scale=0.4,
                 learnt_init_query=False,
                 query_pos_head_inv_sig=False,
                 mask_enhanced=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(MaskRTDETR, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.mask_enhanced = mask_enhanced
        self.eval_size = eval_size

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = MaskTransformerDecoder(hidden_dim, decoder_layer,
                                              num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim,
                                  hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

        # mask embedding
        self.mask_query_head = MLP(hidden_dim, hidden_dim,
                                   num_prototypes, num_layers=3)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))

        # decoder norm layer
        self.dec_norm = nn.LayerNorm(
            hidden_dim,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # shared prediction head
        self.score_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.score_head)
        constant_(self.score_head.bias, bias_cls)
        constant_(self.bbox_head.layers[-1].weight)
        constant_(self.bbox_head.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape],
                'feat_strides': [i.stride for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)),
                    ('norm', nn.BatchNorm2D(
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
                        padding=1,
                        bias_attr=False)),
                    ('norm', nn.BatchNorm2D(
                        self.hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
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
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def forward(self, feats, pad_mask=None, gt_meta=None, is_teacher=False):
        enc_feats, mask_feat = feats
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(enc_feats)

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
            denoising_class, denoising_bbox_unact,\
                attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_out, init_out = \
            self._get_decoder_input(
                memory, mask_feat, spatial_shapes,
                denoising_class, denoising_bbox_unact, is_teacher)

        # decoder
        out_bboxes, out_logits, out_masks = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            mask_feat,
            self.bbox_head,
            self.score_head,
            self.query_pos_head,
            self.mask_query_head,
            self.dec_norm,
            attn_mask=attn_mask,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig)

        return out_logits, out_bboxes, out_masks, enc_out, init_out, dn_meta

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=paddle.float32):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           mask_feat,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           is_teacher=False):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None or is_teacher:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_logits_unact = self.score_head(output_memory)
        enc_bboxes_unact = self.bbox_head(output_memory) + anchors

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
        enc_out_logits, enc_out_masks = _get_pred_class_and_mask(
            target, mask_feat, self.dec_norm,
            self.score_head, self.mask_query_head)
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

        # direct prediction from the matching and denoising part in the beginning
        if self.training and denoising_class is not None:
            init_out_logits, init_out_masks = _get_pred_class_and_mask(
                target, mask_feat, self.dec_norm,
                self.score_head, self.mask_query_head)
            init_out_bboxes = F.sigmoid(reference_points_unact)
            init_out = (init_out_logits, init_out_bboxes, init_out_masks)
        else:
            init_out = None

        return target, reference_points_unact.detach(), enc_out, init_out
