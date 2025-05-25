# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from D-FINE (https://github.com/Peterande/D-FINE)
# Copyright 2024 The IDEA Authors. All rights reserved.
# Modified from DEIM (https://github.com/ShihuaHuang95/DEIM)
# Copyright 2025 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ..heads.detr_head import MLP
from ..initializer import (linear_init_, constant_, xavier_uniform_, bias_init_with_prob)
from ..layers import MultiHeadAttention
from .rtdetr_transformerv2 import MSDeformableAttention
from .utils import _get_clones, bbox_xyxy_to_cxcywh, get_contrastive_denoising_training_group, inverse_sigmoid

__all__ = ['DFINETransformer']


@functools.lru_cache
def weighting_function(reg_max: int, up: float, reg_scale: float):
    """Generates the non-uniform Weighting Function W(n) for bounding box
    regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (float): Controls upper bounds of the sequence,
            where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
            Larger values result in flatter weights near the central axis
            W(reg_max/2)=0 and steeper weights at both ends.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up) * abs(reg_scale)
    upper_bound2 = abs(up) * abs(reg_scale) * 2
    step = (upper_bound1 + 1)**(2 / (reg_max - 2))
    left_values = [-(step)**i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step)**i - 1 for i in range(1, reg_max // 2)]
    project = [-upper_bound2, *left_values, 0, *right_values, upper_bound2]
    return paddle.to_tensor(np.array(project), dtype='float32')


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape([-1])
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = paddle.sum(mask, 1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.to("float32")

    weight_right = paddle.zeros_like(indices)
    weight_left = paddle.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].to("int64")

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    gt_valid = gt[valid_idx_mask]
    left_diffs = paddle.abs(gt_valid - left_values)
    right_diffs = paddle.abs(right_values - gt_valid)

    # Valid weights
    valid_idx = paddle.nonzero(valid_idx_mask).flatten()
    weight_right = paddle.scatter(weight_right, valid_idx, left_diffs / (left_diffs + right_diffs))
    weight_left = paddle.scatter(weight_left, valid_idx, 1.0 - weight_right[valid_idx_mask])

    # Invalid weights (out of range)
    invalid_idx_mask_neg = (indices < 0)
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = (indices >= reg_max)
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale)
    left   = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top    = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right  = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = paddle.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = paddle.clip(four_lens, min=0, max=reg_max - eps)
    return four_lens.reshape([-1]).detach(), weight_right.detach(), weight_left.detach()


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = paddle.stack([x1, y1, x2, y2], -1)

    return bbox_xyxy_to_cxcywh(bboxes)


class Integral(nn.Layer):
    def __init__(self, reg_max=32, reg_scale=4.0):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        project = weighting_function(self.reg_max, 0.5, reg_scale)
        self.register_buffer('project',
                             paddle.to_tensor(project, dtype=paddle.float32))

    def forward(self, x):
        shape = x.shape
        x = F.softmax(x.reshape([-1, self.reg_max + 1]), 1)
        x = F.linear(x, self.project).reshape([-1, 4])
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Layer):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        constant_(self.reg_conf.layers[-1].weight)
        constant_(self.reg_conf.layers[-1].bias)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.shape
        prob = F.softmax(pred_corners.reshape([B, L, 4, self.reg_max + 1]), -1)
        prob_topk, _ = prob.topk(self.k, -1)
        stat = paddle.concat([prob_topk, prob_topk.mean(-1, keepdim=True)], -1)
        quality_score = self.reg_conf(stat.reshape([B, L, -1]))
        return scores + quality_score


class Gate(nn.Layer):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        constant_(self.gate.bias, bias)
        constant_(self.gate.weight)

    def forward(self, x1, x2):
        gate_input = paddle.concat([x1, x2], -1)
        gates = F.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, -1)
        return gate1 * x1 + gate2 * x2


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 sampling_method='default',
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points,
            sampling_method=sampling_method, lr_mult=1.0)
        self.cross_attn.value_proj = nn.Identity()  # diff
        self.cross_attn.output_proj = nn.Identity()  # diff

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

        self.gateway = Gate(d_model)
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
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_mask)
        # tgt = tgt + self.dropout2(tgt2)
        tgt = self.gateway(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt.clip(min=-65504, max=65504))

        return tgt


class TransformerDecoder(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, reg_max, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.lqe_layers = _get_clones(LQE(4, 64, 2, reg_max), num_layers)

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        output = tgt
        output_detach = pred_corners_undetach = 0
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(
                    inverse_sigmoid(ref_points_detach))

            query_pos_embed = paddle.clip(query_pos_embed, min=-10, max=10)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            if i == 0 :
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(
                    ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(paddle.concat([
                    inter_ref_bbox[..., :2], inter_ref_bbox[..., 2:].clip(min=0)], -1))
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training or i == len(self.layers) - 1:
                    break

            output_detach = output.detach()
            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits), \
               paddle.stack(dec_out_pred_corners), paddle.stack(dec_out_refs), pre_bboxes, pre_scores


@register
class DFINETransformer(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 query_pos_head_inv_sig=False,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 mlp_act='relu',
                 cross_attn_sampling_method='default',
                 reg_max=32,
                 reg_scale=4.0):
        super(DFINETransformer, self).__init__()
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
        self.eval_size = eval_size
        self.reg_scale = reg_scale

        assert cross_attn_sampling_method in ['default', 'discrete'], NotImplementedError
        self.cross_attn_sampling_method = cross_attn_sampling_method

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points, sampling_method=cross_attn_sampling_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, reg_max, eval_idx)

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

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2, act=mlp_act)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3, act=mlp_act)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), num_layers=3, act=mlp_act)
            for _ in range(num_decoder_layers)
        ])

        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.integral = Integral(reg_max, reg_scale)

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)

        constant_(self.pre_bbox_head.layers[-1].weight)
        constant_(self.pre_bbox_head.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            if not isinstance(l, nn.Identity):
                xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        ('conv', nn.Conv2D(
                            in_channels,
                            self.hidden_dim,
                            kernel_size=1,
                            bias_attr=False)), ('norm', nn.BatchNorm2D(
                                self.hidden_dim,
                                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        ('conv', nn.Conv2D(
                            in_channels,
                            self.hidden_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias_attr=False)), ('norm', nn.BatchNorm2D(
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
        return (feat_flatten, spatial_shapes, level_start_index)

    def forward(self, feats, pad_mask=None, gt_meta=None, is_teacher=False):
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact,is_teacher)

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.reg_scale,
            attn_mask=attn_mask,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig)
        return (out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits,
                enc_topk_bboxes, enc_topk_logits, dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
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
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
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

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits
