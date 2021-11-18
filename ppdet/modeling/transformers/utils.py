# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..bbox_utils import bbox_overlaps

__all__ = [
    '_get_clones', 'bbox_overlaps', 'bbox_cxcywh_to_xyxy',
    'bbox_xyxy_to_cxcywh', 'sigmoid_focal_loss', 'inverse_sigmoid',
    'deformable_attention_core_func'
]


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


def bbox_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)


def bbox_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return paddle.stack(b, axis=-1)


def sigmoid_focal_loss(logit, label, normalizer=1.0, alpha=0.25, gamma=2.0):
    prob = F.sigmoid(logit)
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    return loss.mean(1).sum() / normalizer


def inverse_sigmoid(x, eps=1e-6):
    x = x.clip(min=0., max=1.)
    return paddle.log(x / (1 - x + eps) + eps)


def deformable_attention_core_func(value, value_spatial_shapes,
                                   sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor): [n_levels, 2]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, Len_v, n_head, c = value.shape
    _, Len_q, n_head, n_levels, n_points, _ = sampling_locations.shape

    value_list = value.split(value_spatial_shapes.prod(1).tolist(), axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes.tolist()):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(
            [0, 2, 1]).reshape([bs * n_head, c, h, w])
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(
            [0, 2, 1, 3, 4]).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
        [bs * n_head, 1, Len_q, n_levels * n_points])
    output = (paddle.stack(
        sampling_value_list, axis=-2).flatten(-2) *
              attention_weights).sum(-1).reshape([bs, n_head * c, Len_q])

    return output.transpose([0, 2, 1])
