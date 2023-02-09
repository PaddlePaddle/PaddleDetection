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
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
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
    cxcy, wh = paddle.split(x, 2, axis=-1)
    return paddle.concat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)


def bbox_xyxy_to_cxcywh(x):
    x1, y1, x2, y2 = x.split(4, axis=-1)
    return paddle.concat(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=-1)


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
                                   value_level_start_index, sampling_locations,
                                   attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor): [n_levels, 2]
        value_level_start_index (Tensor): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    value_list = value.split(
        value_spatial_shapes.prod(1).split(n_levels), axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
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


def get_valid_ratio(mask):
    _, H, W = paddle.shape(mask)
    valid_ratio_h = paddle.sum(mask[:, :, 0], 1) / H
    valid_ratio_w = paddle.sum(mask[:, 0, :], 1) / W
    # [b, 2]
    return paddle.stack([valid_ratio_w, valid_ratio_h], -1)


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t) for t in targets["gt_class"]]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(targets["gt_class"])
    input_query_class = paddle.full(
        [bs, max_gt_num], num_classes, dtype='int32')
    input_query_bbox = paddle.zeros([bs, max_gt_num, 4])
    pad_gt_mask = paddle.zeros([bs, max_gt_num])
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = paddle.zeros([bs, max_gt_num * 2, 1])
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = paddle.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = paddle.split(dn_positive_idx,
                                   [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()
        pad_gt_mask = pad_gt_mask.flatten()
        # half of bbox prob
        mask = paddle.rand(input_query_class.shape) < (label_noise_ratio * 0.5)
        chosen_idx = paddle.nonzero(mask * pad_gt_mask).squeeze(-1)
        # randomly put a new one here
        new_label = paddle.randint_like(
            chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class.scatter_(chosen_idx, new_label)
        input_query_class.reshape_([bs, num_denoising])
        pad_gt_mask.reshape_([bs, num_denoising])

    if box_noise_scale > 0:
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox)

        diff = paddle.tile(input_query_bbox[..., 2:] * 0.5,
                           [1, 1, 2]) * box_noise_scale

        rand_sign = paddle.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = paddle.rand(input_query_bbox.shape)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = bbox_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    class_embed = paddle.concat(
        [class_embed, paddle.zeros([1, class_embed.shape[-1]])])
    input_query_class = paddle.gather(
        class_embed, input_query_class.flatten(),
        axis=0).reshape([bs, num_denoising, -1])

    tgt_size = num_denoising + num_queries
    attn_mask = paddle.ones([tgt_size, tgt_size]) < 0
    # match query cannot see the reconstruct
    attn_mask[num_denoising:, :num_denoising] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      2 * i] = True
    attn_mask = ~attn_mask
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta


def get_sine_pos_embed(pos_tensor,
                       num_pos_feats=128,
                       temperature=10000,
                       exchange_xy=True):
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        torch.Tensor: Returned position embedding  # noqa
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2. * math.pi
    dim_t = 2. * paddle.floor_divide(
        paddle.arange(num_pos_feats), paddle.to_tensor(2))
    dim_t = scale / temperature**(dim_t / num_pos_feats)

    def sine_func(x):
        x *= dim_t
        return paddle.stack(
            (x[:, :, 0::2].sin(), x[:, :, 1::2].cos()), axis=3).flatten(2)

    pos_res = [sine_func(x) for x in pos_tensor.split(pos_tensor.shape[-1], -1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = paddle.concat(pos_res, axis=2)
    return pos_res
