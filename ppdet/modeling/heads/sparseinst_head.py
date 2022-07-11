# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant

from ..layers import Conv2d
from ..initializer import kaiming_uniform_, bias_init_with_prob

from ppdet.core.workspace import register

from six.moves import zip
import numpy as np

__all__ = ['BaseIAMDecoder', 'GroupIAMDecoder']


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


class InstanceBranch(nn.Layer):
    def __init__(self, dim, num_convs, num_masks, kernel_dim, num_classes,
                 in_channels):
        super().__init__()

        bias_value = bias_init_with_prob(0.01)

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2D(
            dim,
            num_masks,
            3,
            padding=1,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.),
            bias_attr=ParamAttr(initializer=Constant(value=bias_value)))

        # outputs
        self.cls_score = nn.Linear(
            dim,
            num_classes,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.),
            bias_attr=ParamAttr(initializer=Constant(value=bias_value)))
        self.mask_kernel = nn.Linear(
            dim,
            kernel_dim,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.))
        self.objectness = nn.Linear(dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        kaiming_uniform_(self.objectness.weight)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = F.sigmoid(iam)

        B, N = iam_prob.shape[:2]
        C = features.shape[1]
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.reshape((B, N, -1))
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = paddle.bmm(iam_prob,
                                   features.reshape((B, C, -1)).transpose(
                                       (0, 2, 1)))
        normalizer = paddle.clip(iam_prob.sum(-1), min=1e-6)
        inst_features = inst_features / normalizer[:, :, None]
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


class MaskBranch(nn.Layer):
    def __init__(self, dim, num_convs, kernel_dim, in_channels):
        super().__init__()
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = Conv2d(dim, kernel_dim, kernel_size=1)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


@register
class BaseIAMDecoder(nn.Layer):
    def __init__(self,
                 num_masks,
                 num_classes,
                 in_channels,
                 inst_dim,
                 mask_dim,
                 inst_num_convs,
                 mask_num_convs,
                 kernel_dim,
                 scale_factor,
                 output_iam=False):
        super(BaseIAMDecoder, self).__init__()

        self.scale_factor = scale_factor
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch(inst_dim, mask_num_convs, num_masks,
                                          kernel_dim, num_classes,
                                          in_channels + 2)
        self.mask_branch = MaskBranch(mask_dim, mask_num_convs, kernel_dim,
                                      in_channels + 2)

    @paddle.no_grad()
    def compute_coordinates(self, x):
        h, w = x.shape[2:4]
        y_loc = paddle.linspace(-1, 1, h)
        x_loc = paddle.linspace(-1, 1, w)
        y_loc, x_loc = paddle.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = paddle.concat([x_loc, y_loc], 1)
        return paddle.cast(locations, x.dtype)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = paddle.concat([coord_features, features], axis=1)
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
        mask_features = self.mask_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = paddle.bmm(pred_kernel,
                                mask_features.reshape((B, C, H * W))).reshape(
                                    (B, N, H, W))

        pred_masks = F.interpolate(
            pred_masks,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)

        output = {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
        }

        if self.output_iam:
            iam = F.interpolate(
                iam,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False)
            output['pred_iam'] = iam

        return output


class GroupInstanceBranch(nn.Layer):
    def __init__(self,
                 dim,
                 num_convs,
                 num_masks,
                 kernel_dim,
                 num_classes,
                 in_channels,
                 groups=1):
        super(GroupInstanceBranch, self).__init__()

        bias_value = bias_init_with_prob(0.01)
        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)

        expand_dim = dim * groups
        self.iam_conv = nn.Conv2D(
            dim,
            num_masks * groups,
            3,
            padding=1,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.),
            bias_attr=ParamAttr(initializer=Constant(value=bias_value)))

        # outputs
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.cls_score = nn.Linear(
            expand_dim,
            num_classes,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.),
            bias_attr=ParamAttr(initializer=Constant(value=bias_value)))
        self.mask_kernel = nn.Linear(
            expand_dim,
            kernel_dim,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.))
        self.objectness = nn.Linear(expand_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        kaiming_uniform_(self.fc.weight, a=1)
        kaiming_uniform_(self.objectness.weight)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = F.sigmoid(iam)
        B, N = iam_prob.shape[:2]
        C = features.shape[1]
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.reshape((B, N, -1))
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = paddle.bmm(iam_prob,
                                   features.reshape((B, C, -1)).transpose(
                                       (0, 2, 1)))
        normalizer = paddle.clip(iam_prob.sum(-1), min=1e-6)
        inst_features = inst_features / normalizer[:, :, None]

        inst_features = inst_features.reshape(
            (B, 4, N // 4, -1)).moveaxis(1, 2).reshape((B, N // 4, -1))

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        return pred_logits, pred_kernel, pred_scores, iam


@register
class GroupIAMDecoder(BaseIAMDecoder):
    def __init__(self,
                 num_masks,
                 num_classes,
                 in_channels,
                 inst_dim,
                 mask_dim,
                 inst_num_convs,
                 mask_num_convs,
                 kernel_dim,
                 scale_factor,
                 output_iam=False,
                 groups=1):
        super(GroupIAMDecoder, self).__init__(
            num_masks, num_classes, in_channels, inst_dim, mask_dim,
            inst_num_convs, mask_num_convs, kernel_dim, scale_factor,
            output_iam)

        self.inst_branch = GroupInstanceBranch(
            inst_dim, mask_num_convs, num_masks, kernel_dim, num_classes,
            in_channels + 2, groups)
