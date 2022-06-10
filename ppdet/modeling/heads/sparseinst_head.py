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

from ppdet.modeling.layers import ConvNormLayer, MaskMatrixNMS, DropBlock
from ppdet.core.workspace import register

from six.moves import zip
import numpy as np

__all__ = ['BaseIAMDecoder', 'GroupIAMDecoder']


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(nn.Conv2D(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


class InstanceBranch(nn.Layer):
    def __init__(self, dim, num_convs, num_masks, kernel_dim, num_classes,
                 in_channels):
        super().__init__()
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)

        self.num_classes = num_classes
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
            self.num_classes,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.))
        self.mask_kernel = nn.Linear(
            dim,
            kernel_dim,
            weight_attr=ParamAttr(
                initializer=Normal(std=0.01), learning_rate=1.))
        self.objectness = nn.Linear(dim, 1)

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
        self.projection = nn.Conv2D(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        pass

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
        super().__init__()

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
        super().__init__()
        self.num_groups = groups
        self.num_classes = num_classes

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a group conv
        expand_dim = dim * self.num_groups
        self.iam_conv = nn.Conv2D(
            dim,
            num_masks * self.num_groups,
            3,
            padding=1,
            groups=self.num_groups)
        # outputs
        self.fc = nn.Linear(expand_dim, expand_dim)

        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        return
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)

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
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        inst_features = inst_features / normalizer[:, :, None]

        inst_features = inst_features.reshape((B, 4, N // 4, -1)).transpose(
            (1, 2)).reshape((B, N // 4, -1))

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


@register
class GroupIAMDecoder(BaseIAMDecoder):
    def __init__(self,
                 dim,
                 num_convs,
                 num_masks,
                 kernel_dim,
                 num_classes,
                 in_channels,
                 scale_factor,
                 output_iam=None,
                 groups=1):
        super().__init__(dim, num_convs, num_masks, kernel_dim, num_classes,
                         in_channels, scale_factor, output_iam, groups)
        self.inst_branch = GroupInstanceBranch(dim, num_convs, num_masks,
                                               kernel_dim, num_classes,
                                               in_channels + 2, groups)
