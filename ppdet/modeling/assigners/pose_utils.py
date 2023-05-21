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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn.functional as F

from ppdet.core.workspace import register

__all__ = ['KptL1Cost', 'OksCost', 'ClassificationCost']


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


@register
class KptL1Cost(object):
    """KptL1Cost.

    this function based on: https://github.com/hikvision-research/opera/blob/main/opera/core/bbox/match_costs/match_cost.py

    Args:
        weight (int | float, optional): loss_weight.
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with normalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].

        Returns:
            paddle.Tensor: kpt_cost value with weight.
        """
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            if gt_keypoints[i].size == 0:
                kpt_cost.append(kpt_pred.sum() * 0)
            kpt_pred_tmp = kpt_pred.clone()
            valid_flag = valid_kpt_flag[i] > 0
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(-1).expand_as(
                kpt_pred_tmp)
            if not valid_flag_expand.all():
                kpt_pred_tmp = masked_fill(kpt_pred_tmp, ~valid_flag_expand, 0)
            cost = F.pairwise_distance(
                kpt_pred_tmp.reshape((kpt_pred_tmp.shape[0], -1)),
                gt_keypoints[i].reshape((-1, )).unsqueeze(0),
                p=1,
                keepdim=True)
            avg_factor = paddle.clip(
                valid_flag.astype('float32').sum() * 2, 1.0)
            cost = cost / avg_factor
            kpt_cost.append(cost)
        kpt_cost = paddle.concat(kpt_cost, axis=1)
        return kpt_cost * self.weight


@register
class OksCost(object):
    """OksCost.

    this function based on: https://github.com/hikvision-research/opera/blob/main/opera/core/bbox/match_costs/match_cost.py

    Args:
        num_keypoints (int): number of keypoints
        weight (int | float, optional): loss_weight.
    """

    def __init__(self, num_keypoints=17, weight=1.0):
        self.weight = weight
        if num_keypoints == 17:
            self.sigmas = np.array(
                [
                    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                    1.07, .87, .87, .89, .89
                ],
                dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array(
                [
                    .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89,
                    .89, .79, .79
                ],
                dtype=np.float32) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag, gt_areas):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].
            gt_areas (Tensor): Ground truth mask areas. Shape [num_gt,].

        Returns:
            paddle.Tensor: oks_cost value with weight.
        """
        sigmas = paddle.to_tensor(self.sigmas)
        variances = (sigmas * 2)**2

        oks_cost = []
        assert len(gt_keypoints) == len(gt_areas)
        for i in range(len(gt_keypoints)):
            if gt_keypoints[i].size == 0:
                oks_cost.append(kpt_pred.sum() * 0)
            squared_distance = \
                (kpt_pred[:, :, 0] - gt_keypoints[i, :, 0].unsqueeze(0)) ** 2 + \
                (kpt_pred[:, :, 1] - gt_keypoints[i, :, 1].unsqueeze(0)) ** 2
            vis_flag = (valid_kpt_flag[i] > 0).astype('int')
            vis_ind = vis_flag.nonzero(as_tuple=False)[:, 0]
            num_vis_kpt = vis_ind.shape[0]
            # assert num_vis_kpt > 0
            if num_vis_kpt == 0:
                oks_cost.append(paddle.zeros((squared_distance.shape[0], 1)))
                continue
            area = gt_areas[i]

            squared_distance0 = squared_distance / (area * variances * 2)
            squared_distance0 = paddle.index_select(
                squared_distance0, vis_ind, axis=1)
            squared_distance1 = paddle.exp(-squared_distance0).sum(axis=1,
                                                                   keepdim=True)
            oks = squared_distance1 / num_vis_kpt
            # The 1 is a constant that doesn't change the matching, so omitted.
            oks_cost.append(-oks)
        oks_cost = paddle.concat(oks_cost, axis=1)
        return oks_cost * self.weight


@register
class ClassificationCost:
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            paddle.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@register
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12
         binary_input (bool, optional): Whether the input is binary,
            default False.
    """

    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            paddle.Tensor: cls_cost value with weight
        """
        if gt_labels.size == 0:
            return cls_pred.sum() * 0
        cls_pred = F.sigmoid(cls_pred)
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = paddle.index_select(
            pos_cost, gt_labels, axis=1) - paddle.index_select(
                neg_cost, gt_labels, axis=1)
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits
                in shape (num_query, d1, ..., dn), dtype=paddle.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=paddle.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = F.sigmoid(cls_pred)
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = paddle.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            paddle.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits.
            gt_labels (Tensor)): Labels.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            return self._focal_loss_cost(cls_pred, gt_labels)
