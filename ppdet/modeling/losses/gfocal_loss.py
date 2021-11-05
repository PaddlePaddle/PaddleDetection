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

# The code is based on:
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/gfocal_loss.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling import ops

__all__ = ['QualityFocalLoss', 'DistributionFocalLoss']


def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):
    """
    Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy

    # negatives are supervised by 0 quality score
    pred_sigmoid = F.sigmoid(pred) if use_sigmoid else pred
    scale_factor = pred_sigmoid
    zerolabel = paddle.zeros(pred.shape, dtype='float32')
    loss = func(pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.shape[1]
    pos = paddle.logical_and((label >= 0),
                             (label < bg_class_ind)).nonzero().squeeze(1)
    if pos.shape[0] == 0:
        return loss.sum(axis=1)
    pos_label = paddle.gather(label, pos, axis=0)
    pos_mask = np.zeros(pred.shape, dtype=np.int32)
    pos_mask[pos.numpy(), pos_label.numpy()] = 1
    pos_mask = paddle.to_tensor(pos_mask, dtype='bool')
    score = score.unsqueeze(-1).expand([-1, pred.shape[1]]).cast('float32')
    # positives are supervised by bbox quality (IoU) score
    scale_factor_new = score - pred_sigmoid

    loss_pos = func(
        pred, score, reduction='none') * scale_factor_new.abs().pow(beta)
    loss = loss * paddle.logical_not(pos_mask) + loss_pos * pos_mask
    loss = loss.sum(axis=1)
    return loss


def distribution_focal_loss(pred, label):
    """Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (Tensor): Target distance label for bounding boxes with
            shape (N,).
    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.cast('int64')
    dis_right = dis_left + 1
    weight_left = dis_right.cast('float32') - label
    weight_right = label - dis_left.cast('float32')
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


@register
@serializable
class QualityFocalLoss(nn.Layer):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.
        Args:
            pred (Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """

        loss = self.loss_weight * quality_focal_loss(
            pred, target, beta=self.beta, use_sigmoid=self.use_sigmoid)

        if weight is not None:
            loss = loss * weight
        if avg_factor is None:
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if self.reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif self.reduction != 'none':
                raise ValueError(
                    'avg_factor can not be used with reduction="sum"')
        return loss


@register
@serializable
class DistributionFocalLoss(nn.Layer):
    """Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.
        Args:
            pred (Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        loss = self.loss_weight * distribution_focal_loss(pred, target)
        if weight is not None:
            loss = loss * weight
        if avg_factor is None:
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if self.reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif self.reduction != 'none':
                raise ValueError(
                    'avg_factor can not be used with reduction="sum"')
        return loss
