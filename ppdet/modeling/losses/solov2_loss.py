# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable

__all__ = ['SOLOv2Loss']


@register
@serializable
class SOLOv2Loss(object):
    """
    SOLOv2Loss
    Args:
        ins_loss_weight (float): Weight of instance loss.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
    """

    def __init__(self,
                 ins_loss_weight=3.0,
                 focal_loss_gamma=2.0,
                 focal_loss_alpha=0.25):
        self.ins_loss_weight = ins_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

    def _dice_loss(self, input, target):
        input = paddle.reshape(input, shape=(input.shape[0], -1))
        target = paddle.reshape(target, shape=(target.shape[0], -1))
        a = paddle.sum(input * target, axis=1)
        b = paddle.sum(input * input, axis=1) + 0.001
        c = paddle.sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def __call__(self, ins_pred_list, ins_label_list, cate_preds, cate_labels,
                 num_ins):
        """
        Get loss of network of SOLOv2.
        Args:
            ins_pred_list (list): Variable list of instance branch output.
            ins_label_list (list): List of instance labels pre batch.
            cate_preds (list): Concat Variable list of categroy branch output.
            cate_labels (list): Concat list of categroy labels pre batch.
            num_ins (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Variable): The instance loss Variable of SOLOv2 network.
            loss_cate (Variable): The category loss Variable of SOLOv2 network.
        """

        #1. Ues dice_loss to calculate instance loss
        loss_ins = []
        total_weights = paddle.zeros(shape=[1], dtype='float32')
        for input, target in zip(ins_pred_list, ins_label_list):
            if input is None:
                continue
            target = paddle.cast(target, 'float32')
            target = paddle.reshape(
                target,
                shape=[-1, input.shape[-2], input.shape[-1]])
            weights = paddle.cast(
                paddle.sum(target, axis=[1, 2]) > 0, 'float32')
            input = F.sigmoid(input)
            dice_out = paddle.multiply(self._dice_loss(input, target), weights)
            total_weights += paddle.sum(weights)
            loss_ins.append(dice_out)
        loss_ins = paddle.sum(paddle.concat(loss_ins)) / total_weights
        loss_ins = loss_ins * self.ins_loss_weight

        #2. Ues sigmoid_focal_loss to calculate category loss
        # expand onehot labels
        num_classes = cate_preds.shape[-1]
        cate_labels_bin = F.one_hot(cate_labels, num_classes=num_classes + 1)
        cate_labels_bin = cate_labels_bin[:, 1:]

        loss_cate = F.sigmoid_focal_loss(
            cate_preds,
            label=cate_labels_bin,
            normalizer=num_ins + 1.,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha)

        return loss_ins, loss_cate
