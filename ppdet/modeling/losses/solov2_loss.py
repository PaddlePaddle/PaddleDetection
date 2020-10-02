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
from paddle import fluid
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
        input = fluid.layers.reshape(
            input, shape=(fluid.layers.shape(input)[0], -1))
        target = fluid.layers.reshape(
            target, shape=(fluid.layers.shape(target)[0], -1))
        target = fluid.layers.cast(target, 'float32')
        a = fluid.layers.reduce_sum(input * target, dim=1)
        b = fluid.layers.reduce_sum(input * input, dim=1) + 0.001
        c = fluid.layers.reduce_sum(target * target, dim=1) + 0.001
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

        # Ues dice_loss to calculate instance loss
        loss_ins = []
        total_weights = fluid.layers.zeros(shape=[1], dtype='float32')
        for input, target in zip(ins_pred_list, ins_label_list):
            weights = fluid.layers.cast(
                fluid.layers.reduce_sum(
                    target, dim=[1, 2]) > 0, 'float32')
            input = fluid.layers.sigmoid(input)
            dice_out = fluid.layers.elementwise_mul(
                self._dice_loss(input, target), weights)
            total_weights += fluid.layers.reduce_sum(weights)
            loss_ins.append(dice_out)
        loss_ins = fluid.layers.reduce_sum(fluid.layers.concat(
            loss_ins)) / total_weights
        loss_ins = loss_ins * self.ins_loss_weight

        # Ues sigmoid_focal_loss to calculate category loss
        loss_cate = fluid.layers.sigmoid_focal_loss(
            x=cate_preds,
            label=cate_labels,
            fg_num=num_ins + 1,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha)
        loss_cate = fluid.layers.reduce_sum(loss_cate)

        return loss_ins, loss_cate
