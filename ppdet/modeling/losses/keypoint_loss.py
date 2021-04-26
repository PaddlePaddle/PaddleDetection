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

from itertools import cycle, islice
from collections import abc
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable

__all__ = ['HrnetLoss']


@register
@serializable
class HrnetLoss(nn.Layer):
    def __init__(self, use_target_weight=True):
        """
        HrnetLoss layer

        Args:
            use_target_weight (bool): whether to use target weight
        """
        super(HrnetLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, records):
        target = records['target']
        target_weight = records['target_weight']
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(num_joints, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(num_joints, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                        heatmap_pred.multiply(target_weight[:,idx]),
                        heatmap_gt.multiply(target_weight[:, idx])
                        )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        keypoint_losses = dict()
        keypoint_losses['loss'] = loss / num_joints
        return keypoint_losses
