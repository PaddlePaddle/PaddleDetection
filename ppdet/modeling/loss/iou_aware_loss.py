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

import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from .iou_loss import IouLoss
from ..utils import xywh2xyxy, bbox_iou, decode_yolo
from paddle import fluid


@register
@serializable
class IouAwareLoss(IouLoss):
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self, loss_weight=1.0, giou=False, diou=False, ciou=False):
        super(IouAwareLoss, self).__init__(
            loss_weight=loss_weight, giou=giou, diou=diou, ciou=ciou)

    def __call__(self, ioup, pbox, gbox, anchor, downsample):
        b, na, h, w = ioup.shape
        iou = self._iou(pbox, gbox, anchor, downsample)
        iou.stop_gradient = True
        iou = iou.reshape((b, h, w, na)).transpose((0, 3, 1, 2))
        ioup = F.sigmoid(ioup)
        loss_iou_aware = fluid.layers.cross_entropy(ioup, iou, soft_label=True)
        loss_iou_aware = loss_iou_aware * self.loss_weight
        return loss_iou_aware
