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

import paddle.fluid as fluid
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register
from ..backbone.darknet import ConvBNLayer


@register
class YOLOv3Loss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=32,
                 use_fine_grained_loss=False):
        super(YOLOv3Loss, self).__init__()
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.use_fine_grained_loss = use_fine_grained_loss

    def forward(self, inputs, head_outputs, anchors, anchor_masks):
        if self.use_fine_grained_loss:
            raise NotImplementedError(
                "fine grained loss not implement currently")

        yolo_losses = []
        for i, out in enumerate(head_outputs):
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=inputs['gt_bbox'],
                gt_label=inputs['gt_class'],
                gt_score=inputs['gt_score'],
                anchors=anchors,
                anchor_mask=anchor_masks[i],
                class_num=self.num_classes,
                ignore_thresh=self.ignore_thresh,
                downsample_ratio=self.downsample // 2**i,
                use_label_smooth=self.label_smooth,
                name='yolo_loss_' + str(i))
            loss = paddle.mean(loss)
            yolo_losses.append(loss)
        return {'loss': sum(yolo_losses)}
