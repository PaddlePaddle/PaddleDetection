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

import math

import paddle
import paddle.nn as nn

from .. import initializer as init
from ppdet.core.workspace import register


@register
class RetinaNetHead(nn.Layer):
    '''
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    '''
    __inject__ = ['loss_func']

    def __init__(
        self,
        num_classes,
        num_convs,
        input_shape={"channels": 256},
        num_anchors=9,
        norm="",
        loss_func="RetinaNetLoss",
        prior_prob=0.01
    ):
        '''
        Args:
            input_shape (List[ShapeSpec]): input shape.
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors.
            conv_dims (List[int]): dimensions for each convolution layer.
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            loss_func (class): the class is used to compute loss.
            prior_prob (float): Prior weight for computing bias.
        '''
        super(RetinaNetHead, self).__init__()

        self.num_classes = num_classes
        self.get_loss = loss_func
        self.prior_prob = prior_prob

        # conv_dims = [input_shape[0].channels] * num_convs
        input_channels = input_shape["channels"]
        conv_dims = [input_channels] * num_convs

        cls_net = []
        reg_net = []

        for in_channels, out_channels in zip(
            [input_channels] + list(conv_dims), conv_dims
        ):
            cls_net.append(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm == "bn":
                cls_net.append(nn.BatchNorm2D(out_channels))
            cls_net.append(nn.ReLU())

            reg_net.append(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm == "bn":
                reg_net.append(nn.BatchNorm2D(out_channels))
            reg_net.append(nn.ReLU())

        self.cls_net = nn.Sequential(*cls_net)
        self.reg_net = nn.Sequential(*reg_net)

        self.cls_score = nn.Conv2D(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2D(
            conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        init.reset_initialized_parameter(self)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                init.normal_(m.weight, mean=0., std=0.01)
                init.constant_(m.bias, 0)

        init.constant_(self.cls_score.bias, bias_value)

    def forward(self, feats):
        pred_scores = []
        pred_boxes = []

        for feat in feats:
            pred_scores.append(self.cls_score(self.cls_net(feat)))
            pred_boxes.append(self.bbox_pred(self.reg_net(feat)))

        return pred_scores, pred_boxes

    def losses(self, anchors, preds, inputs):
        anchors = paddle.concat(anchors)

        return self.get_loss(anchors, preds, inputs)

        
        


