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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from paddle.regularizer import L2Decay
from paddle import ParamAttr

from ..layers import AnchorGeneratorSSD


@register
class FaceHead(nn.Layer):
    """
    Head block for Face detection network

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        anchor_generator(object): instance of anchor genertor method.
        kernel_size (int): kernel size of Conv2D in FaceHead.
        padding (int): padding of Conv2D in FaceHead.
        conv_decay (float): norm_decay (float): weight decay for conv layer weights.
        loss (object): loss of face detection model.
    """
    __shared__ = ['num_classes']
    __inject__ = ['anchor_generator', 'loss']

    def __init__(self,
                 num_classes=80,
                 in_channels=(96, 96),
                 anchor_generator=AnchorGeneratorSSD().__dict__,
                 kernel_size=3,
                 padding=1,
                 conv_decay=0.,
                 loss='SSDLoss'):
        super(FaceHead, self).__init__()
        # add background class
        self.num_classes = num_classes + 1
        self.in_channels = in_channels
        self.anchor_generator = anchor_generator
        self.loss = loss

        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGeneratorSSD(**anchor_generator)

        self.num_priors = self.anchor_generator.num_priors
        self.box_convs = []
        self.score_convs = []
        for i, num_prior in enumerate(self.num_priors):
            box_conv_name = "boxes{}".format(i)
            box_conv = self.add_sublayer(
                box_conv_name,
                nn.Conv2D(
                    in_channels=in_channels[i],
                    out_channels=num_prior * 4,
                    kernel_size=kernel_size,
                    padding=padding))
            self.box_convs.append(box_conv)

            score_conv_name = "scores{}".format(i)
            score_conv = self.add_sublayer(
                score_conv_name,
                nn.Conv2D(
                    in_channels=in_channels[i],
                    out_channels=num_prior * self.num_classes,
                    kernel_size=kernel_size,
                    padding=padding))
            self.score_convs.append(score_conv)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def forward(self, feats, image, gt_bbox=None, gt_class=None):
        box_preds = []
        cls_scores = []
        prior_boxes = []
        for feat, box_conv, score_conv in zip(feats, self.box_convs,
                                              self.score_convs):
            box_pred = box_conv(feat)
            box_pred = paddle.transpose(box_pred, [0, 2, 3, 1])
            box_pred = paddle.reshape(box_pred, [0, -1, 4])
            box_preds.append(box_pred)

            cls_score = score_conv(feat)
            cls_score = paddle.transpose(cls_score, [0, 2, 3, 1])
            cls_score = paddle.reshape(cls_score, [0, -1, self.num_classes])
            cls_scores.append(cls_score)

        prior_boxes = self.anchor_generator(feats, image)

        if self.training:
            return self.get_loss(box_preds, cls_scores, gt_bbox, gt_class,
                                 prior_boxes)
        else:
            return (box_preds, cls_scores), prior_boxes

    def get_loss(self, boxes, scores, gt_bbox, gt_class, prior_boxes):
        return self.loss(boxes, scores, gt_bbox, gt_class, prior_boxes)
