import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from ppdet.core.workspace import register


@register
class SSDHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['anchor_generator', 'loss']

    def __init__(self,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator='AnchorGeneratorSSD',
                 loss='SSDLoss'):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.anchor_generator = anchor_generator
        self.loss = loss
        self.num_priors = self.anchor_generator.num_priors

        self.box_convs = []
        self.score_convs = []
        for i, num_prior in enumerate(self.num_priors):
            self.box_convs.append(
                self.add_sublayer(
                    "boxes{}".format(i),
                    nn.Conv2D(
                        in_channels=in_channels[i],
                        out_channels=num_prior * 4,
                        kernel_size=3,
                        padding=1)))
            self.score_convs.append(
                self.add_sublayer(
                    "scores{}".format(i),
                    nn.Conv2D(
                        in_channels=in_channels[i],
                        out_channels=num_prior * num_classes,
                        kernel_size=3,
                        padding=1)))

    def forward(self, feats, image):
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

        outputs = {}
        outputs['boxes'] = box_preds
        outputs['scores'] = cls_scores
        return outputs, prior_boxes

    def get_loss(self, inputs, targets, prior_boxes):
        return self.loss(inputs, targets, prior_boxes)
