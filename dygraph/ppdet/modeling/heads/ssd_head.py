import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from paddle.regularizer import L2Decay
from paddle import ParamAttr


class SepConvLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 conv_decay=0,
                 name=None):
        super(SepConvLayer, self).__init__()
        self.dw_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_channels,
            weight_attr=ParamAttr(
                name=name + "_dw_weights", regularizer=L2Decay(conv_decay)),
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            in_channels,
            weight_attr=ParamAttr(
                name=name + "_bn_scale", regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                name=name + "_bn_offset", regularizer=L2Decay(0.)))

        self.pw_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(
                name=name + "_pw_weights", regularizer=L2Decay(conv_decay)),
            bias_attr=False)

    def forward(self, x):
        x = self.dw_conv(x)
        x = F.relu6(self.bn(x))
        x = self.pw_conv(x)
        return x


@register
class SSDHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['anchor_generator', 'loss']

    def __init__(self,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator='AnchorGeneratorSSD',
                 kernel_size=3,
                 padding=1,
                 use_sepconv=False,
                 conv_decay=0.,
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
            box_conv_name = "boxes{}".format(i)
            if not use_sepconv:
                box_conv = self.add_sublayer(
                    box_conv_name,
                    nn.Conv2D(
                        in_channels=in_channels[i],
                        out_channels=num_prior * 4,
                        kernel_size=kernel_size,
                        padding=padding))
            else:
                box_conv = self.add_sublayer(
                    box_conv_name,
                    SepConvLayer(
                        in_channels=in_channels[i],
                        out_channels=num_prior * 4,
                        kernel_size=kernel_size,
                        padding=padding,
                        conv_decay=conv_decay,
                        name=box_conv_name))
            self.box_convs.append(box_conv)

            score_conv_name = "scores{}".format(i)
            if not use_sepconv:
                score_conv = self.add_sublayer(
                    score_conv_name,
                    nn.Conv2D(
                        in_channels=in_channels[i],
                        out_channels=num_prior * num_classes,
                        kernel_size=kernel_size,
                        padding=padding))
            else:
                score_conv = self.add_sublayer(
                    score_conv_name,
                    SepConvLayer(
                        in_channels=in_channels[i],
                        out_channels=num_prior * num_classes,
                        kernel_size=kernel_size,
                        padding=padding,
                        conv_decay=conv_decay,
                        name=score_conv_name))
            self.score_convs.append(score_conv)

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
