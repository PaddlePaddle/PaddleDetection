import paddle
import paddle.nn as nn
import paddle.nn.functional as F


@register
class SSDHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['anchor_generator']

    def __init__(self,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator=None):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.anchor_generator = anchor_generator
        self.boxes = anchor_generator()

        box_convs = []
        score_convs = []
        for i, box in enumerate(self.boxes):
            num_boxes = box.shape[2]
            box_convs.append(
                self.add_sublayer(
                    "boxes{}".format(i),
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=num_boxes * 4,
                        kernel_size=3,
                        padding=1)))
            score_convs.append(
                self.add_sublayer(
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=num_classes * num_classes,
                        kernel_size=3,
                        padding=1)))

    def forward(self, feats):
        box_preds = []
        cls_scores = []
        for feat, box_conv, score_conv in zip(feats, box_convs, score_convs):
            box_pred = box_conv(feat)
            box_pred = paddle.tranpose(box_pred, [0, 2, 3, 1])
            box_pred = paddle.reshape(box_pred, [0, -1, 4])
            box_preds.append(box_pred)

            cls_score = score_conv(feat)
            cls_score = paddle.tranpose(box_pred, [0, 2, 3, 1])
            cls_score = paddle.reshape(box_pred, [0, -1, self.num_classes])
            cls_scores.append(cls_score)

        return box_preds, cls_scores, self.boxes

    def get_loss(self, ):
        pass
