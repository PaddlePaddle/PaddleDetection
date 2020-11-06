import paddle.fluid as fluid
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register
from ..backbone.darknet import ConvBNLayer


@register
class YOLOv3Head(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['loss']

    def __init__(self,
                 anchors=[
                     10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90,
                     156, 198, 373, 326
                 ],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss'):
        super(YOLOv3Head, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.num_classes = num_classes
        self.loss = loss

        self.mask_anchors = self.parse_anchor(self.anchors, self.anchor_masks)
        self.num_outputs = len(self.mask_anchors)

        self.yolo_outputs = []
        for i in range(len(self.mask_anchors)):
            num_filters = self.num_outputs * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            yolo_output = self.add_sublayer(
                name,
                nn.Conv2D(
                    in_channels=1024 // (2**i),
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(name=name + '.conv.weights'),
                    bias_attr=ParamAttr(
                        name=name + '.conv.bias', regularizer=L2Decay(0.))))
            self.yolo_outputs.append(yolo_output)

    def parse_anchor(self, anchors, anchor_masks):
        anchor_num = len(self.anchors)
        mask_anchors = []
        for i in range(len(self.anchor_masks)):
            mask_anchor = []
            for m in self.anchor_masks[i]:
                assert m < anchor_num, "anchor mask index overflow"
                mask_anchor.extend(self.anchors[2 * m:2 * m + 2])
            mask_anchors.append(mask_anchor)

        return mask_anchors

    def forward(self, feats):
        assert len(feats) == len(self.mask_anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            yolo_outputs.append(yolo_output)
        return yolo_outputs

    def loss(self, inputs, head_outputs):
        return self.loss(inputs, head_outputs, anchors, anchor_masks)
