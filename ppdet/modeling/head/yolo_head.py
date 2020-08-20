import paddle.fluid as fluid
import paddle
from paddle.fluid.dygraph import Layer
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph import Sequential
from ppdet.core.workspace import register
from ..backbone.darknet import ConvBNLayer


class YoloDetBlock(Layer):
    def __init__(self, ch_in, channel, name):
        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
        conv_def = [
            ['conv0', ch_in, channel, 1, '.0.0'],
            ['conv1', channel, channel * 2, 3, '.0.1'],
            ['conv2', channel * 2, channel, 1, '.1.0'],
            ['conv3', channel, channel * 2, 3, '.1.1'],
            ['route', channel * 2, channel, 1, '.2'],
            #['tip', channel, channel * 2, 3],
        ]

        self.conv_module = Sequential()
        for idx, (conv_name, ch_in, ch_out, filter_size,
                  post_name) in enumerate(conv_def):
            self.conv_module.add_sublayer(
                conv_name,
                ConvBNLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=filter_size,
                    padding=(filter_size - 1) // 2,
                    name=name + post_name))

        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            padding=1,
            name=name + '.tip')

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


@register
class YOLOFeat(Layer):
    __shared__ = ['num_levels']

    def __init__(self, feat_in_list=[1024, 768, 384], num_levels=3):
        super(YOLOFeat, self).__init__()
        self.feat_in_list = feat_in_list
        self.yolo_blocks = []
        self.route_blocks = []
        self.num_levels = num_levels
        for i in range(self.num_levels):
            name = 'yolo_block.{}'.format(i)
            yolo_block = self.add_sublayer(
                name,
                YoloDetBlock(
                    feat_in_list[i], channel=512 // (2**i), name=name))
            self.yolo_blocks.append(yolo_block)

            if i < self.num_levels - 1:
                name = 'yolo_transition.{}'.format(i)
                route = self.add_sublayer(
                    name,
                    ConvBNLayer(
                        ch_in=512 // (2**i),
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        name=name))
                self.route_blocks.append(route)

    def forward(self, body_feats):
        assert len(body_feats) == self.num_levels
        body_feats = body_feats[::-1]
        yolo_feats = []
        for i, block in enumerate(body_feats):
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if i < self.num_levels - 1:
                route = self.route_blocks[i](route)
                route = fluid.layers.resize_nearest(route, scale=2.)

        return yolo_feats


@register
class YOLOv3Head(Layer):
    __shared__ = ['num_classes', 'num_levels', 'use_fine_grained_loss']
    __inject__ = ['yolo_feat']

    def __init__(self,
                 yolo_feat,
                 num_classes=80,
                 anchor_per_position=3,
                 num_levels=3,
                 use_fine_grained_loss=False,
                 ignore_thresh=0.7,
                 downsample=32,
                 label_smooth=True):
        super(YOLOv3Head, self).__init__()
        self.num_classes = num_classes
        self.anchor_per_position = anchor_per_position
        self.yolo_feat = yolo_feat
        self.num_levels = num_levels
        self.use_fine_grained_loss = use_fine_grained_loss
        self.ignore_thresh = ignore_thresh
        self.downsample = downsample
        self.label_smooth = label_smooth
        self.yolo_out_list = []
        for i in range(num_levels):
            # TODO: optim here
            #num_filters = len(cfg.anchor_masks[i]) * (self.num_classes + 5)
            num_filters = self.anchor_per_position * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            yolo_out = self.add_sublayer(
                name,
                Conv2D(
                    num_channels=1024 // (2**i),
                    num_filters=num_filters,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(name=name + '.conv.weights'),
                    bias_attr=ParamAttr(
                        name=name + '.conv.bias', regularizer=L2Decay(0.))))
            self.yolo_out_list.append(yolo_out)

    def forward(self, body_feats):
        assert len(body_feats) == self.num_levels
        yolo_feats = self.yolo_feat(body_feats)
        yolo_head_out = []
        for i, feat in enumerate(yolo_feats):
            yolo_out = self.yolo_out_list[i](feat)
            yolo_head_out.append(yolo_out)
        return yolo_head_out

    def loss(self, inputs, head_out, anchors, anchor_masks, mask_anchors):
        if self.use_fine_grained_loss:
            raise NotImplementedError

        yolo_losses = []
        for i, out in enumerate(head_out):
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
            loss = fluid.layers.reduce_mean(loss)
            yolo_losses.append(loss)
        return {'loss': sum(yolo_losses)}
