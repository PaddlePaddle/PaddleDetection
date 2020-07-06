import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from ppdet.core.workspace import register
from ..backbone.darknet import ConvBNLayer


class YoloDetBlock(Layer):
    def __init__(self, ch_in, channel):
        super(YoloDetBlock, self).__init__()

        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)

        self.conv0 = ConvBNLayer(
            ch_in=ch_in, ch_out=channel, filter_size=1, stride=1, padding=0)

        self.conv1 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1)

        self.conv2 = ConvBNLayer(
            ch_in=channel * 2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0)

        self.conv3 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1)

        self.route = ConvBNLayer(
            ch_in=channel * 2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0)

        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class Upsample(Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(
            shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out


@register
class YOLOFeat(Layer):
    def __init__(self, feat_in_list=[1024, 768, 384]):
        super(YOLOFeat, self).__init__()
        self.feat_in_list = feat_in_list
        self.yolo_blocks = []
        self.route_blocks = []
        for i in range(3):
            yolo_block = self.add_sublayer(
                "yolo_det_block_%d" % (i),
                YoloDetBlock(
                    feat_in_list[i], channel=512 // (2**i)))
            self.yolo_blocks.append(yolo_block)

            if i < 2:
                route = self.add_sublayer(
                    "route_%d" % i,
                    ConvBNLayer(
                        ch_in=512 // (2**i),
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0))
                self.route_blocks.append(route)
        self.upsample = Upsample()

    def forward(self, inputs):
        yolo_feats = []
        for i, block in enumerate(inputs['darknet_outs']):
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)

            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if i < 2:
                route = self.route_blocks[i](route)
                route = self.upsample(route)

        outs = {'yolo_feat': yolo_feats}
        return outs


@register
class YOLOv3Head(Layer):
    __shared__ = ['num_classes']
    __inject__ = ['yolo_feat']

    def __init__(
            self,
            num_classes=80,
            anchor_per_position=3,
            mode='train',
            yolo_feat=YOLOFeat().__dict__, ):
        super(YOLOv3Head, self).__init__()
        self.num_classes = num_classes
        self.anchor_per_position = anchor_per_position
        self.mode = mode
        self.yolo_feat = yolo_feat
        if isinstance(yolo_feat, dict):
            self.yolo_feat = YOLOFeat(**yolo_feat)

        self.yolo_outs = []
        for i in range(3):
            # TODO: optim here
            #num_filters = len(cfg.anchor_masks[i]) * (self.num_classes + 5)
            num_filters = self.anchor_per_position * (self.num_classes + 5)
            yolo_out = self.add_sublayer(
                "yolo_out_%d" % (i),
                Conv2D(
                    num_channels=1024 // (2**i),
                    num_filters=num_filters,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        initializer=fluid.initializer.Normal(0., 0.02)),
                    bias_attr=ParamAttr(
                        initializer=fluid.initializer.Constant(0.0),
                        regularizer=L2Decay(0.))))
            self.yolo_outs.append(yolo_out)

    def forward(self, inputs):
        outs = self.yolo_feat(inputs)
        x = outs['yolo_feat']
        yolo_out_list = []
        for i, yolo_f in enumerate(x):
            yolo_out = self.yolo_outs[i](yolo_f)
            yolo_out_list.append(yolo_out)
        outs.update({"yolo_outs": yolo_out_list})
        return outs

    def loss(self, inputs):
        if callable(inputs['anchor_module']):
            yolo_targets = inputs['anchor_module'].generate_anchors_target(
                inputs)
        yolo_losses = []
        for i, out in enumerate(inputs['yolo_outs']):
            # TODO: split yolov3_loss into small ops
            # 1. compute target 2. loss
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=inputs['gt_bbox'],
                gt_label=inputs['gt_class'],
                gt_score=inputs['gt_score'],
                anchors=inputs['anchors'],
                anchor_mask=inputs['anchor_masks'][i],
                class_num=self.num_classes,
                ignore_thresh=yolo_targets['ignore_thresh'],
                downsample_ratio=yolo_targets['downsample_ratio'] // 2**i,
                use_label_smooth=yolo_targets['label_smooth'],
                name='yolo_loss_' + str(i))
            loss = fluid.layers.reduce_mean(loss)
            yolo_losses.append(loss)
        yolo_loss = sum(yolo_losses)
        return yolo_loss
