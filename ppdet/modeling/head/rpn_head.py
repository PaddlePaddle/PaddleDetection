import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D
from ppdet.core.workspace import register


@register
class RPNFeat(Layer):
    def __init__(self, feat_in=1024, feat_out=1024):
        super(RPNFeat, self).__init__()
        self.rpn_conv = Conv2D(
            num_channels=1024,
            num_filters=1024,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            param_attr=ParamAttr(
                name="conv_rpn_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))

    def forward(self, inputs):
        x = inputs.get('res4')
        y = self.rpn_conv(x)
        outs = {'rpn_feat': y}
        return outs


@register
class RPNHead(Layer):
    __inject__ = ['rpn_feat']

    def __init__(self, anchor_per_position=15, rpn_feat=RPNFeat().__dict__):
        super(RPNHead, self).__init__()
        self.anchor_per_position = anchor_per_position
        self.rpn_feat = rpn_feat
        if isinstance(rpn_feat, dict):
            self.rpn_feat = RPNFeat(**rpn_feat)

        # rpn roi classification scores
        self.rpn_rois_score = Conv2D(
            num_channels=1024,
            num_filters=1 * self.anchor_per_position,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                name="rpn_cls_logits_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_cls_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        # rpn roi bbox regression deltas
        self.rpn_rois_delta = Conv2D(
            num_channels=1024,
            num_filters=4 * self.anchor_per_position,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                name="rpn_bbox_pred_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_bbox_pred_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

    def forward(self, inputs):
        outs = self.rpn_feat(inputs)
        x = outs['rpn_feat']
        rrs = self.rpn_rois_score(x)
        rrd = self.rpn_rois_delta(x)
        outs.update({'rpn_rois_score': rrs, 'rpn_rois_delta': rrd})
        return outs

    def loss(self, inputs):
        if callable(inputs['anchor_module']):
            rpn_targets = inputs['anchor_module'].generate_anchors_target(
                inputs)
        # cls loss
        score_tgt = fluid.layers.cast(
            x=rpn_targets['rpn_score_target'], dtype='float32')
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=rpn_targets['rpn_score_pred'], label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        # reg loss
        rpn_reg_loss = fluid.layers.smooth_l1(
            x=rpn_targets['rpn_rois_pred'],
            y=rpn_targets['rpn_rois_target'],
            sigma=3.0,
            inside_weight=rpn_targets['rpn_rois_weight'],
            outside_weight=rpn_targets['rpn_rois_weight'])
        rpn_reg_loss = fluid.layers.reduce_mean(
            rpn_reg_loss, name='loss_rpn_reg')

        return rpn_cls_loss, rpn_reg_loss
