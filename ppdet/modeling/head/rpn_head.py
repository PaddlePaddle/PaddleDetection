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
        # rpn feat is shared with each level
        self.rpn_conv = Conv2D(
            num_channels=feat_in,
            num_filters=feat_out,
            filter_size=3,
            padding=1,
            act='relu',
            param_attr=ParamAttr(
                #name="conv_rpn_fpn2_w", 
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                #name="conv_rpn_fpn2_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

    def forward(self, inputs, feats):
        rpn_feats = []
        for feat in feats:
            rpn_feats.append(self.rpn_conv(feat))
        return rpn_feats


@register
class RPNHead(Layer):
    __inject__ = ['rpn_feat']

    def __init__(self, rpn_feat, anchor_per_position=15, rpn_channel=1024):
        super(RPNHead, self).__init__()
        self.rpn_feat = rpn_feat
        if isinstance(rpn_feat, dict):
            self.rpn_feat = RPNFeat(**rpn_feat)
        # rpn head is shared with each level
        # rpn roi classification scores
        self.rpn_rois_score = Conv2D(
            num_channels=rpn_channel,
            num_filters=anchor_per_position,
            filter_size=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                #name="rpn_cls_logits_fpn2_w",
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                #name="rpn_cls_logits_fpn2_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        # rpn roi bbox regression deltas
        self.rpn_rois_delta = Conv2D(
            num_channels=rpn_channel,
            num_filters=4 * anchor_per_position,
            filter_size=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                #name="rpn_bbox_pred_fpn2_w", 
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                #name="rpn_bbox_pred_fpn2_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

    def forward(self, inputs, feats):
        rpn_feats = self.rpn_feat(inputs, feats)
        rpn_head_out = []
        for rpn_feat in rpn_feats:
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            rpn_head_out.append((rrs, rrd))
        return rpn_feats, rpn_head_out

    def loss(self, loss_inputs):
        # cls loss
        score_tgt = fluid.layers.cast(
            x=loss_inputs['rpn_score_target'], dtype='float32')
        loss_rpn_cls = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=loss_inputs['rpn_score_pred'], label=score_tgt)
        loss_rpn_cls = fluid.layers.reduce_mean(
            loss_rpn_cls, name='loss_rpn_cls')

        # reg loss
        loss_rpn_reg = fluid.layers.smooth_l1(
            x=loss_inputs['rpn_rois_pred'],
            y=loss_inputs['rpn_rois_target'],
            sigma=3.0,
            inside_weight=loss_inputs['rpn_rois_weight'],
            outside_weight=loss_inputs['rpn_rois_weight'])
        loss_rpn_reg = fluid.layers.reduce_sum(loss_rpn_reg)
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        loss_rpn_reg = loss_rpn_reg / norm

        return {'loss_rpn_cls': loss_rpn_cls, 'loss_rpn_reg': loss_rpn_reg}
