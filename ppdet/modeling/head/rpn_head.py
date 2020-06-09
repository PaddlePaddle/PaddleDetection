import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D

from ppdet.core.workspace import register
from ..ops import RPNAnchorTargetGenerator


@register
class RPNFeat(Layer):
    def __init__(self, feat_in=1024, feat_out=1024):
        super(RPNFeat, self).__init__()
        self.rpn_conv = fluid.dygraph.Conv2D(
            num_channels=1024,
            num_filters=1024,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            param_attr=ParamAttr(
                "conv_rpn_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                "conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))

    def forward(self, inputs):
        x = inputs.get('res4')
        y = self.rpn_conv(x)
        outs = {'rpn_feat': y}
        return outs


@register
class RPNHead(Layer):
    __inject__ = ['rpn_feat', 'rpn_target_assign']

    def __init__(self,
                 anchor_per_position=15,
                 rpn_feat=RPNFeat().__dict__,
                 rpn_target_assign=RPNAnchorTargetGenerator().__dict__):
        super(RPNHead, self).__init__()
        self.anchor_per_position = anchor_per_position
        self.rpn_feat = rpn_feat
        self.rpn_target_assign = rpn_target_assign
        if isinstance(rpn_feat, dict):
            self.rpn_feat = RPNFeat(**rpn_feat)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNAnchorTargetGenerator(
                **rpn_target_assign)

        # rpn roi classification scores
        self.rpn_rois_score = fluid.dygraph.Conv2D(
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
        self.rpn_rois_delta = fluid.dygraph.Conv2D(
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
        #x = inputs.get('rpn_neck')
        rpn_feat_out = self.rpn_feat(inputs)
        x = rpn_feat_out['rpn_feat']
        rrs = self.rpn_rois_score(x)
        rrd = self.rpn_rois_delta(x)
        outs = {'rpn_rois_score': rrs, 'rpn_rois_delta': rrd}
        outs.update(rpn_feat_out)
        return outs

    def loss(self, inputs):
        # generate anchor target
        # TODO: move anchor target func into 'anchor.py'
        rpn_rois_score = fluid.layers.transpose(
            inputs['rpn_rois_score'], perm=[0, 2, 3, 1])
        rpn_rois_delta = fluid.layers.transpose(
            inputs['rpn_rois_delta'], perm=[0, 2, 3, 1])
        rpn_rois_score = fluid.layers.reshape(
            x=rpn_rois_score, shape=(0, -1, 1))
        rpn_rois_delta = fluid.layers.reshape(
            x=rpn_rois_delta, shape=(0, -1, 4))

        anchor = fluid.layers.reshape(inputs['anchor'], shape=(-1, 4))
        #var = fluid.layers.reshape(inputs['var'], shape=(-1, 4))

        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = self.rpn_target_assign(
            bbox_pred=rpn_rois_delta,
            cls_logits=rpn_rois_score,
            anchor_box=anchor,
            gt_boxes=inputs['gt_bbox'],
            is_crowd=inputs['is_crowd'],
            im_info=inputs['im_info'])

        # cls loss
        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=score_pred, label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')
        # reg loss
        rpn_reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        rpn_reg_loss = fluid.layers.reduce_mean(
            rpn_reg_loss, name='loss_rpn_reg')
        '''
        rpn_reg_loss_sum = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = score_tgt.shape
        norm = 1
        for i in score_shape:
            norm *= i
        rpn_reg_loss = rpn_reg_loss_sum / norm
        '''

        return rpn_cls_loss, rpn_reg_loss
