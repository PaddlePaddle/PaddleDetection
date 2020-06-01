import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, MSRA
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, Pool2D

from ppdet.core.workspace import register


@register
class MaskNeck(Layer):
    def __init__(self, ):
        super(MaskNeck, self).__init__()

    def forward(self, inputs):
        conv5 = fluid.layers.gather(inputs, inputs['roi_has_mask_int32'])
        return conv5


@register
class MaskHead(Layer):
    def __init__(self, ):
        super(MaskHead, self).__init__()

        self.mask_out = fluid.dygraph.Conv2DTranspose(
            num_channels=2048,
            num_filters=self.cfg.dim_reduced,
            filter_size=2,
            stride=2,
            act='relu',
            param_attr=ParamAttr(
                name='conv5_mask_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))

        self.mask_fcn_logits = fluid.dygraph.Conv2D(
            num_channels=self.cfg.dim_reduced,
            num_filters=self.cfg.class_num,
            filter_size=1,
            act='sigmoid' if self.mode != 'train' else None,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='mask_fcn_logits_b',
                learning_rate=2.,
                regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        m_out = self.mask_out(inputs)
        m_logits = self.mask_fcn_logits(m_out)

        return m_logits

    def loss(self, inputs):
        # input needs (model_out, target)
        reshape_dim = self.cfg.Dataset.NUM_CLASSES * cfg.resolution * cfg.resolution
        mask_logits = fluid.layers.reshape(inputs['mask_logits'],
                                           (-1, reshape_dim))
        mask_label = fluid.layers.cast(x=inputs['mask_int32'], dtype='float32')

        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=mask_logits, label=mask_label, ignore_index=-1, normalize=True)
        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')

        return loss_mask
