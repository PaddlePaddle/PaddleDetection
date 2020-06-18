import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, MSRA
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, Pool2D
from ppdet.core.workspace import register
from ..ops import RoIExtractor
from ..backbone.resnet import Blocks


@register
class MaskFeat(Layer):
    __inject__ = ['mask_roi_extractor']

    def __init__(self,
                 feat_in=2048,
                 feat_out=256,
                 mask_roi_extractor=RoIExtractor().__dict__,
                 stage=0):
        super(MaskFeat, self).__init__()
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.mask_roi_extractor = mask_roi_extractor
        if isinstance(mask_roi_extractor, dict):
            self.mask_roi_extractor = RoIExtractor(**mask_roi_extractor)
        if stage == 0:
            postfix = ''
        else:
            postfix = '_' + str(stage)
        self.upsample = fluid.dygraph.Conv2DTranspose(
            num_channels=self.feat_in,
            num_filters=self.feat_out,
            filter_size=2,
            stride=2,
            act='relu',
            param_attr=ParamAttr(
                name='conv5_mask_w' + postfix, initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='conv5_mask_b' + postfix,
                learning_rate=2.,
                regularizer=L2Decay(0.)))

    def forward(self, inputs):
        if inputs['mode'] == 'train':
            x = inputs['res5']
            rois_feat = fluid.layers.gather(x, inputs['rois_has_mask_int32'])
        elif inputs['mode'] == 'infer':
            rois = inputs['predicted_bbox'][:, 2:] * inputs['im_info'][:, 2]
            rois_num = inputs['predicted_bbox_nums']
            # TODO: optim here 
            if callable(inputs['shared_roi_extractor']):
                rois_feat = inputs['shared_roi_extractor'](inputs['res4'], rois,
                                                           rois_num)
            if callable(inputs['shared_res5_block']):
                rois_feat = inputs['shared_res5_block'](rois_feat)
        # upsample 
        y = self.upsample(rois_feat)
        outs = {'mask_feat': y}
        return outs


@register
class MaskHead(Layer):
    __shared__ = ['num_classes']
    __inject__ = ['mask_feat']

    def __init__(self,
                 feat_in=256,
                 resolution=14,
                 num_classes=81,
                 mask_feat=MaskFeat().__dict__,
                 stage=0):
        super(MaskHead, self).__init__()
        self.feat_in = feat_in
        self.resolution = resolution
        self.num_classes = num_classes
        self.mask_feat = mask_feat
        if isinstance(mask_feat, dict):
            self.mask_feat = MaskFeat(**mask_feat)
        if stage == 0:
            postfix = ''
        else:
            postfix = '_' + str(stage)
        self.mask_fcn_logits = fluid.dygraph.Conv2D(
            num_channels=self.feat_in,
            num_filters=self.num_classes,
            filter_size=1,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w' + postfix,
                initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='mask_fcn_logits_b' + postfix,
                learning_rate=2.,
                regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        # feat 
        outs = self.mask_feat(inputs)
        x = outs['mask_feat']
        # logits 
        mask_logits = self.mask_fcn_logits(x)
        if inputs['mode'] == 'infer':
            pred_bbox = inputs['predicted_bbox']
            shape = reduce((lambda x, y: x * y), pred_bbox.shape)
            shape = np.asarray(shape).reshape((1, 1))
            ones = np.ones((1, 1), dtype=np.int32)
            cond = (shape == ones).all()
            if cond:
                mask_logits = pred_bbox

        outs['mask_logits'] = mask_logits

        return outs

    def loss(self, inputs):
        # input needs (model_out, target)
        reshape_dim = self.num_classes * self.resolution * self.resolution
        mask_logits = fluid.layers.reshape(inputs['mask_logits'],
                                           (-1, reshape_dim))
        mask_label = fluid.layers.cast(x=inputs['mask_int32'], dtype='float32')

        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=mask_logits, label=mask_label, ignore_index=-1, normalize=True)
        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')

        return loss_mask
