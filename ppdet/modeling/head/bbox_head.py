import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, MSRA
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, Pool2D

from ppdet.core.workspace import register
from ..backbone.resnet import Blocks as resnet_blocks
from ..ops import RoIExtractor


@register
class BBoxFeat(Layer):
    __inject__ = ['roi_extractor']

    def __init__(self,
                 feat_in=1024,
                 feat_out=512,
                 roi_extractor=RoIExtractor().__dict__):
        super(BBoxFeat, self).__init__()
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIExtractor(**roi_extractor)

        self.res5 = resnet_blocks(
            "res5", ch_in=feat_in, ch_out=feat_out, count=3, stride=2)
        self.res5_pool = fluid.dygraph.Pool2D(
            pool_type='avg', global_pooling=True)

    def forward(self, inputs):
        roi_ext_out = self.roi_extractor(inputs['res4'], inputs['rois'],
                                         inputs['rois_nums'])
        x = roi_ext_out['rois_feat']
        y_res5 = self.res5(x)
        y = self.res5_pool(y_res5)
        y = fluid.layers.squeeze(y, axes=[2, 3])
        outs = {'res5': y_res5, "bbox_feat": y}
        outs.update(roi_ext_out)
        return outs


@register
class BBoxHead(Layer):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_feat']

    def __init__(self, num_classes=81, bbox_feat=BBoxFeat().__dict__):
        super(BBoxHead, self).__init__()
        self.num_classes = num_classes
        self.bbox_feat = bbox_feat
        if isinstance(bbox_feat, dict):
            print("bbox head: ", bbox_feat)
            self.bbox_feat = BBoxFeat(**bbox_feat)

        self.bbox_score = fluid.dygraph.Linear(
            input_dim=2048,
            output_dim=1 * self.num_classes,
            act=None,
            param_attr=ParamAttr(
                name='cls_score_w', initializer=Normal(
                    loc=0.0, scale=0.001)),
            bias_attr=ParamAttr(
                name='cls_score_b', learning_rate=2., regularizer=L2Decay(0.)))

        self.bbox_delta = fluid.dygraph.Linear(
            input_dim=2048,
            output_dim=4 * self.num_classes,
            act=None,
            param_attr=ParamAttr(
                name='bbox_pred_w', initializer=Normal(
                    loc=0.0, scale=0.01)),
            bias_attr=ParamAttr(
                name='bbox_pred_b', learning_rate=2., regularizer=L2Decay(0.)))

    def forward(self, inputs):
        bbox_feat_out = self.bbox_feat(inputs)
        x = bbox_feat_out['bbox_feat']
        bs = self.bbox_score(x)
        bd = self.bbox_delta(x)
        outs = {'bbox_score': bs, 'bbox_delta': bd}
        outs.update(bbox_feat_out)
        return outs

    def loss(self, inputs):
        # bbox cls  
        labels_int64 = fluid.layers.cast(
            x=inputs['labels_int32'], dtype='int64')
        labels_int64.stop_gradient = True
        bbox_score = fluid.layers.reshape(inputs['bbox_score'],
                                          (-1, self.num_classes))
        loss_bbox_cls = fluid.layers.softmax_with_cross_entropy(
            logits=bbox_score, label=labels_int64)
        loss_bbox_cls = fluid.layers.reduce_mean(
            loss_bbox_cls, name='loss_bbox_cls')
        # bbox reg
        loss_bbox_reg = fluid.layers.smooth_l1(
            x=inputs['bbox_delta'],
            y=inputs['bbox_targets'],
            inside_weight=inputs['bbox_inside_weights'],
            outside_weight=inputs['bbox_outside_weights'],
            sigma=1.0)
        loss_bbox_reg = fluid.layers.reduce_mean(
            loss_bbox_reg, name='loss_bbox_loc')

        return loss_bbox_cls, loss_bbox_reg
