import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, MSRA
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, Pool2D
from ppdet.core.workspace import register
from ..backbone.resnet import Blocks
from ..ops import RoIExtractor


@register
class BBoxFeat(Layer):
    __shared__ = ['num_stages']
    __inject__ = ['roi_extractor']

    def __init__(self,
                 feat_in=1024,
                 feat_out=512,
                 roi_extractor=RoIExtractor().__dict__,
                 num_stages=3):
        super(BBoxFeat, self).__init__()
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIExtractor(**roi_extractor)
        self.num_stages = num_stages
        self.res5s = []
        for i in range(self.num_stages):
            if i == 0:
                postfix = ''
            else:
                postfix = '_' + str(i)
            # TODO: set norm type  
            res5 = Blocks(
                "res5" + postfix,
                ch_in=feat_in,
                ch_out=feat_out,
                count=3,
                stride=2)
            self.res5s.append(res5)
        self.res5_pool = fluid.dygraph.Pool2D(
            pool_type='avg', global_pooling=True)

    def forward(self, inputs):

        if inputs['mode'] == 'train':
            in_rois = inputs['proposal_' + str(inputs['stage'])]
            rois = in_rois['rois']
            rois_num = in_rois['rois_nums']
        elif inputs['mode'] == 'infer':
            rois = inputs['rpn_rois']
            rois_num = inputs['rpn_rois_nums']
        else:
            raise "BBoxFeat only support train or infer mode!"

        rois_feat = self.roi_extractor(inputs['res4'], rois, rois_num)
        # TODO: add others 
        y_res5 = self.res5s[inputs['stage']](rois_feat)
        y = self.res5_pool(y_res5)
        y = fluid.layers.squeeze(y, axes=[2, 3])
        outs = {
            'rois_feat': rois_feat,
            'res5': y_res5,
            "bbox_feat": y,
            'shared_res5_block': self.res5s[inputs['stage']],
            'shared_roi_extractor': self.roi_extractor
        }
        return outs


@register
class BBoxHead(Layer):
    __shared__ = ['num_classes', 'num_stages']
    __inject__ = ['bbox_feat']

    def __init__(self,
                 in_feat=2048,
                 num_classes=81,
                 cls_agnostic_bbox_reg=81,
                 bbox_feat=BBoxFeat().__dict__,
                 num_stages=3):
        super(BBoxHead, self).__init__()
        self.num_classes = num_classes
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_feat = bbox_feat
        if isinstance(bbox_feat, dict):
            self.bbox_feat = BBoxFeat(**bbox_feat)
        self.num_stages = num_stages
        self.bbox_scores = []
        self.bbox_deltas = []
        for i in range(self.num_stages):
            if i == 0:
                postfix = ''
            else:
                postfix = '_' + str(i)
            bbox_score = fluid.dygraph.Linear(
                input_dim=in_feat,
                output_dim=1 * self.num_classes,
                act=None,
                param_attr=ParamAttr(
                    name='cls_score_w' + postfix,
                    initializer=Normal(
                        loc=0.0, scale=0.001)),
                bias_attr=ParamAttr(
                    name='cls_score_b' + postfix,
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))

            bbox_delta = fluid.dygraph.Linear(
                input_dim=in_feat,
                output_dim=4 * self.cls_agnostic_bbox_reg,
                act=None,
                param_attr=ParamAttr(
                    name='bbox_pred_w' + postfix,
                    initializer=Normal(
                        loc=0.0, scale=0.01)),
                bias_attr=ParamAttr(
                    name='bbox_pred_b' + postfix,
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
            self.bbox_scores.append(bbox_score)
            self.bbox_deltas.append(bbox_delta)

    def forward(self, inputs):
        outs = self.bbox_feat(inputs)
        x = outs['bbox_feat']
        bs = self.bbox_scores[inputs['stage']](x)
        bd = self.bbox_deltas[inputs['stage']](x)
        outs.update({'bbox_score': bs, 'bbox_delta': bd})
        if inputs['stage'] == 0:
            outs.update({"cls_agnostic_bbox_reg": self.cls_agnostic_bbox_reg})
        if inputs['mode'] == 'infer':
            bbox_prob = fluid.layers.softmax(bs, use_cudnn=False)
            outs['bbox_prob'] = bbox_prob
        return outs

    def loss(self, inputs):
        bbox_out = inputs['bbox_head_' + str(inputs['stage'])]
        bbox_target = inputs['proposal_' + str(inputs['stage'])]

        # bbox cls  
        labels_int64 = fluid.layers.cast(
            x=bbox_target['labels_int32'], dtype='int64')
        labels_int64.stop_gradient = True
        bbox_score = fluid.layers.reshape(bbox_out['bbox_score'],
                                          (-1, self.num_classes))
        loss_bbox_cls = fluid.layers.softmax_with_cross_entropy(
            logits=bbox_score, label=labels_int64)
        loss_bbox_cls = fluid.layers.reduce_mean(
            loss_bbox_cls, name='loss_bbox_cls_' + str(inputs['stage']))

        # bbox reg
        loss_bbox_reg = fluid.layers.smooth_l1(
            x=bbox_out['bbox_delta'],
            y=bbox_target['bbox_targets'],
            inside_weight=bbox_target['bbox_inside_weights'],
            outside_weight=bbox_target['bbox_outside_weights'],
            sigma=1.0)
        loss_bbox_reg = fluid.layers.reduce_mean(
            loss_bbox_reg, name='loss_bbox_loc_' + str(inputs['stage']))

        return loss_bbox_cls, loss_bbox_reg
