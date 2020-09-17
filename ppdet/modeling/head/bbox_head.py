import paddle.fluid as fluid
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register


@register
class TwoFCHead(nn.Layer):

    __shared__ = ['num_stages']

    def __init__(self, in_dim=256, mlp_dim=1024, resolution=7, num_stages=1):
        super(TwoFCHead, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.num_stages = num_stages
        fan = in_dim * resolution * resolution
        self.fc6_list = []
        self.fc7_list = []
        for stage in range(num_stages):
            fc6_name = 'fc6_{}'.format(stage)
            fc7_name = 'fc7_{}'.format(stage)
            fc6 = self.add_sublayer(
                fc6_name,
                nn.Linear(
                    in_dim * resolution * resolution,
                    mlp_dim,
                    act='relu',
                    param_attr=ParamAttr(
                        #name='fc6_w',
                        initializer=Xavier(fan_out=fan)),
                    bias_attr=ParamAttr(
                        #name='fc6_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))
            fc7 = self.add_sublayer(
                fc7_name,
                nn.Linear(
                    mlp_dim,
                    mlp_dim,
                    act='relu',
                    param_attr=ParamAttr(
                        #name='fc7_w',
                        initializer=Xavier()),
                    bias_attr=ParamAttr(
                        #name='fc7_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))
            self.fc6_list.append(fc6)
            self.fc7_list.append(fc7)

    def forward(self, rois_feat, stage=0):
        rois_feat = fluid.layers.flatten(rois_feat)
        fc6 = self.fc6_list[stage](rois_feat)
        fc7 = self.fc7_list[stage](fc6)
        return fc7


@register
class BBoxFeat(nn.Layer):
    __inject__ = ['roi_extractor', 'head_feat']

    def __init__(self, roi_extractor, head_feat):
        super(BBoxFeat, self).__init__()
        self.roi_extractor = roi_extractor
        self.head_feat = head_feat

    def forward(self, body_feats, rois, spatial_scale, stage=0):
        rois_feat = self.roi_extractor(body_feats, rois, spatial_scale)
        bbox_feat = self.head_feat(rois_feat, stage)
        return bbox_feat


@register
class BBoxHead(nn.Layer):
    __shared__ = ['num_classes', 'num_stages']
    __inject__ = ['bbox_feat']

    def __init__(self,
                 bbox_feat,
                 in_feat=1024,
                 num_classes=81,
                 cls_agnostic=False,
                 num_stages=1,
                 with_pool=False):
        super(BBoxHead, self).__init__()
        self.num_classes = num_classes
        self.delta_dim = 2 if cls_agnostic else num_classes
        self.bbox_feat = bbox_feat
        self.num_stages = num_stages
        self.bbox_score_list = []
        self.bbox_delta_list = []
        self.with_pool = with_pool
        for stage in range(num_stages):
            score_name = 'bbox_score_{}'.format(stage)
            delta_name = 'bbox_delta_{}'.format(stage)
            bbox_score = self.add_sublayer(
                score_name,
                nn.Linear(
                    input_dim=in_feat,
                    output_dim=1 * self.num_classes,
                    act=None,
                    param_attr=ParamAttr(
                        #name='cls_score_w', 
                        initializer=Normal(
                            loc=0.0, scale=0.01)),
                    bias_attr=ParamAttr(
                        #name='cls_score_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))

            bbox_delta = self.add_sublayer(
                delta_name,
                nn.Linear(
                    input_dim=in_feat,
                    output_dim=4 * self.delta_dim,
                    act=None,
                    param_attr=ParamAttr(
                        #name='bbox_pred_w', 
                        initializer=Normal(
                            loc=0.0, scale=0.001)),
                    bias_attr=ParamAttr(
                        #name='bbox_pred_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))
            self.bbox_score_list.append(bbox_score)
            self.bbox_delta_list.append(bbox_delta)

    def forward(self, body_feats, rois, spatial_scale, stage=0):
        bbox_feat = self.bbox_feat(body_feats, rois, spatial_scale, stage)
        if self.with_pool:
            bbox_feat = F.pool2d(
                bbox_feat, pool_type='avg', global_pooling=True)
        bbox_head_out = []
        scores = self.bbox_score_list[stage](bbox_feat)
        deltas = self.bbox_delta_list[stage](bbox_feat)
        bbox_head_out.append((scores, deltas))
        return bbox_feat, bbox_head_out

    def _get_head_loss(self, score, delta, target):
        # bbox cls  
        labels_int64 = paddle.cast(x=target['labels_int32'], dtype='int64')
        labels_int64.stop_gradient = True
        loss_bbox_cls = F.softmax_with_cross_entropy(
            logits=score, label=labels_int64)
        loss_bbox_cls = paddle.reduce_mean(loss_bbox_cls)
        # bbox reg
        loss_bbox_reg = F.smooth_l1(
            x=delta,
            y=target['bbox_targets'],
            inside_weight=target['bbox_inside_weights'],
            outside_weight=target['bbox_outside_weights'],
            sigma=1.0)
        loss_bbox_reg = paddle.reduce_mean(loss_bbox_reg)
        return loss_bbox_cls, loss_bbox_reg

    def loss(self, bbox_head_out, targets):
        loss_bbox = {}
        for lvl, (bboxhead, target) in enumerate(zip(bbox_head_out, targets)):
            score, delta = bboxhead
            cls_name = 'loss_bbox_cls_{}'.format(lvl)
            reg_name = 'loss_bbox_reg_{}'.format(lvl)
            loss_bbox_cls, loss_bbox_reg = self._get_head_loss(score, delta,
                                                               target)
            loss_bbox[cls_name] = loss_bbox_cls
            loss_bbox[reg_name] = loss_bbox_reg
        return loss_bbox
