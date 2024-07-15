# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
from paddle.nn.initializer import Normal, Constant

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.cspresnet import ConvBNLayer, RepVggBlock
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['PPYOLOEHead', 'SimpleConvHead']


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish', attn_conv='convbn'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        if attn_conv == 'convbn':
            self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        elif attn_conv == 'repvgg':
            self.conv = RepVggBlock(feat_channels, feat_channels, act=act)
        else:
            self.conv = None
        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        if self.conv:
            return self.conv(feat * weight)
        else:
            return feat * weight


@register
class PPYOLOEHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 for_distill=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.sm_use = True
            self.reg_range = reg_range
        else:
            self.sm_use = False
            self.reg_range = (0, reg_max + 1)
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.use_shared_conv = use_shared_conv
        self.for_distill = for_distill
        self.is_teacher = False

        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
            self.stem_reg.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2D(
                    in_c, 4 * self.reg_channels, 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2D(self.reg_channels, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True
        self._init_weights()

        if self.for_distill:
            self.distill_pairs = {}

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        proj = paddle.linspace(self.reg_range[0], self.reg_range[1] - 1,
                               self.reg_channels).reshape(
                                   [1, self.reg_channels, 1, 1])
        self.proj_conv.weight.set_value(proj)
        self.proj_conv.weight.stop_gradient = True
        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def m_avg_pool2d(self, feat, w, h):
        batch_size, channels, _, _ = feat.shape
        feat_flat = paddle.reshape(feat, [batch_size, channels, -1])
        feat_mean = paddle.mean(feat_flat, axis=2)
        feat_mean = paddle.reshape(
            feat_mean, [batch_size, channels, w, h])
        return feat_mean

    def forward_train(self, feats, targets, aux_pred=None):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            if (paddle.get_device()[:3]=='npu'):
                avg_feat = self.m_avg_pool2d(feat, 1, 1)
            else:
                avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        if targets.get('is_teacher', False):
            pred_deltas, pred_dfls = self._bbox_decode_fake(reg_distri_list)
            return cls_score_list, pred_deltas * stride_tensor, pred_dfls

        if targets.get('get_data', False):
            pred_deltas, pred_dfls = self._bbox_decode_fake(reg_distri_list)
            return cls_score_list, pred_deltas * stride_tensor, pred_dfls

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets, aux_pred)

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            if (paddle.device.get_device()[:3]=='npu'):
                avg_feat = self.m_avg_pool2d(feat, 1, 1)
            else:
                avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None, aux_pred=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets, aux_pred)
        else:
            if targets is not None:
                # only for semi-det
                self.is_teacher = targets.get('is_teacher', False)
                if self.is_teacher:
                    return self.forward_train(feats, targets, aux_pred=None)
                else:
                    return self.forward_eval(feats)

            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist.transpose([0, 3, 1, 2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox_decode_fake(self, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        pred_dist_dfl = F.softmax(
            pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist_dfl.transpose([0, 3, 1, 2
                                                            ])).squeeze(1)
        return pred_dist, pred_dist_dfl

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(self.reg_range[0],
                                                self.reg_range[1] - 1 - 0.01)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = paddle.cast(target.floor(), 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left - lower_bound,
            reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right - lower_bound,
            reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)

        if self.for_distill:
            # only used for LD main_kd distill
            self.distill_pairs['mask_positive_select'] = mask_positive

        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.astype('int32').unsqueeze(-1).tile(
                [1, 1, 4]).astype('bool')
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).astype('int32').tile(
                [1, 1, self.reg_channels * 4]).astype('bool')
            pred_dist_pos = paddle.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_channels])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos,
                                     self.reg_range[0]) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
            if self.for_distill:
                self.distill_pairs['pred_bboxes_pos'] = pred_bboxes_pos
                self.distill_pairs['pred_dist_pos'] = pred_dist_pos
                self.distill_pairs['bbox_weight'] = bbox_weight
        else:
            loss_l1 = paddle.zeros([])
            loss_iou = paddle.zeros([])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta, aux_pred=None):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        if aux_pred is not None:
            pred_scores_aux = aux_pred[0]
            pred_bboxes_aux = self._bbox_decode(anchor_points_s, aux_pred[1])

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            if self.sm_use:
                # only used in smalldet of PPYOLOE-SOD model
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            else:
                if aux_pred is None:
                    if not hasattr(self, "assigned_labels"):
                        assigned_labels, assigned_bboxes, assigned_scores = \
                            self.assigner(
                            pred_scores.detach(),
                            pred_bboxes.detach() * stride_tensor,
                            anchor_points,
                            num_anchors_list,
                            gt_labels,
                            gt_bboxes,
                            pad_gt_mask,
                            bg_index=self.num_classes)
                        if self.for_distill:
                            self.assigned_labels = assigned_labels
                            self.assigned_bboxes = assigned_bboxes
                            self.assigned_scores = assigned_scores

                    else:
                        # only used in distill
                        assigned_labels = self.assigned_labels
                        assigned_bboxes = self.assigned_bboxes
                        assigned_scores = self.assigned_scores

                else:
                    assigned_labels, assigned_bboxes, assigned_scores = \
                            self.assigner(
                            pred_scores_aux.detach(),
                            pred_bboxes_aux.detach() * stride_tensor,
                            anchor_points,
                            num_anchors_list,
                            gt_labels,
                            gt_bboxes,
                            pad_gt_mask,
                            bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor

        assign_out_dict = self.get_loss_from_assign(
            pred_scores, pred_distri, pred_bboxes, anchor_points_s,
            assigned_labels, assigned_bboxes, assigned_scores, alpha_l)

        if aux_pred is not None:
            assign_out_dict_aux = self.get_loss_from_assign(
                aux_pred[0], aux_pred[1], pred_bboxes_aux, anchor_points_s,
                assigned_labels, assigned_bboxes, assigned_scores, alpha_l)
            loss = {}
            for key in assign_out_dict.keys():
                loss[key] = assign_out_dict[key] + assign_out_dict_aux[key]
        else:
            loss = assign_out_dict

        return loss

    def get_loss_from_assign(self, pred_scores, pred_distri, pred_bboxes,
                             anchor_points_s, assigned_labels, assigned_bboxes,
                             assigned_scores, alpha_l):
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        if self.for_distill:
            self.distill_pairs['pred_cls_scores'] = pred_scores
            self.distill_pairs['pos_num'] = assigned_scores_sum
            self.distill_pairs['assigned_scores'] = assigned_scores

            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            self.distill_pairs['target_labels'] = one_hot_label

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])],
                axis=-1), None, None
        else:
            # scale bbox to origin
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores, None
            else:
                bbox_pred, bbox_num, nms_keep_idx = self.nms(pred_bboxes,
                                                             pred_scores)
                return bbox_pred, bbox_num, nms_keep_idx


def get_activation(name="LeakyReLU"):
    if name == "silu":
        module = nn.Silu()
    elif name == "relu":
        module = nn.ReLU()
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm_type='gn',
                 activation="LeakyReLU"):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'syncbn', 'gn', None]
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
            weight_attr=ParamAttr(initializer=KaimingNormal()))

        if norm_type in ['bn', 'sync_bn', 'syncbn']:
            self.norm = nn.BatchNorm2D(out_channels)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        else:
            self.norm = None

        self.act = get_activation(activation)

    def forward(self, x):
        y = self.conv(x)
        if self.norm is not None:
            y = self.norm(y)
        y = self.act(y)
        return y


class ScaleReg(nn.Layer):
    """
    Parameter for scaling the regression outputs.
    """

    def __init__(self, scale=1.0):
        super(ScaleReg, self).__init__()
        scale = paddle.to_tensor(scale)
        self.scale = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Assign(scale))

    def forward(self, x):
        return x * self.scale


@register
class SimpleConvHead(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 feat_in=288,
                 feat_out=288,
                 num_convs=1,
                 fpn_strides=[32, 16, 8, 4],
                 norm_type='gn',
                 act='LeakyReLU',
                 prior_prob=0.01,
                 reg_max=16):
        super(SimpleConvHead, self).__init__()
        self.num_classes = num_classes
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.num_convs = num_convs
        self.fpn_strides = fpn_strides
        self.reg_max = reg_max

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        for i in range(self.num_convs):
            in_c = feat_in if i == 0 else feat_out
            self.cls_convs.append(
                ConvNormLayer(
                    in_c,
                    feat_out,
                    3,
                    stride=1,
                    padding=1,
                    norm_type=norm_type,
                    activation=act))
            self.reg_convs.append(
                ConvNormLayer(
                    in_c,
                    feat_out,
                    3,
                    stride=1,
                    padding=1,
                    norm_type=norm_type,
                    activation=act))

        bias_cls = bias_init_with_prob(prior_prob)
        self.gfl_cls = nn.Conv2D(
            feat_out,
            self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=bias_cls)))
        self.gfl_reg = nn.Conv2D(
            feat_out,
            4 * (self.reg_max + 1),
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)),
            bias_attr=ParamAttr(initializer=Constant(value=0)))

        self.scales = nn.LayerList()
        for i in range(len(self.fpn_strides)):
            self.scales.append(ScaleReg(1.0))

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

            cls_score = self.gfl_cls(cls_feat)
            cls_score = F.sigmoid(cls_score)
            cls_score = cls_score.flatten(2).transpose([0, 2, 1])
            cls_scores.append(cls_score)

            bbox_pred = scale(self.gfl_reg(reg_feat))
            bbox_pred = bbox_pred.flatten(2).transpose([0, 2, 1])
            bbox_preds.append(bbox_pred)

        cls_scores = paddle.concat(cls_scores, axis=1)
        bbox_preds = paddle.concat(bbox_preds, axis=1)
        return cls_scores, bbox_preds
