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

from ..losses import ProbIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_, vector_
from ppdet.modeling.backbones.cspresnet import ConvBNLayer
from ppdet.modeling.ops import get_static_shape, get_act_fn, anchor_generator
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['PPYOLOERHead']


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.01)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


@register
class PPYOLOERHead(nn.Layer):
    __shared__ = ['num_classes', 'trt', 'export_onnx']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=15,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_offset=0.5,
                 angle_max=90,
                 use_varifocal_loss=True,
                 static_assigner_epoch=4,
                 trt=False,
                 export_onnx=False,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={'class': 1.0,
                              'iou': 2.5,
                              'dfl': 0.05}):
        super(PPYOLOERHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_offset = grid_cell_offset
        self.angle_max = angle_max
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.half_pi_bin = self.half_pi / angle_max
        self.iou_loss = ProbIoULoss()
        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        self.stem_angle = nn.LayerList()
        trt = False if export_onnx else trt
        self.export_onnx = export_onnx
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.trt = trt
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
            self.stem_angle.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        self.pred_angle = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2D(in_c, 4, 3, padding=1))
            self.pred_angle.append(
                nn.Conv2D(
                    in_c, self.angle_max + 1, 3, padding=1))
        self.angle_proj_conv = nn.Conv2D(
            self.angle_max + 1, 1, 1, bias_attr=False)
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_angle = [10.] + [1.] * self.angle_max
        for cls_, reg_, angle_ in zip(self.pred_cls, self.pred_reg,
                                      self.pred_angle):
            normal_(cls_.weight, std=0.01)
            constant_(cls_.bias, bias_cls)
            normal_(reg_.weight, std=0.01)
            constant_(reg_.bias)
            constant_(angle_.weight)
            vector_(angle_.bias, bias_angle)

        angle_proj = paddle.linspace(0, self.angle_max, self.angle_max + 1)
        self.angle_proj = angle_proj * self.half_pi_bin
        self.angle_proj_conv.weight.set_value(
            self.angle_proj.reshape([1, self.angle_max + 1, 1, 1]))
        self.angle_proj_conv.weight.stop_gradient = True

    def _generate_anchors(self, feats):
        if self.trt:
            anchor_points = []
            for feat, stride in zip(feats, self.fpn_strides):
                _, _, h, w = feat.shape
                anchor, _ = anchor_generator(
                    feat,
                    stride * 4,
                    1.0, [1.0, 1.0, 1.0, 1.0], [stride, stride],
                    offset=0.5)
                x1, y1, x2, y2 = paddle.split(anchor, 4, axis=-1)
                xc = (x1 + x2 + 1) / 2
                yc = (y1 + y2 + 1) / 2
                anchor_point = paddle.concat(
                    [xc, yc], axis=-1).reshape((1, h * w, 2))
                anchor_points.append(anchor_point)
            anchor_points = paddle.concat(anchor_points, axis=1)
            return anchor_points, None, None
        else:
            anchor_points = []
            stride_tensor = []
            num_anchors_list = []
            for feat, stride in zip(feats, self.fpn_strides):
                _, _, h, w = feat.shape
                shift_x = (paddle.arange(end=w) + 0.5) * stride
                shift_y = (paddle.arange(end=h) + 0.5) * stride
                shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
                anchor_point = paddle.cast(
                    paddle.stack(
                        [shift_x, shift_y], axis=-1), dtype='float32')
                anchor_points.append(anchor_point.reshape([1, -1, 2]))
                stride_tensor.append(
                    paddle.full(
                        [1, h * w, 1], stride, dtype='float32'))
                num_anchors_list.append(h * w)
            anchor_points = paddle.concat(anchor_points, axis=1)
            stride_tensor = paddle.concat(stride_tensor, axis=1)
            return anchor_points, stride_tensor, num_anchors_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        anchor_points, stride_tensor, num_anchors_list = self._generate_anchors(
            feats)

        cls_score_list, reg_dist_list, reg_angle_list = [], [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_angle = self.pred_angle[i](self.stem_angle[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_dist_list.append(reg_dist.flatten(2).transpose([0, 2, 1]))
            reg_angle_list.append(reg_angle.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        reg_angle_list = paddle.concat(reg_angle_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_dist_list, reg_angle_list, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        cls_score_list, reg_box_list = [], []
        anchor_points, _, _ = self._generate_anchors(feats)
        for i, (feat, stride) in enumerate(zip(feats, self.fpn_strides)):
            b, _, h, w = feat.shape
            l = h * w
            # cls
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            # reg
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_xy, reg_wh = paddle.split(reg_dist, 2, axis=1)
            reg_xy = reg_xy * stride
            reg_wh = (F.elu(reg_wh) + 1.) * stride
            reg_angle = self.pred_angle[i](self.stem_angle[i](feat, avg_feat))
            reg_angle = self.angle_proj_conv(F.softmax(reg_angle, axis=1))
            reg_box = paddle.concat([reg_xy, reg_wh, reg_angle], axis=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_box_list.append(reg_box.reshape([b, 5, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_box_list = paddle.concat(reg_box_list, axis=-1).transpose([0, 2, 1])
        reg_xy, reg_wha = paddle.split(reg_box_list, [2, 3], axis=-1)
        reg_xy = reg_xy + anchor_points
        reg_box_list = paddle.concat([reg_xy, reg_wha], axis=-1)
        return cls_score_list, reg_box_list

    def _bbox_decode(self, points, pred_dist, pred_angle, stride_tensor):
        # predict vector to x, y, w, h, angle
        b, l = pred_angle.shape[:2]
        xy, wh = paddle.split(pred_dist, 2, axis=-1)
        xy = xy * stride_tensor + points
        wh = (F.elu(wh) + 1.) * stride_tensor
        angle = F.softmax(pred_angle.reshape([b, l, 1, self.angle_max + 1
                                              ])).matmul(self.angle_proj)
        return paddle.concat([xy, wh, angle], axis=-1)

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_dist, pred_angle, \
        anchor_points, num_anchors_list, stride_tensor = head_outs
        # [B, N, 5] -> [B, N, 5]
        pred_bboxes = self._bbox_decode(anchor_points, pred_dist, pred_angle,
                                        stride_tensor)
        gt_labels = gt_meta['gt_class']
        # [B, N, 5]
        gt_bboxes = gt_meta['gt_rbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchor_points,
                    stride_tensor,
                    num_anchors_list,
                    gt_labels,
                    gt_meta['gt_bbox'],
                    gt_bboxes,
                    pad_gt_mask,
                    self.num_classes,
                    pred_bboxes.detach()
                )
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach(),
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        alpha_l = -1
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
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1.)
        else:
            assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_iou, loss_dfl = self._bbox_loss(pred_angle, pred_bboxes,
                                             anchor_points, assigned_labels,
                                             assigned_bboxes, assigned_scores,
                                             assigned_scores_sum, stride_tensor)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl
        }
        return out_dict

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

    @staticmethod
    def _df_loss(pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_angle, pred_bboxes, anchor_points,
                   assigned_labels, assigned_bboxes, assigned_scores,
                   assigned_scores_sum, stride_tensor):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 5])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 5])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 5])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).reshape([-1])

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # dfl
            angle_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.angle_max + 1])
            pred_angle_pos = paddle.masked_select(
                pred_angle, angle_mask).reshape([-1, self.angle_max + 1])
            assigned_angle_pos = (
                assigned_bboxes_pos[:, 4] /
                self.half_pi_bin).clip(0, self.angle_max - 0.01)
            loss_dfl = self._df_loss(pred_angle_pos, assigned_angle_pos)
        else:
            loss_iou = pred_bboxes.sum() * 0.
            loss_dfl = paddle.zeros([1])

        return loss_iou, loss_dfl

    def _box2corners(self, pred_bboxes):
        """ convert (x, y, w, h, angle) to (x1, y1, x2, y2, x3, y3, x4, y4)

        Args:
            pred_bboxes (Tensor): [B, N, 5]
        
        Returns:
            polys (Tensor): [B, N, 8]
        """
        x, y, w, h, angle = paddle.split(pred_bboxes, 5, axis=-1)
        cos_a_half = paddle.cos(angle) * 0.5
        sin_a_half = paddle.sin(angle) * 0.5
        w_x = cos_a_half * w
        w_y = sin_a_half * w
        h_x = -sin_a_half * h
        h_y = cos_a_half * h
        return paddle.concat(
            [
                x + w_x + h_x, y + w_y + h_y, x - w_x + h_x, y - w_y + h_y,
                x - w_x - h_x, y - w_y - h_y, x + w_x - h_x, y + w_y - h_y
            ],
            axis=-1)

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_bboxes = head_outs
        # [B, N, 5] -> [B, N, 8]
        pred_bboxes = self._box2corners(pred_bboxes)
        # scale bbox to origin
        scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
        scale_factor = paddle.concat(
            [
                scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x,
                scale_y
            ],
            axis=-1).reshape([-1, 1, 8])
        pred_bboxes /= scale_factor
        if self.export_onnx:
            return pred_bboxes, pred_scores, None
        bbox_pred, bbox_num, nms_keep_idx = self.nms(pred_bboxes,
                                                           pred_scores)
        return bbox_pred, bbox_num, nms_keep_idx
