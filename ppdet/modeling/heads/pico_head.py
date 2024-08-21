# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.modeling.ops import get_static_shape
from ..initializer import normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ..bbox_utils import bbox_center, batch_distance2bbox, bbox2distance
from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer
from .simota_head import OTAVFLHead
from .gfl_head import Integral, GFLHead
from ppdet.modeling.necks.csp_pan import DPModule

eps = 1e-9

__all__ = ['PicoHead', 'PicoHeadV2', 'PicoFeat']


def npu_avg_pool2d(feat, w, h):
    batch_size, channels, _, _ = feat.shape
    feat_flat = paddle.reshape(feat, [batch_size, channels, -1])
    feat_mean = paddle.mean(feat_flat, axis=2)
    feat_mean = paddle.reshape(
        feat_mean, [batch_size, channels, w, h])
    return feat_mean

class PicoSE(nn.Layer):
    def __init__(self, feat_channels):
        super(PicoSE, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvNormLayer(feat_channels, feat_channels, 1, 1)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        out = self.conv(feat * weight)
        return out


@register
class PicoFeat(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        share_cls_reg (bool): Whether to share the cls and reg output.
        act (str): The act of per layers.
        use_se (bool): Whether to use se module.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='hard_swish',
                 use_se=False):
        super(PicoFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.use_se = use_se
        self.cls_convs = []
        self.reg_convs = []

        if paddle.device.get_device().startswith("npu"):
            self.device = "npu"
        else:
            self.device = None
            
        if use_se:
            assert share_cls_reg == True, \
                'In the case of using se, share_cls_reg must be set to True'
            self.se = nn.LayerList()
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                cls_conv_dw = self.add_sublayer(
                    'cls_conv_dw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=5,
                        stride=1,
                        groups=feat_out,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.))
                cls_subnet_convs.append(cls_conv_dw)
                cls_conv_pw = self.add_sublayer(
                    'cls_conv_pw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.))
                cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    reg_conv_dw = self.add_sublayer(
                        'reg_conv_dw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=5,
                            stride=1,
                            groups=feat_out,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.))
                    reg_subnet_convs.append(reg_conv_dw)
                    reg_conv_pw = self.add_sublayer(
                        'reg_conv_pw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.))
                    reg_subnet_convs.append(reg_conv_pw)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)
            if use_se:
                self.se.append(PicoSE(feat_out))

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            reg_feat = cls_feat
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        if self.use_se:
            if self.device == "npu":
                avg_feat = npu_avg_pool2d(cls_feat, 1, 1)
            else:
                avg_feat = F.adaptive_avg_pool2d(cls_feat, (1, 1))
            se_feat = self.act_func(self.se[stage_idx](cls_feat, avg_feat))
            return cls_feat, se_feat
        return cls_feat, reg_feat


@register
class PicoHead(OTAVFLHead):
    """
    PicoHead
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'assigner', 'nms'
    ]
    __shared__ = ['num_classes', 'eval_size']

    def __init__(self,
                 conv_feat='PicoFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32],
                 prior_prob=0.01,
                 loss_class='VariFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 assigner='SimOTAAssigner',
                 reg_max=16,
                 feat_in_chan=96,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0,
                 eval_size=None):
        super(PicoHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            assigner=assigner,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.assigner = assigner
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.eval_size = eval_size
        self.device = paddle.device.get_device()

        self.use_sigmoid = self.loss_vfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # Clear the super class initialization
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None

        self.head_cls_list = []
        self.head_reg_list = []
        for i in range(len(fpn_stride)):
            head_cls = self.add_sublayer(
                "head_cls" + str(i),
                nn.Conv2D(
                    in_channels=self.feat_in_chan,
                    out_channels=self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.conv_feat.share_cls_reg else self.cls_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=ParamAttr(
                        initializer=Constant(value=bias_init_value))))
            self.head_cls_list.append(head_cls)
            if not self.conv_feat.share_cls_reg:
                head_reg = self.add_sublayer(
                    "head_reg" + str(i),
                    nn.Conv2D(
                        in_channels=self.feat_in_chan,
                        out_channels=4 * (self.reg_max + 1),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        weight_attr=ParamAttr(initializer=Normal(
                            mean=0., std=0.01)),
                        bias_attr=ParamAttr(initializer=Constant(value=0))))
                self.head_reg_list.append(head_reg)

        # initialize the anchor points
        if self.eval_size:
            self.anchor_points, self.stride_tensor = self._generate_anchors()

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"

        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(
                fpn_feats, export_post_process=export_post_process)

    def forward_train(self, fpn_feats):
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = paddle.split(
                    cls_logits,
                    [self.cls_out_channels, 4 * (self.reg_max + 1)],
                    axis=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)

            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return (cls_logits_list, bboxes_reg_list)

    def forward_eval(self, fpn_feats, export_post_process=True):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = paddle.split(
                    cls_logits,
                    [self.cls_out_channels, 4 * (self.reg_max + 1)],
                    axis=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)

            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score

            if not export_post_process:
                # Now only supports batch size = 1 in deploy
                # TODO(ygh): support batch size > 1
                cls_score_out = F.sigmoid(cls_score).reshape(
                    [1, self.cls_out_channels, -1]).transpose([0, 2, 1])
                bbox_pred = bbox_pred.reshape([1, (self.reg_max + 1) * 4,
                                               -1]).transpose([0, 2, 1])
            else:
                _, _, h, w = fpn_feat.shape
                l = h * w
                cls_score_out = F.sigmoid(
                    cls_score.reshape([-1, self.cls_out_channels, l]))
                bbox_pred = bbox_pred.transpose([0, 2, 3, 1])
                bbox_pred = self.distribution_project(bbox_pred)
                bbox_pred = bbox_pred.reshape([-1, l, 4])

            cls_logits_list.append(cls_score_out)
            bboxes_reg_list.append(bbox_pred)

        if export_post_process:
            cls_logits_list = paddle.concat(cls_logits_list, axis=-1)
            bboxes_reg_list = paddle.concat(bboxes_reg_list, axis=1)
            bboxes_reg_list = batch_distance2bbox(anchor_points,
                                                  bboxes_reg_list)
            bboxes_reg_list *= stride_tensor

        return (cls_logits_list, bboxes_reg_list)

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = math.ceil(self.eval_size[0] / stride)
                w = math.ceil(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.cell_offset
            shift_y = paddle.arange(end=h) + self.cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype='float32')
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [h * w, 1], stride, dtype='float32'))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def post_process(self,
                     head_outs,
                     scale_factor,
                     export_nms=True,
                     nms_cpu=False):
        pred_scores, pred_bboxes = head_outs
        if not export_nms:
            return pred_bboxes, pred_scores
        else:
            # rescale: [h_scale, w_scale] -> [w_scale, h_scale, w_scale, h_scale]
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            # scale bbox to origin image size.
            pred_bboxes /= scale_factor
            if nms_cpu:
                paddle.set_device("cpu")
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                paddle.set_device(self.device)
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num


@register
class PicoHeadV2(GFLHead):
    """
    PicoHeadV2
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'static_assigner', 'assigner', 'nms'
    ]
    __shared__ = ['num_classes', 'eval_size']

    def __init__(self,
                 conv_feat='PicoFeatV2',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32],
                 prior_prob=0.01,
                 use_align_head=True,
                 loss_class='VariFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 static_assigner_epoch=60,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 reg_max=16,
                 feat_in_chan=96,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0,
                 act='hard_swish',
                 grid_cell_scale=5.0,
                 eval_size=None):
        super(PicoHeadV2, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset, )
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner

        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.act = act
        self.grid_cell_scale = grid_cell_scale
        self.use_align_head = use_align_head
        self.cls_out_channels = self.num_classes
        self.eval_size = eval_size

        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # Clear the super class initialization
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None

        self.head_cls_list = nn.LayerList()
        self.head_reg_list = nn.LayerList()
        self.cls_align = nn.LayerList()

        for i in range(len(fpn_stride)):
            head_cls = self.add_sublayer(
                "head_cls" + str(i),
                nn.Conv2D(
                    in_channels=self.feat_in_chan,
                    out_channels=self.cls_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=ParamAttr(
                        initializer=Constant(value=bias_init_value))))
            self.head_cls_list.append(head_cls)
            head_reg = self.add_sublayer(
                "head_reg" + str(i),
                nn.Conv2D(
                    in_channels=self.feat_in_chan,
                    out_channels=4 * (self.reg_max + 1),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=ParamAttr(initializer=Constant(value=0))))
            self.head_reg_list.append(head_reg)
            if self.use_align_head:
                self.cls_align.append(
                    DPModule(
                        self.feat_in_chan,
                        1,
                        5,
                        act=self.act,
                        use_act_in_out=False))

        # initialize the anchor points
        if self.eval_size:
            self.anchor_points, self.stride_tensor = self._generate_anchors()

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"

        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(
                fpn_feats, export_post_process=export_post_process)

    def forward_train(self, fpn_feats):
        cls_score_list, reg_list, box_list = [], [], []
        for i, (fpn_feat, stride) in enumerate(zip(fpn_feats, self.fpn_stride)):
            b, _, h, w = get_static_shape(fpn_feat)
            # task decomposition
            conv_cls_feat, se_feat = self.conv_feat(fpn_feat, i)
            cls_logit = self.head_cls_list[i](se_feat)
            reg_pred = self.head_reg_list[i](se_feat)

            # cls prediction and alignment
            if self.use_align_head:
                cls_prob = F.sigmoid(self.cls_align[i](conv_cls_feat))
                cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            else:
                cls_score = F.sigmoid(cls_logit)

            cls_score_out = cls_score.transpose([0, 2, 3, 1])
            bbox_pred = reg_pred.transpose([0, 2, 3, 1])
            b, cell_h, cell_w, _ = cls_score_out.shape
            y, x = self.get_single_level_center_point(
                [cell_h, cell_w], stride, cell_offset=self.cell_offset)
            center_points = paddle.stack([x, y], axis=-1)
            cls_score_out = cls_score_out.reshape(
                [b, -1, self.cls_out_channels])
            bbox_pred = self.distribution_project(bbox_pred) * stride
            bbox_pred = bbox_pred.reshape([b, cell_h * cell_w, 4])
            bbox_pred = batch_distance2bbox(
                center_points, bbox_pred, max_shapes=None)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_list.append(reg_pred.flatten(2).transpose([0, 2, 1]))
            box_list.append(bbox_pred / stride)

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        box_list = paddle.concat(box_list, axis=1)
        reg_list = paddle.concat(reg_list, axis=1)
        return cls_score_list, reg_list, box_list, fpn_feats

    def forward_eval(self, fpn_feats, export_post_process=True):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        cls_score_list, box_list = [], []
        for i, (fpn_feat, stride) in enumerate(zip(fpn_feats, self.fpn_stride)):
            _, _, h, w = fpn_feat.shape
            # task decomposition
            conv_cls_feat, se_feat = self.conv_feat(fpn_feat, i)
            cls_logit = self.head_cls_list[i](se_feat)
            reg_pred = self.head_reg_list[i](se_feat)

            # cls prediction and alignment
            if self.use_align_head:
                cls_prob = F.sigmoid(self.cls_align[i](conv_cls_feat))
                cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            else:
                cls_score = F.sigmoid(cls_logit)

            if not export_post_process:
                # Now only supports batch size = 1 in deploy
                cls_score_list.append(
                    cls_score.reshape([1, self.cls_out_channels, -1]).transpose(
                        [0, 2, 1]))
                box_list.append(
                    reg_pred.reshape([1, (self.reg_max + 1) * 4, -1]).transpose(
                        [0, 2, 1]))
            else:
                l = h * w
                cls_score_out = cls_score.reshape(
                    [-1, self.cls_out_channels, l])
                bbox_pred = reg_pred.transpose([0, 2, 3, 1])
                bbox_pred = self.distribution_project(bbox_pred)
                bbox_pred = bbox_pred.reshape([-1, l, 4])
                cls_score_list.append(cls_score_out)
                box_list.append(bbox_pred)

        if export_post_process:
            cls_score_list = paddle.concat(cls_score_list, axis=-1)
            box_list = paddle.concat(box_list, axis=1)
            box_list = batch_distance2bbox(anchor_points, box_list)
            box_list *= stride_tensor

        return cls_score_list, box_list

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_regs, pred_bboxes, fpn_feats = head_outs
        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        num_imgs = gt_meta['im_id'].shape[0]
        pad_gt_mask = gt_meta['pad_gt_mask']

        anchors, _, num_anchors_list, stride_tensor_list = generate_anchors_for_grid_cell(
            fpn_feats, self.fpn_stride, self.grid_cell_scale, self.cell_offset)

        centers = bbox_center(anchors)

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_scores=gt_scores,
                pred_bboxes=pred_bboxes.detach() * stride_tensor_list)

        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor_list,
                centers,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_scores=gt_scores)

        assigned_bboxes /= stride_tensor_list

        centers_shape = centers.shape
        flatten_centers = centers.expand(
            [num_imgs, centers_shape[0], centers_shape[1]]).reshape([-1, 2])
        flatten_strides = stride_tensor_list.expand(
            [num_imgs, centers_shape[0], 1]).reshape([-1, 1])
        flatten_cls_preds = pred_scores.reshape([-1, self.num_classes])
        flatten_regs = pred_regs.reshape([-1, 4 * (self.reg_max + 1)])
        flatten_bboxes = pred_bboxes.reshape([-1, 4])
        flatten_bbox_targets = assigned_bboxes.reshape([-1, 4])
        flatten_labels = assigned_labels.reshape([-1])
        flatten_assigned_scores = assigned_scores.reshape(
            [-1, self.num_classes])

        pos_inds = paddle.nonzero(
            paddle.logical_and((flatten_labels >= 0),
                               (flatten_labels < self.num_classes)),
            as_tuple=False).squeeze(1)

        num_total_pos = len(pos_inds)

        if num_total_pos > 0:
            pos_bbox_targets = paddle.gather(
                flatten_bbox_targets, pos_inds, axis=0)
            pos_decode_bbox_pred = paddle.gather(
                flatten_bboxes, pos_inds, axis=0)
            pos_reg = paddle.gather(flatten_regs, pos_inds, axis=0)
            pos_strides = paddle.gather(flatten_strides, pos_inds, axis=0)
            pos_centers = paddle.gather(
                flatten_centers, pos_inds, axis=0) / pos_strides

            weight_targets = flatten_assigned_scores.detach()
            weight_targets = paddle.gather(
                weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)

            pred_corners = pos_reg.reshape([-1, self.reg_max + 1])
            target_corners = bbox2distance(pos_centers, pos_bbox_targets,
                                           self.reg_max).reshape([-1])
            # regression loss
            loss_bbox = paddle.sum(
                self.loss_bbox(pos_decode_bbox_pred,
                               pos_bbox_targets) * weight_targets)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets.expand([-1, 4]).reshape([-1]),
                avg_factor=4.0)
        else:
            loss_bbox = paddle.zeros([])
            loss_dfl = paddle.zeros([])

        avg_factor = flatten_assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(avg_factor)
            avg_factor = paddle.clip(
                avg_factor / paddle.distributed.get_world_size(), min=1)
        loss_vfl = self.loss_vfl(
            flatten_cls_preds, flatten_assigned_scores, avg_factor=avg_factor)

        loss_bbox = loss_bbox / avg_factor
        loss_dfl = loss_dfl / avg_factor

        loss_states = dict(
            loss_vfl=loss_vfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = math.ceil(self.eval_size[0] / stride)
                w = math.ceil(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.cell_offset
            shift_y = paddle.arange(end=h) + self.cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype='float32')
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [h * w, 1], stride, dtype='float32'))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def post_process(self,
                     head_outs,
                     scale_factor,
                     export_nms=True,
                     nms_cpu=False):
        pred_scores, pred_bboxes = head_outs
        if not export_nms:
            return pred_bboxes, pred_scores
        else:
            # rescale: [h_scale, w_scale] -> [w_scale, h_scale, w_scale, h_scale]
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            # scale bbox to origin image size.
            pred_bboxes /= scale_factor
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
