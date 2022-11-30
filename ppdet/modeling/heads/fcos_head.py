# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer, MultiClassNMS

__all__ = ['FCOSFeat', 'FCOSHead']


class ScaleReg(nn.Layer):
    """
    Parameter for scaling the regression outputs.
    """

    def __init__(self):
        super(ScaleReg, self).__init__()
        self.scale_reg = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=1.)),
            dtype="float32")

    def forward(self, inputs):
        out = inputs * self.scale_reg
        return out


@register
class FCOSFeat(nn.Layer):
    """
    FCOSFeat of FCOS

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the FCOSFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        use_dcn (bool): Whether to use dcn in tower or not.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=256,
                 num_convs=4,
                 norm_type='bn',
                 use_dcn=False):
        super(FCOSFeat, self).__init__()
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.cls_subnet_convs = []
        self.reg_subnet_convs = []
        for i in range(self.num_convs):
            in_c = feat_in if i == 0 else feat_out

            cls_conv_name = 'fcos_head_cls_tower_conv_{}'.format(i)
            cls_conv = self.add_sublayer(
                cls_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    use_dcn=use_dcn,
                    bias_on=True,
                    lr_scale=2.))
            self.cls_subnet_convs.append(cls_conv)

            reg_conv_name = 'fcos_head_reg_tower_conv_{}'.format(i)
            reg_conv = self.add_sublayer(
                reg_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    use_dcn=use_dcn,
                    bias_on=True,
                    lr_scale=2.))
            self.reg_subnet_convs.append(reg_conv)

    def forward(self, fpn_feat):
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(self.num_convs):
            cls_feat = F.relu(self.cls_subnet_convs[i](cls_feat))
            reg_feat = F.relu(self.reg_subnet_convs[i](reg_feat))
        return cls_feat, reg_feat


@register
class FCOSHead(nn.Layer):
    """
    FCOSHead
    Args:
        num_classes (int): Number of classes
        fcos_feat (object): Instance of 'FCOSFeat'
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        norm_reg_targets (bool): Normalization the regression target if true
        centerness_on_reg (bool): The prediction of centerness on regression or clssification branch
        num_shift (float): Relative offset between the center of the first shift and the top-left corner of img
        fcos_loss (object): Instance of 'FCOSLoss'
        nms (object): Instance of 'MultiClassNMS'
        trt (bool): Whether to use trt in nms of deploy
    """
    __inject__ = ['fcos_feat', 'fcos_loss', 'nms']
    __shared__ = ['num_classes', 'trt']

    def __init__(self,
                 num_classes=80,
                 fcos_feat='FCOSFeat',
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 multiply_strides_reg_targets=False,
                 norm_reg_targets=True,
                 centerness_on_reg=True,
                 num_shift=0.5,
                 sqrt_score=False,
                 fcos_loss='FCOSLoss',
                 nms='MultiClassNMS',
                 trt=False):
        super(FCOSHead, self).__init__()
        self.fcos_feat = fcos_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.fcos_loss = fcos_loss
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.multiply_strides_reg_targets = multiply_strides_reg_targets
        self.num_shift = num_shift
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.sqrt_score = sqrt_score
        self.is_teacher = False

        conv_cls_name = "fcos_head_cls"
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.fcos_head_cls = self.add_sublayer(
            conv_cls_name,
            nn.Conv2D(
                in_channels=256,
                out_channels=self.num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(
                    initializer=Constant(value=bias_init_value))))

        conv_reg_name = "fcos_head_reg"
        self.fcos_head_reg = self.add_sublayer(
            conv_reg_name,
            nn.Conv2D(
                in_channels=256,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

        conv_centerness_name = "fcos_head_centerness"
        self.fcos_head_centerness = self.add_sublayer(
            conv_centerness_name,
            nn.Conv2D(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

        self.scales_regs = []
        for i in range(len(self.fpn_stride)):
            lvl = int(math.log(int(self.fpn_stride[i]), 2))
            feat_name = 'p{}_feat'.format(lvl)
            scale_reg = self.add_sublayer(feat_name, ScaleReg())
            self.scales_regs.append(scale_reg)

    def _compute_locations_by_level(self, fpn_stride, feature, num_shift=0.5):
        """
        Compute locations of anchor points of each FPN layer
        Args:
            fpn_stride (int): The stride of current FPN feature map
            feature (Tensor): Tensor of current FPN feature map
        Return:
            Anchor points locations of current FPN feature map
        """
        h, w = feature.shape[2], feature.shape[3]
        shift_x = paddle.arange(0, w * fpn_stride, fpn_stride)
        shift_y = paddle.arange(0, h * fpn_stride, fpn_stride)
        shift_x = paddle.unsqueeze(shift_x, axis=0)
        shift_y = paddle.unsqueeze(shift_y, axis=1)
        shift_x = paddle.expand(shift_x, shape=[h, w])
        shift_y = paddle.expand(shift_y, shape=[h, w])

        shift_x = paddle.reshape(shift_x, shape=[-1])
        shift_y = paddle.reshape(shift_y, shape=[-1])
        location = paddle.stack(
            [shift_x, shift_y], axis=-1) + float(fpn_stride * num_shift)
        return location

    def forward(self, fpn_feats, targets=None):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        cls_logits_list = []
        bboxes_reg_list = []
        centerness_list = []
        for scale_reg, fpn_stride, fpn_feat in zip(self.scales_regs,
                                                   self.fpn_stride, fpn_feats):
            fcos_cls_feat, fcos_reg_feat = self.fcos_feat(fpn_feat)
            cls_logits = self.fcos_head_cls(fcos_cls_feat)
            bbox_reg = scale_reg(self.fcos_head_reg(fcos_reg_feat))
            if self.centerness_on_reg:
                centerness = self.fcos_head_centerness(fcos_reg_feat)
            else:
                centerness = self.fcos_head_centerness(fcos_cls_feat)
            if self.norm_reg_targets:
                bbox_reg = F.relu(bbox_reg)
                if self.multiply_strides_reg_targets:
                    bbox_reg = bbox_reg * fpn_stride
                else:
                    if not self.training or targets.get(
                            'get_data',
                            False) or targets.get('is_teacher', False):
                        bbox_reg = bbox_reg * fpn_stride
            else:
                bbox_reg = paddle.exp(bbox_reg)
            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)
            centerness_list.append(centerness)

        if targets is not None:
            self.is_teacher = targets.get('is_teacher', False)
            if self.is_teacher:
                return [cls_logits_list, bboxes_reg_list, centerness_list]

        if self.training and targets is not None:
            get_data = targets.get('get_data', False)
            if get_data:
                return [cls_logits_list, bboxes_reg_list, centerness_list]

            losses = {}
            fcos_head_outs = [cls_logits_list, bboxes_reg_list, centerness_list]
            losses_fcos = self.get_loss(fcos_head_outs, targets)
            losses.update(losses_fcos)

            total_loss = paddle.add_n(list(losses.values()))
            losses.update({'loss': total_loss})
            return losses
        else:
            # eval or infer
            locations_list = []
            for fpn_stride, feature in zip(self.fpn_stride, fpn_feats):
                location = self._compute_locations_by_level(fpn_stride, feature,
                                                            self.num_shift)
                locations_list.append(location)

            fcos_head_outs = [
                locations_list, cls_logits_list, bboxes_reg_list,
                centerness_list
            ]
            return fcos_head_outs

    def get_loss(self, fcos_head_outs, targets):
        cls_logits, bboxes_reg, centerness = fcos_head_outs

        # get labels,reg_target,centerness
        tag_labels, tag_bboxes, tag_centerness = [], [], []
        for i in range(len(self.fpn_stride)):
            k_lbl = 'labels{}'.format(i)
            if k_lbl in targets:
                tag_labels.append(targets[k_lbl])
            k_box = 'reg_target{}'.format(i)
            if k_box in targets:
                tag_bboxes.append(targets[k_box])
            k_ctn = 'centerness{}'.format(i)
            if k_ctn in targets:
                tag_centerness.append(targets[k_ctn])

        losses_fcos = self.fcos_loss(cls_logits, bboxes_reg, centerness,
                                     tag_labels, tag_bboxes, tag_centerness)
        return losses_fcos

    def _post_process_by_level(self,
                               locations,
                               box_cls,
                               box_reg,
                               box_ctn,
                               sqrt_score=False):
        box_scores = F.sigmoid(box_cls).flatten(2).transpose([0, 2, 1])
        box_centerness = F.sigmoid(box_ctn).flatten(2).transpose([0, 2, 1])
        pred_scores = box_scores * box_centerness
        if sqrt_score:
            pred_scores = paddle.sqrt(pred_scores)

        box_reg_ch_last = box_reg.flatten(2).transpose([0, 2, 1])
        box_reg_decoding = paddle.stack(
            [
                locations[:, 0] - box_reg_ch_last[:, :, 0],
                locations[:, 1] - box_reg_ch_last[:, :, 1],
                locations[:, 0] + box_reg_ch_last[:, :, 2],
                locations[:, 1] + box_reg_ch_last[:, :, 3]
            ],
            axis=1)
        pred_boxes = box_reg_decoding.transpose([0, 2, 1])

        return pred_scores, pred_boxes

    def post_process(self, fcos_head_outs, scale_factor):
        locations, cls_logits, bboxes_reg, centerness = fcos_head_outs
        pred_bboxes, pred_scores = [], []

        for pts, cls, reg, ctn in zip(locations, cls_logits, bboxes_reg,
                                      centerness):
            scores, boxes = self._post_process_by_level(pts, cls, reg, ctn,
                                                        self.sqrt_score)
            pred_scores.append(scores)
            pred_bboxes.append(boxes)
        pred_bboxes = paddle.concat(pred_bboxes, axis=1)
        pred_scores = paddle.concat(pred_scores, axis=1)

        # scale bbox to origin
        scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
        scale_factor = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y], axis=-1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor

        pred_scores = pred_scores.transpose([0, 2, 1])
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num
