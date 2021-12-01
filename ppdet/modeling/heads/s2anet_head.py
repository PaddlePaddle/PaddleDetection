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
#
# The code is based on https://github.com/csuhan/s2anet/blob/master/mmdet/models/anchor_heads_rotated/s2anet_head.py

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant
from ppdet.core.workspace import register
from ppdet.modeling import ops
from ppdet.modeling import bbox_utils
from ppdet.modeling.proposal_generator.target_layer import RBoxAssigner
import numpy as np


class S2ANetAnchorGenerator(nn.Layer):
    """
    AnchorGenerator by paddle
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        super(S2ANetAnchorGenerator, self).__init__()
        self.base_size = base_size
        self.scales = paddle.to_tensor(scales)
        self.ratios = paddle.to_tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.shape[0]

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = paddle.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:] * self.scales[:]).reshape([-1])
            hs = (h * h_ratios[:] * self.scales[:]).reshape([-1])
        else:
            ws = (w * self.scales[:] * w_ratios[:]).reshape([-1])
            hs = (h * self.scales[:] * h_ratios[:]).reshape([-1])

        base_anchors = paddle.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            axis=-1)
        base_anchors = paddle.round(base_anchors)
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = paddle.meshgrid(y, x)
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def forward(self, featmap_size, stride=16):
        # featmap_size*stride project it to original area

        feat_h = featmap_size[0]
        feat_w = featmap_size[1]
        shift_x = paddle.arange(0, feat_w, 1, 'int32') * stride
        shift_y = paddle.arange(0, feat_h, 1, 'int32') * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = paddle.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)

        all_anchors = self.base_anchors[:, :] + shifts[:, :]
        all_anchors = all_anchors.reshape([feat_h * feat_w, 4])
        return all_anchors

    def valid_flags(self, featmap_size, valid_size):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = paddle.zeros([feat_w], dtype='int32')
        valid_y = paddle.zeros([feat_h], dtype='int32')
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = paddle.reshape(valid, [-1, 1])
        valid = paddle.expand(valid, [-1, self.num_base_anchors]).reshape([-1])
        return valid


class AlignConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.align_conv = paddle.vision.ops.DeformConv2D(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=Normal(0, 0.01)),
            bias_attr=None)

    @paddle.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """
        Args:
            anchors: [M,5] xc,yc,w,h,angle
            featmap_size: (feat_h, feat_w)
            stride: 8
        Returns:

        """
        anchors = paddle.reshape(anchors, [-1, 5])  # (NA,5)
        dtype = anchors.dtype
        feat_h = featmap_size[0]
        feat_w = featmap_size[1]
        pad = (self.kernel_size - 1) // 2
        idx = paddle.arange(-pad, pad + 1, dtype=dtype)

        yy, xx = paddle.meshgrid(idx, idx)
        xx = paddle.reshape(xx, [-1])
        yy = paddle.reshape(yy, [-1])

        # get sampling locations of default conv
        xc = paddle.arange(0, feat_w, dtype=dtype)
        yc = paddle.arange(0, feat_h, dtype=dtype)
        yc, xc = paddle.meshgrid(yc, xc)

        xc = paddle.reshape(xc, [-1, 1])
        yc = paddle.reshape(yc, [-1, 1])
        x_conv = xc + xx
        y_conv = yc + yy

        # get sampling locations of anchors
        # x_ctr, y_ctr, w, h, a = np.unbind(anchors, dim=1)
        x_ctr = anchors[:, 0]
        y_ctr = anchors[:, 1]
        w = anchors[:, 2]
        h = anchors[:, 3]
        a = anchors[:, 4]

        x_ctr = paddle.reshape(x_ctr, [-1, 1])
        y_ctr = paddle.reshape(y_ctr, [-1, 1])
        w = paddle.reshape(w, [-1, 1])
        h = paddle.reshape(h, [-1, 1])
        a = paddle.reshape(a, [-1, 1])

        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = paddle.cos(a), paddle.sin(a)
        dw, dh = w_s / self.kernel_size, h_s / self.kernel_size
        x, y = dw * xx, dh * yy
        xr = cos * x - sin * y
        yr = sin * x + cos * y
        x_anchor, y_anchor = xr + x_ctr, yr + y_ctr
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        offset = paddle.stack([offset_y, offset_x], axis=-1)
        offset = paddle.reshape(
            offset, [feat_h * feat_w, self.kernel_size * self.kernel_size * 2])
        offset = paddle.transpose(offset, [1, 0])
        offset = paddle.reshape(
            offset,
            [1, self.kernel_size * self.kernel_size * 2, feat_h, feat_w])
        return offset

    def forward(self, x, refine_anchors, featmap_size, stride):
        offset = self.get_offset(refine_anchors, featmap_size, stride)
        x = F.relu(self.align_conv(x, offset))
        return x


@register
class S2ANetHead(nn.Layer):
    """
    S2Anet head
    Args:
        stacked_convs (int): number of stacked_convs
        feat_in (int): input channels of feat
        feat_out (int): output channels of feat
        num_classes (int): num_classes
        anchor_strides (list): stride of anchors
        anchor_scales (list): scale of anchors
        anchor_ratios (list): ratios of anchors
        target_means (list): target_means
        target_stds (list): target_stds
        align_conv_type (str): align_conv_type ['Conv', 'AlignConv']
        align_conv_size (int): kernel size of align_conv
        use_sigmoid_cls (bool): use sigmoid_cls or not
        reg_loss_weight (list): loss weight for regression
    """
    __shared__ = ['num_classes']
    __inject__ = ['anchor_assign']

    def __init__(self,
                 stacked_convs=2,
                 feat_in=256,
                 feat_out=256,
                 num_classes=15,
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_scales=[4],
                 anchor_ratios=[1.0],
                 target_means=0.0,
                 target_stds=1.0,
                 align_conv_type='AlignConv',
                 align_conv_size=3,
                 use_sigmoid_cls=True,
                 anchor_assign=RBoxAssigner().__dict__,
                 reg_loss_weight=[1.0, 1.0, 1.0, 1.0, 1.1],
                 cls_loss_weight=[1.1, 1.05],
                 reg_loss_type='l1'):
        super(S2ANetHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.anchor_list = None
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_strides = paddle.to_tensor(anchor_strides)
        self.anchor_base_sizes = list(anchor_strides)
        self.means = paddle.ones(shape=[5]) * target_means
        self.stds = paddle.ones(shape=[5]) * target_stds
        assert align_conv_type in ['AlignConv', 'Conv', 'DCN']
        self.align_conv_type = align_conv_type
        self.align_conv_size = align_conv_size

        self.use_sigmoid_cls = use_sigmoid_cls
        self.cls_out_channels = num_classes if self.use_sigmoid_cls else 1
        self.sampling = False
        self.anchor_assign = anchor_assign
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.alpha = 1.0
        self.beta = 1.0
        self.reg_loss_type = reg_loss_type
        self.s2anet_head_out = None

        # anchor
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                S2ANetAnchorGenerator(anchor_base, anchor_scales,
                                      anchor_ratios))

        self.anchor_generators = nn.LayerList(self.anchor_generators)
        self.fam_cls_convs = nn.Sequential()
        self.fam_reg_convs = nn.Sequential()

        for i in range(self.stacked_convs):
            chan_in = self.feat_in if i == 0 else self.feat_out

            self.fam_cls_convs.add_sublayer(
                'fam_cls_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=chan_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.fam_cls_convs.add_sublayer('fam_cls_conv_{}_act'.format(i),
                                            nn.ReLU())

            self.fam_reg_convs.add_sublayer(
                'fam_reg_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=chan_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.fam_reg_convs.add_sublayer('fam_reg_conv_{}_act'.format(i),
                                            nn.ReLU())

        self.fam_reg = nn.Conv2D(
            self.feat_out,
            5,
            1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(0)))
        prior_prob = 0.01
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.fam_cls = nn.Conv2D(
            self.feat_out,
            self.cls_out_channels,
            1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(bias_init)))

        if self.align_conv_type == "AlignConv":
            self.align_conv = AlignConv(self.feat_out, self.feat_out,
                                        self.align_conv_size)
        elif self.align_conv_type == "Conv":
            self.align_conv = nn.Conv2D(
                self.feat_out,
                self.feat_out,
                self.align_conv_size,
                padding=(self.align_conv_size - 1) // 2,
                bias_attr=ParamAttr(initializer=Constant(0)))

        elif self.align_conv_type == "DCN":
            self.align_conv_offset = nn.Conv2D(
                self.feat_out,
                2 * self.align_conv_size**2,
                1,
                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                bias_attr=ParamAttr(initializer=Constant(0)))

            self.align_conv = paddle.vision.ops.DeformConv2D(
                self.feat_out,
                self.feat_out,
                self.align_conv_size,
                padding=(self.align_conv_size - 1) // 2,
                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                bias_attr=False)

        self.or_conv = nn.Conv2D(
            self.feat_out,
            self.feat_out,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(0)))

        # ODM
        self.odm_cls_convs = nn.Sequential()
        self.odm_reg_convs = nn.Sequential()

        for i in range(self.stacked_convs):
            ch_in = self.feat_out
            # ch_in = int(self.feat_out / 8) if i == 0 else self.feat_out

            self.odm_cls_convs.add_sublayer(
                'odm_cls_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=ch_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.odm_cls_convs.add_sublayer('odm_cls_conv_{}_act'.format(i),
                                            nn.ReLU())

            self.odm_reg_convs.add_sublayer(
                'odm_reg_conv_{}'.format(i),
                nn.Conv2D(
                    in_channels=self.feat_out,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                    bias_attr=ParamAttr(initializer=Constant(0))))

            self.odm_reg_convs.add_sublayer('odm_reg_conv_{}_act'.format(i),
                                            nn.ReLU())

        self.odm_cls = nn.Conv2D(
            self.feat_out,
            self.cls_out_channels,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(bias_init)))
        self.odm_reg = nn.Conv2D(
            self.feat_out,
            5,
            3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(0)))

        self.featmap_sizes = []
        self.base_anchors_list = []
        self.refine_anchor_list = []

    def forward(self, feats):
        fam_reg_branch_list = []
        fam_cls_branch_list = []

        odm_reg_branch_list = []
        odm_cls_branch_list = []

        self.featmap_sizes_list = []
        self.base_anchors_list = []
        self.refine_anchor_list = []

        for feat_idx in range(len(feats)):
            feat = feats[feat_idx]
            fam_cls_feat = self.fam_cls_convs(feat)

            fam_cls = self.fam_cls(fam_cls_feat)
            # [N, CLS, H, W] --> [N, H, W, CLS]
            fam_cls = fam_cls.transpose([0, 2, 3, 1])
            fam_cls_reshape = paddle.reshape(
                fam_cls, [fam_cls.shape[0], -1, self.cls_out_channels])
            fam_cls_branch_list.append(fam_cls_reshape)

            fam_reg_feat = self.fam_reg_convs(feat)

            fam_reg = self.fam_reg(fam_reg_feat)
            # [N, 5, H, W] --> [N, H, W, 5]
            fam_reg = fam_reg.transpose([0, 2, 3, 1])
            fam_reg_reshape = paddle.reshape(fam_reg, [fam_reg.shape[0], -1, 5])
            fam_reg_branch_list.append(fam_reg_reshape)

            # prepare anchor
            featmap_size = (paddle.shape(feat)[2], paddle.shape(feat)[3])
            self.featmap_sizes_list.append(featmap_size)
            init_anchors = self.anchor_generators[feat_idx](
                featmap_size, self.anchor_strides[feat_idx])

            init_anchors = paddle.to_tensor(init_anchors, dtype='float32')
            NA = featmap_size[0] * featmap_size[1]
            init_anchors = paddle.reshape(init_anchors, [NA, 4])
            init_anchors = self.rect2rbox(init_anchors)
            self.base_anchors_list.append(init_anchors)

            if self.training:
                refine_anchor = self.bbox_decode(fam_reg.detach(), init_anchors)
            else:
                refine_anchor = self.bbox_decode(fam_reg, init_anchors)

            self.refine_anchor_list.append(refine_anchor)

            if self.align_conv_type == 'AlignConv':
                align_feat = self.align_conv(feat,
                                             refine_anchor.clone(),
                                             featmap_size,
                                             self.anchor_strides[feat_idx])
            elif self.align_conv_type == 'DCN':
                align_offset = self.align_conv_offset(feat)
                align_feat = self.align_conv(feat, align_offset)
            elif self.align_conv_type == 'Conv':
                align_feat = self.align_conv(feat)

            or_feat = self.or_conv(align_feat)
            odm_reg_feat = or_feat
            odm_cls_feat = or_feat

            odm_reg_feat = self.odm_reg_convs(odm_reg_feat)
            odm_cls_feat = self.odm_cls_convs(odm_cls_feat)

            odm_cls_score = self.odm_cls(odm_cls_feat)
            # [N, CLS, H, W] --> [N, H, W, CLS]
            odm_cls_score = odm_cls_score.transpose([0, 2, 3, 1])
            odm_cls_score_shape = odm_cls_score.shape
            odm_cls_score_reshape = paddle.reshape(odm_cls_score, [
                odm_cls_score_shape[0], odm_cls_score_shape[1] *
                odm_cls_score_shape[2], self.cls_out_channels
            ])

            odm_cls_branch_list.append(odm_cls_score_reshape)

            odm_bbox_pred = self.odm_reg(odm_reg_feat)
            # [N, 5, H, W] --> [N, H, W, 5]
            odm_bbox_pred = odm_bbox_pred.transpose([0, 2, 3, 1])
            odm_bbox_pred_reshape = paddle.reshape(odm_bbox_pred, [-1, 5])
            odm_bbox_pred_reshape = paddle.unsqueeze(
                odm_bbox_pred_reshape, axis=0)
            odm_reg_branch_list.append(odm_bbox_pred_reshape)

        self.s2anet_head_out = (fam_cls_branch_list, fam_reg_branch_list,
                                odm_cls_branch_list, odm_reg_branch_list)
        return self.s2anet_head_out

    def get_prediction(self, nms_pre=2000):
        refine_anchors = self.refine_anchor_list
        fam_cls_branch_list = self.s2anet_head_out[0]
        fam_reg_branch_list = self.s2anet_head_out[1]
        odm_cls_branch_list = self.s2anet_head_out[2]
        odm_reg_branch_list = self.s2anet_head_out[3]
        pred_scores, pred_bboxes = self.get_bboxes(
            odm_cls_branch_list, odm_reg_branch_list, refine_anchors, nms_pre,
            self.cls_out_channels, self.use_sigmoid_cls)
        return pred_scores, pred_bboxes

    def smooth_l1_loss(self, pred, label, delta=1.0 / 9.0):
        """
        Args:
            pred: pred score
            label: label
            delta: delta
        Returns: loss
        """
        assert pred.shape == label.shape and label.numel() > 0
        assert delta > 0
        diff = paddle.abs(pred - label)
        loss = paddle.where(diff < delta, 0.5 * diff * diff / delta,
                            diff - 0.5 * delta)
        return loss

    def get_fam_loss(self, fam_target, s2anet_head_out, reg_loss_type='gwd'):
        (labels, label_weights, bbox_targets, bbox_weights, bbox_gt_bboxes,
         pos_inds, neg_inds) = fam_target
        fam_cls_branch_list, fam_reg_branch_list, odm_cls_branch_list, odm_reg_branch_list = s2anet_head_out

        fam_cls_losses = []
        fam_bbox_losses = []
        st_idx = 0
        num_total_samples = len(pos_inds) + len(
            neg_inds) if self.sampling else len(pos_inds)
        num_total_samples = max(1, num_total_samples)

        for idx, feat_size in enumerate(self.featmap_sizes_list):
            feat_anchor_num = feat_size[0] * feat_size[1]

            # step1:  get data
            feat_labels = labels[st_idx:st_idx + feat_anchor_num]
            feat_label_weights = label_weights[st_idx:st_idx + feat_anchor_num]

            feat_bbox_targets = bbox_targets[st_idx:st_idx + feat_anchor_num, :]
            feat_bbox_weights = bbox_weights[st_idx:st_idx + feat_anchor_num, :]

            # step2: calc cls loss
            feat_labels = feat_labels.reshape(-1)
            feat_label_weights = feat_label_weights.reshape(-1)

            fam_cls_score = fam_cls_branch_list[idx]
            fam_cls_score = paddle.squeeze(fam_cls_score, axis=0)
            fam_cls_score1 = fam_cls_score

            feat_labels = paddle.to_tensor(feat_labels)
            feat_labels_one_hot = paddle.nn.functional.one_hot(
                feat_labels, self.cls_out_channels + 1)
            feat_labels_one_hot = feat_labels_one_hot[:, 1:]
            feat_labels_one_hot.stop_gradient = True

            num_total_samples = paddle.to_tensor(
                num_total_samples, dtype='float32', stop_gradient=True)

            fam_cls = F.sigmoid_focal_loss(
                fam_cls_score1,
                feat_labels_one_hot,
                normalizer=num_total_samples,
                reduction='none')

            feat_label_weights = feat_label_weights.reshape(
                feat_label_weights.shape[0], 1)
            feat_label_weights = np.repeat(
                feat_label_weights, self.cls_out_channels, axis=1)
            feat_label_weights = paddle.to_tensor(
                feat_label_weights, stop_gradient=True)

            fam_cls = fam_cls * feat_label_weights
            fam_cls_total = paddle.sum(fam_cls)
            fam_cls_losses.append(fam_cls_total)

            # step3: regression loss
            feat_bbox_targets = paddle.to_tensor(
                feat_bbox_targets, dtype='float32', stop_gradient=True)
            feat_bbox_targets = paddle.reshape(feat_bbox_targets, [-1, 5])

            fam_bbox_pred = fam_reg_branch_list[idx]
            fam_bbox_pred = paddle.squeeze(fam_bbox_pred, axis=0)
            fam_bbox_pred = paddle.reshape(fam_bbox_pred, [-1, 5])
            fam_bbox = self.smooth_l1_loss(fam_bbox_pred, feat_bbox_targets)
            loss_weight = paddle.to_tensor(
                self.reg_loss_weight, dtype='float32', stop_gradient=True)
            fam_bbox = paddle.multiply(fam_bbox, loss_weight)
            feat_bbox_weights = paddle.to_tensor(
                feat_bbox_weights, stop_gradient=True)

            if reg_loss_type == 'l1':
                fam_bbox = fam_bbox * feat_bbox_weights
                fam_bbox_total = paddle.sum(fam_bbox) / num_total_samples
            elif reg_loss_type == 'iou' or reg_loss_type == 'gwd':
                fam_bbox = paddle.sum(fam_bbox, axis=-1)
                feat_bbox_weights = paddle.sum(feat_bbox_weights, axis=-1)
                try:
                    from rbox_iou_ops import rbox_iou
                except Exception as e:
                    print("import custom_ops error, try install rbox_iou_ops " \
                          "following ppdet/ext_op/README.md", e)
                    sys.stdout.flush()
                    sys.exit(-1)
                # calc iou
                fam_bbox_decode = self.delta2rbox(self.base_anchors_list[idx],
                                                  fam_bbox_pred)
                bbox_gt_bboxes = paddle.to_tensor(
                    bbox_gt_bboxes,
                    dtype=fam_bbox_decode.dtype,
                    place=fam_bbox_decode.place)
                bbox_gt_bboxes.stop_gradient = True
                iou = rbox_iou(fam_bbox_decode, bbox_gt_bboxes)
                iou = paddle.diag(iou)

                if reg_loss_type == 'gwd':
                    bbox_gt_bboxes_level = bbox_gt_bboxes[st_idx:st_idx +
                                                          feat_anchor_num, :]
                    fam_bbox_total = self.gwd_loss(fam_bbox_decode,
                                                   bbox_gt_bboxes_level)
                    fam_bbox_total = fam_bbox_total * feat_bbox_weights
                    fam_bbox_total = paddle.sum(
                        fam_bbox_total) / num_total_samples

            fam_bbox_losses.append(fam_bbox_total)
            st_idx += feat_anchor_num

        fam_cls_loss = paddle.add_n(fam_cls_losses)
        fam_cls_loss_weight = paddle.to_tensor(
            self.cls_loss_weight[0], dtype='float32', stop_gradient=True)
        fam_cls_loss = fam_cls_loss * fam_cls_loss_weight
        fam_reg_loss = paddle.add_n(fam_bbox_losses)
        return fam_cls_loss, fam_reg_loss

    def get_odm_loss(self, odm_target, s2anet_head_out, reg_loss_type='gwd'):
        (labels, label_weights, bbox_targets, bbox_weights, bbox_gt_bboxes,
         pos_inds, neg_inds) = odm_target
        fam_cls_branch_list, fam_reg_branch_list, odm_cls_branch_list, odm_reg_branch_list = s2anet_head_out

        odm_cls_losses = []
        odm_bbox_losses = []
        st_idx = 0
        num_total_samples = len(pos_inds) + len(
            neg_inds) if self.sampling else len(pos_inds)
        num_total_samples = max(1, num_total_samples)

        for idx, feat_size in enumerate(self.featmap_sizes_list):
            feat_anchor_num = feat_size[0] * feat_size[1]

            # step1:  get data
            feat_labels = labels[st_idx:st_idx + feat_anchor_num]
            feat_label_weights = label_weights[st_idx:st_idx + feat_anchor_num]

            feat_bbox_targets = bbox_targets[st_idx:st_idx + feat_anchor_num, :]
            feat_bbox_weights = bbox_weights[st_idx:st_idx + feat_anchor_num, :]

            # step2: calc cls loss
            feat_labels = feat_labels.reshape(-1)
            feat_label_weights = feat_label_weights.reshape(-1)

            odm_cls_score = odm_cls_branch_list[idx]
            odm_cls_score = paddle.squeeze(odm_cls_score, axis=0)
            odm_cls_score1 = odm_cls_score

            feat_labels = paddle.to_tensor(feat_labels)
            feat_labels_one_hot = paddle.nn.functional.one_hot(
                feat_labels, self.cls_out_channels + 1)
            feat_labels_one_hot = feat_labels_one_hot[:, 1:]
            feat_labels_one_hot.stop_gradient = True

            num_total_samples = paddle.to_tensor(
                num_total_samples, dtype='float32', stop_gradient=True)
            odm_cls = F.sigmoid_focal_loss(
                odm_cls_score1,
                feat_labels_one_hot,
                normalizer=num_total_samples,
                reduction='none')

            feat_label_weights = feat_label_weights.reshape(
                feat_label_weights.shape[0], 1)
            feat_label_weights = np.repeat(
                feat_label_weights, self.cls_out_channels, axis=1)
            feat_label_weights = paddle.to_tensor(feat_label_weights)
            feat_label_weights.stop_gradient = True

            odm_cls = odm_cls * feat_label_weights
            odm_cls_total = paddle.sum(odm_cls)
            odm_cls_losses.append(odm_cls_total)

            # # step3: regression loss
            feat_bbox_targets = paddle.to_tensor(
                feat_bbox_targets, dtype='float32')
            feat_bbox_targets = paddle.reshape(feat_bbox_targets, [-1, 5])
            feat_bbox_targets.stop_gradient = True

            odm_bbox_pred = odm_reg_branch_list[idx]
            odm_bbox_pred = paddle.squeeze(odm_bbox_pred, axis=0)
            odm_bbox_pred = paddle.reshape(odm_bbox_pred, [-1, 5])
            odm_bbox = self.smooth_l1_loss(odm_bbox_pred, feat_bbox_targets)

            loss_weight = paddle.to_tensor(
                self.reg_loss_weight, dtype='float32', stop_gradient=True)
            odm_bbox = paddle.multiply(odm_bbox, loss_weight)
            feat_bbox_weights = paddle.to_tensor(
                feat_bbox_weights, stop_gradient=True)

            if reg_loss_type == 'l1':
                odm_bbox = odm_bbox * feat_bbox_weights
                odm_bbox_total = paddle.sum(odm_bbox) / num_total_samples
            elif reg_loss_type == 'iou' or reg_loss_type == 'gwd':
                odm_bbox = paddle.sum(odm_bbox, axis=-1)
                feat_bbox_weights = paddle.sum(feat_bbox_weights, axis=-1)
                try:
                    from rbox_iou_ops import rbox_iou
                except Exception as e:
                    print("import custom_ops error, try install rbox_iou_ops " \
                          "following ppdet/ext_op/README.md", e)
                    sys.stdout.flush()
                    sys.exit(-1)
                # calc iou
                odm_bbox_decode = self.delta2rbox(self.refine_anchor_list[idx],
                                                  odm_bbox_pred)
                bbox_gt_bboxes = paddle.to_tensor(
                    bbox_gt_bboxes,
                    dtype=odm_bbox_decode.dtype,
                    place=odm_bbox_decode.place)
                bbox_gt_bboxes.stop_gradient = True
                iou = rbox_iou(odm_bbox_decode, bbox_gt_bboxes)
                iou = paddle.diag(iou)

                if reg_loss_type == 'gwd':
                    bbox_gt_bboxes_level = bbox_gt_bboxes[st_idx:st_idx +
                                                          feat_anchor_num, :]
                    odm_bbox_total = self.gwd_loss(odm_bbox_decode,
                                                   bbox_gt_bboxes_level)
                    odm_bbox_total = odm_bbox_total * feat_bbox_weights
                    odm_bbox_total = paddle.sum(
                        odm_bbox_total) / num_total_samples

            odm_bbox_losses.append(odm_bbox_total)
            st_idx += feat_anchor_num

        odm_cls_loss = paddle.add_n(odm_cls_losses)
        odm_cls_loss_weight = paddle.to_tensor(
            self.cls_loss_weight[1], dtype='float32', stop_gradient=True)
        odm_cls_loss = odm_cls_loss * odm_cls_loss_weight
        odm_reg_loss = paddle.add_n(odm_bbox_losses)
        return odm_cls_loss, odm_reg_loss

    def get_loss(self, inputs):
        # inputs: im_id image im_shape scale_factor gt_bbox gt_class is_crowd

        # compute loss
        fam_cls_loss_lst = []
        fam_reg_loss_lst = []
        odm_cls_loss_lst = []
        odm_reg_loss_lst = []

        im_shape = inputs['im_shape']
        for im_id in range(im_shape.shape[0]):
            np_im_shape = inputs['im_shape'][im_id].numpy()
            np_scale_factor = inputs['scale_factor'][im_id].numpy()
            # data_format: (xc, yc, w, h, theta)
            gt_bboxes = inputs['gt_rbox'][im_id].numpy()
            gt_labels = inputs['gt_class'][im_id].numpy()
            is_crowd = inputs['is_crowd'][im_id].numpy()
            gt_labels = gt_labels + 1

            # featmap_sizes
            anchors_list_all = np.concatenate(self.base_anchors_list)

            # get im_feat
            fam_cls_feats_list = [e[im_id] for e in self.s2anet_head_out[0]]
            fam_reg_feats_list = [e[im_id] for e in self.s2anet_head_out[1]]
            odm_cls_feats_list = [e[im_id] for e in self.s2anet_head_out[2]]
            odm_reg_feats_list = [e[im_id] for e in self.s2anet_head_out[3]]
            im_s2anet_head_out = (fam_cls_feats_list, fam_reg_feats_list,
                                  odm_cls_feats_list, odm_reg_feats_list)

            # FAM
            im_fam_target = self.anchor_assign(anchors_list_all, gt_bboxes,
                                               gt_labels, is_crowd)
            if im_fam_target is not None:
                im_fam_cls_loss, im_fam_reg_loss = self.get_fam_loss(
                    im_fam_target, im_s2anet_head_out, self.reg_loss_type)
                fam_cls_loss_lst.append(im_fam_cls_loss)
                fam_reg_loss_lst.append(im_fam_reg_loss)

            # ODM
            np_refine_anchors_list = paddle.concat(
                self.refine_anchor_list).numpy()
            np_refine_anchors_list = np.concatenate(np_refine_anchors_list)
            np_refine_anchors_list = np_refine_anchors_list.reshape(-1, 5)
            im_odm_target = self.anchor_assign(np_refine_anchors_list,
                                               gt_bboxes, gt_labels, is_crowd)

            if im_odm_target is not None:
                im_odm_cls_loss, im_odm_reg_loss = self.get_odm_loss(
                    im_odm_target, im_s2anet_head_out, self.reg_loss_type)
                odm_cls_loss_lst.append(im_odm_cls_loss)
                odm_reg_loss_lst.append(im_odm_reg_loss)
        fam_cls_loss = paddle.add_n(fam_cls_loss_lst)
        fam_reg_loss = paddle.add_n(fam_reg_loss_lst)
        odm_cls_loss = paddle.add_n(odm_cls_loss_lst)
        odm_reg_loss = paddle.add_n(odm_reg_loss_lst)
        return {
            'fam_cls_loss': fam_cls_loss,
            'fam_reg_loss': fam_reg_loss,
            'odm_cls_loss': odm_cls_loss,
            'odm_reg_loss': odm_reg_loss
        }

    def get_bboxes(self, cls_score_list, bbox_pred_list, mlvl_anchors, nms_pre,
                   cls_out_channels, use_sigmoid_cls):
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)

        mlvl_bboxes = []
        mlvl_scores = []

        idx = 0
        for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list,
                                                 mlvl_anchors):
            cls_score = paddle.reshape(cls_score, [-1, cls_out_channels])
            if use_sigmoid_cls:
                scores = F.sigmoid(cls_score)
            else:
                scores = F.softmax(cls_score, axis=-1)

            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            bbox_pred = paddle.transpose(bbox_pred, [1, 2, 0])
            bbox_pred = paddle.reshape(bbox_pred, [-1, 5])
            anchors = paddle.reshape(anchors, [-1, 5])

            if scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if use_sigmoid_cls:
                    max_scores = paddle.max(scores, axis=1)
                else:
                    max_scores = paddle.max(scores[:, 1:], axis=1)

                topk_val, topk_inds = paddle.topk(max_scores, nms_pre)
                anchors = paddle.gather(anchors, topk_inds)
                bbox_pred = paddle.gather(bbox_pred, topk_inds)
                scores = paddle.gather(scores, topk_inds)

            bbox_delta = paddle.reshape(bbox_pred, [-1, 5])
            bboxes = self.delta2rbox(anchors, bbox_delta)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

            idx += 1

        mlvl_bboxes = paddle.concat(mlvl_bboxes, axis=0)
        mlvl_scores = paddle.concat(mlvl_scores)

        return mlvl_scores, mlvl_bboxes

    def rect2rbox(self, bboxes):
        """
        :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
        :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
        """
        bboxes = paddle.reshape(bboxes, [-1, 4])
        num_boxes = paddle.shape(bboxes)[0]
        x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
        y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
        edges1 = paddle.abs(bboxes[:, 2] - bboxes[:, 0])
        edges2 = paddle.abs(bboxes[:, 3] - bboxes[:, 1])

        rbox_w = paddle.maximum(edges1, edges2)
        rbox_h = paddle.minimum(edges1, edges2)

        # set angle
        inds = edges1 < edges2
        inds = paddle.cast(inds, 'int32')
        rboxes_angle = inds * np.pi / 2.0

        rboxes = paddle.stack(
            (x_ctr, y_ctr, rbox_w, rbox_h, rboxes_angle), axis=1)
        return rboxes

    # deltas to rbox
    def delta2rbox(self, rrois, deltas, wh_ratio_clip=1e-6):
        """
        :param rrois: (cx, cy, w, h, theta)
        :param deltas: (dx, dy, dw, dh, dtheta)
        :param means: means of anchor
        :param stds: stds of anchor
        :param wh_ratio_clip: clip threshold of wh_ratio
        :return:
        """
        deltas = paddle.reshape(deltas, [-1, 5])
        rrois = paddle.reshape(rrois, [-1, 5])
        # fix dy2st bug denorm_deltas = deltas * self.stds + self.means
        denorm_deltas = paddle.add(
            paddle.multiply(deltas, self.stds), self.means)

        dx = denorm_deltas[:, 0]
        dy = denorm_deltas[:, 1]
        dw = denorm_deltas[:, 2]
        dh = denorm_deltas[:, 3]
        dangle = denorm_deltas[:, 4]
        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = paddle.clip(dw, min=-max_ratio, max=max_ratio)
        dh = paddle.clip(dh, min=-max_ratio, max=max_ratio)

        rroi_x = rrois[:, 0]
        rroi_y = rrois[:, 1]
        rroi_w = rrois[:, 2]
        rroi_h = rrois[:, 3]
        rroi_angle = rrois[:, 4]

        gx = dx * rroi_w * paddle.cos(rroi_angle) - dy * rroi_h * paddle.sin(
            rroi_angle) + rroi_x
        gy = dx * rroi_w * paddle.sin(rroi_angle) + dy * rroi_h * paddle.cos(
            rroi_angle) + rroi_y
        gw = rroi_w * dw.exp()
        gh = rroi_h * dh.exp()
        ga = np.pi * dangle + rroi_angle
        ga = (ga + np.pi / 4) % np.pi - np.pi / 4
        ga = paddle.to_tensor(ga)
        gw = paddle.to_tensor(gw, dtype='float32')
        gh = paddle.to_tensor(gh, dtype='float32')
        bboxes = paddle.stack([gx, gy, gw, gh, ga], axis=-1)
        return bboxes

    def bbox_decode(self, bbox_preds, anchors):
        """decode bbox from deltas
        Args:
            bbox_preds: [N,H,W,5]
            anchors: [H*W,5]
        return:
            bboxes: [N,H,W,5]
        """
        num_imgs, H, W, _ = bbox_preds.shape
        bbox_delta = paddle.reshape(bbox_preds, [-1, 5])
        bboxes = self.delta2rbox(anchors, bbox_delta)
        return bboxes

    def trace(self, A):
        tr = paddle.diagonal(A, axis1=-2, axis2=-1)
        tr = paddle.sum(tr, axis=-1)
        return tr

    def sqrt_newton_schulz_autograd(self, A, numIters):
        A_shape = A.shape
        batchSize = A_shape[0]
        dim = A_shape[1]

        normA = A * A
        normA = paddle.sum(normA, axis=1)
        normA = paddle.sum(normA, axis=1)
        normA = paddle.sqrt(normA)
        normA1 = normA.reshape([batchSize, 1, 1])
        Y = paddle.divide(A, paddle.expand_as(normA1, A))
        I = paddle.eye(dim, dim).reshape([1, dim, dim])
        l0 = []
        for i in range(batchSize):
            l0.append(I)
        I = paddle.concat(l0, axis=0)
        I.stop_gradient = False
        Z = paddle.eye(dim, dim).reshape([1, dim, dim])
        l1 = []
        for i in range(batchSize):
            l1.append(Z)
        Z = paddle.concat(l1, axis=0)
        Z.stop_gradient = False

        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * paddle.sqrt(normA1).reshape([batchSize, 1, 1])
        sA = paddle.expand_as(sA, A)
        return sA

    def wasserstein_distance_sigma(sigma1, sigma2):
        wasserstein_distance_item2 = paddle.matmul(
            sigma1, sigma1) + paddle.matmul(
                sigma2, sigma2) - 2 * self.sqrt_newton_schulz_autograd(
                    paddle.matmul(
                        paddle.matmul(sigma1, paddle.matmul(sigma2, sigma2)),
                        sigma1), 10)
        wasserstein_distance_item2 = self.trace(wasserstein_distance_item2)

        return wasserstein_distance_item2

    def xywhr2xyrs(self, xywhr):
        xywhr = paddle.reshape(xywhr, [-1, 5])
        xy = xywhr[:, :2]
        wh = paddle.clip(xywhr[:, 2:4], min=1e-7, max=1e7)
        r = xywhr[:, 4]
        cos_r = paddle.cos(r)
        sin_r = paddle.sin(r)
        R = paddle.stack(
            (cos_r, -sin_r, sin_r, cos_r), axis=-1).reshape([-1, 2, 2])
        S = 0.5 * paddle.nn.functional.diag_embed(wh)
        return xy, R, S

    def gwd_loss(self,
                 pred,
                 target,
                 fun='log',
                 tau=1.0,
                 alpha=1.0,
                 normalize=False):

        xy_p, R_p, S_p = self.xywhr2xyrs(pred)
        xy_t, R_t, S_t = self.xywhr2xyrs(target)

        xy_distance = (xy_p - xy_t).square().sum(axis=-1)

        Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.transpose([0, 2, 1]))
        Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.transpose([0, 2, 1]))

        whr_distance = paddle.diagonal(
            S_p, axis1=-2, axis2=-1).square().sum(axis=-1)

        whr_distance = whr_distance + paddle.diagonal(
            S_t, axis1=-2, axis2=-1).square().sum(axis=-1)
        _t = Sigma_p.matmul(Sigma_t)

        _t_tr = paddle.diagonal(_t, axis1=-2, axis2=-1).sum(axis=-1)
        _t_det_sqrt = paddle.diagonal(S_p, axis1=-2, axis2=-1).prod(axis=-1)
        _t_det_sqrt = _t_det_sqrt * paddle.diagonal(
            S_t, axis1=-2, axis2=-1).prod(axis=-1)
        whr_distance = whr_distance + (-2) * (
            (_t_tr + 2 * _t_det_sqrt).clip(0).sqrt())

        distance = (xy_distance + alpha * alpha * whr_distance).clip(0)

        if normalize:
            wh_p = pred[..., 2:4].clip(min=1e-7, max=1e7)
            wh_t = target[..., 2:4].clip(min=1e-7, max=1e7)
            scale = ((wh_p.log() + wh_t.log()).sum(dim=-1) / 4).exp()
            distance = distance / scale

        if fun == 'log':
            distance = paddle.log1p(distance)

        if tau >= 1.0:
            return 1 - 1 / (tau + distance)

        return distance
