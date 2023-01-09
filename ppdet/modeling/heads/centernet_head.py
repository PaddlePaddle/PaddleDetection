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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, Uniform
from ppdet.core.workspace import register
from ppdet.modeling.losses import CTFocalLoss, GIoULoss


class ConvLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(ConvLayer, self).__init__()
        bias_attr = False
        fan_in = ch_in * kernel_size**2
        bound = 1 / math.sqrt(fan_in)
        param_attr = paddle.ParamAttr(initializer=Uniform(-bound, bound))
        if bias:
            bias_attr = paddle.ParamAttr(initializer=Constant(0.))
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=param_attr,
            bias_attr=bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)
        return out


@register
class CenterNetHead(nn.Layer):
    """
    Args:
        in_channels (int): the channel number of input to CenterNetHead.
        num_classes (int): the number of classes, 80 (COCO dataset) by default.
        head_planes (int): the channel number in all head, 256 by default.
        prior_bias (float): prior bias in heatmap head, -2.19 by default, -4.6 in CenterTrack
        regress_ltrb (bool): whether to regress left/top/right/bottom or
            width/height for a box, True by default.
        size_loss (str): the type of size regression loss, 'L1' by default, can be 'giou'.
        loss_weight (dict): the weight of each loss.
        add_iou (bool): whether to add iou branch, False by default.
    """

    __shared__ = ['num_classes']

    def __init__(self,
                 in_channels,
                 num_classes=80,
                 head_planes=256,
                 prior_bias=-2.19,
                 regress_ltrb=True,
                 size_loss='L1',
                 loss_weight={
                     'heatmap': 1.0,
                     'size': 0.1,
                     'offset': 1.0,
                     'iou': 0.0,
                 },
                 add_iou=False):
        super(CenterNetHead, self).__init__()
        self.regress_ltrb = regress_ltrb
        self.loss_weight = loss_weight
        self.add_iou = add_iou

        # heatmap head
        self.heatmap = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        with paddle.no_grad():
            self.heatmap[2].conv.bias[:] = prior_bias

        # size(ltrb or wh) head
        self.size = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                4 if regress_ltrb else 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        self.size_loss = size_loss

        # offset head
        self.offset = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                head_planes, 2, kernel_size=1, stride=1, padding=0, bias=True))

        # iou head (optinal)
        if self.add_iou and 'iou' in self.loss_weight:
            self.iou = nn.Sequential(
                ConvLayer(
                    in_channels,
                    head_planes,
                    kernel_size=3,
                    padding=1,
                    bias=True),
                nn.ReLU(),
                ConvLayer(
                    head_planes,
                    4 if regress_ltrb else 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self, feat, inputs):
        heatmap = F.sigmoid(self.heatmap(feat))
        size = self.size(feat)
        offset = self.offset(feat)
        head_outs = {'heatmap': heatmap, 'size': size, 'offset': offset}
        if self.add_iou and 'iou' in self.loss_weight:
            iou = self.iou(feat)
            head_outs.update({'iou': iou})

        if self.training:
            losses = self.get_loss(inputs, self.loss_weight, head_outs)
            return losses
        else:
            return head_outs

    def get_loss(self, inputs, weights, head_outs):
        # 1.heatmap(hm) head loss: CTFocalLoss
        heatmap = head_outs['heatmap']
        heatmap_target = inputs['heatmap']
        heatmap = paddle.clip(heatmap, 1e-4, 1 - 1e-4)
        ctfocal_loss = CTFocalLoss()
        heatmap_loss = ctfocal_loss(heatmap, heatmap_target)

        # 2.size(wh) head loss: L1 loss or GIoU loss
        size = head_outs['size']
        index = inputs['index']
        mask = inputs['index_mask']
        size = paddle.transpose(size, perm=[0, 2, 3, 1])
        size_n, _, _, size_c = size.shape
        size = paddle.reshape(size, shape=[size_n, -1, size_c])
        index = paddle.unsqueeze(index, 2)
        batch_inds = list()
        for i in range(size_n):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)
        pos_size = paddle.gather_nd(size, index=index)
        mask = paddle.unsqueeze(mask, axis=2)
        size_mask = paddle.expand_as(mask, pos_size)
        size_mask = paddle.cast(size_mask, dtype=pos_size.dtype)
        pos_num = size_mask.sum()
        size_mask.stop_gradient = True
        if self.size_loss == 'L1':
            if self.regress_ltrb:
                size_target = inputs['size']
                # shape: [bs, max_per_img, 4]
            else:
                if inputs['size'].shape[-1] == 2:
                    # inputs['size'] is wh, and regress as wh
                    # shape: [bs, max_per_img, 2]
                    size_target = inputs['size']
                else:
                    # inputs['size'] is ltrb, but regress as wh
                    # shape: [bs, max_per_img, 4]
                    size_target = inputs['size'][:, :, 0:2] + inputs[
                        'size'][:, :, 2:]

            size_target.stop_gradient = True
            size_loss = F.l1_loss(
                pos_size * size_mask, size_target * size_mask, reduction='sum')
            size_loss = size_loss / (pos_num + 1e-4)
        elif self.size_loss == 'giou':
            size_target = inputs['bbox_xys']
            size_target.stop_gradient = True
            centers_x = (size_target[:, :, 0:1] + size_target[:, :, 2:3]) / 2.0
            centers_y = (size_target[:, :, 1:2] + size_target[:, :, 3:4]) / 2.0
            x1 = centers_x - pos_size[:, :, 0:1]
            y1 = centers_y - pos_size[:, :, 1:2]
            x2 = centers_x + pos_size[:, :, 2:3]
            y2 = centers_y + pos_size[:, :, 3:4]
            pred_boxes = paddle.concat([x1, y1, x2, y2], axis=-1)
            giou_loss = GIoULoss(reduction='sum')
            size_loss = giou_loss(
                pred_boxes * size_mask,
                size_target * size_mask,
                iou_weight=size_mask,
                loc_reweight=None)
            size_loss = size_loss / (pos_num + 1e-4)

        # 3.offset(reg) head loss: L1 loss
        offset = head_outs['offset']
        offset_target = inputs['offset']
        offset = paddle.transpose(offset, perm=[0, 2, 3, 1])
        offset_n, _, _, offset_c = offset.shape
        offset = paddle.reshape(offset, shape=[offset_n, -1, offset_c])
        pos_offset = paddle.gather_nd(offset, index=index)
        offset_mask = paddle.expand_as(mask, pos_offset)
        offset_mask = paddle.cast(offset_mask, dtype=pos_offset.dtype)
        pos_num = offset_mask.sum()
        offset_mask.stop_gradient = True
        offset_target.stop_gradient = True
        offset_loss = F.l1_loss(
            pos_offset * offset_mask,
            offset_target * offset_mask,
            reduction='sum')
        offset_loss = offset_loss / (pos_num + 1e-4)

        # 4.iou head loss: GIoU loss (optinal)
        if self.add_iou and 'iou' in self.loss_weight:
            iou = head_outs['iou']
            iou = paddle.transpose(iou, perm=[0, 2, 3, 1])
            iou_n, _, _, iou_c = iou.shape
            iou = paddle.reshape(iou, shape=[iou_n, -1, iou_c])
            pos_iou = paddle.gather_nd(iou, index=index)
            iou_mask = paddle.expand_as(mask, pos_iou)
            iou_mask = paddle.cast(iou_mask, dtype=pos_iou.dtype)
            pos_num = iou_mask.sum()
            iou_mask.stop_gradient = True
            gt_bbox_xys = inputs['bbox_xys']
            gt_bbox_xys.stop_gradient = True
            centers_x = (gt_bbox_xys[:, :, 0:1] + gt_bbox_xys[:, :, 2:3]) / 2.0
            centers_y = (gt_bbox_xys[:, :, 1:2] + gt_bbox_xys[:, :, 3:4]) / 2.0
            x1 = centers_x - pos_size[:, :, 0:1]
            y1 = centers_y - pos_size[:, :, 1:2]
            x2 = centers_x + pos_size[:, :, 2:3]
            y2 = centers_y + pos_size[:, :, 3:4]
            pred_boxes = paddle.concat([x1, y1, x2, y2], axis=-1)
            giou_loss = GIoULoss(reduction='sum')
            iou_loss = giou_loss(
                pred_boxes * iou_mask,
                gt_bbox_xys * iou_mask,
                iou_weight=iou_mask,
                loc_reweight=None)
            iou_loss = iou_loss / (pos_num + 1e-4)

        losses = {
            'heatmap_loss': heatmap_loss,
            'size_loss': size_loss,
            'offset_loss': offset_loss,
        }
        det_loss = weights['heatmap'] * heatmap_loss + weights[
            'size'] * size_loss + weights['offset'] * offset_loss

        if self.add_iou and 'iou' in self.loss_weight:
            losses.update({'iou_loss': iou_loss})
            det_loss += weights['iou'] * iou_loss
        losses.update({'det_loss': det_loss})
        return losses
