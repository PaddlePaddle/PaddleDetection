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
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling.ops import DeformConv, SimpleNMS, TopK
from ppdet.modeling.losses import GiouLoss

__all__ = ['TTFHead']


@register
class TTFHead(object):
    """
    TTFHead
    """

    __inject__ = ['wh_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 head_conv=128,
                 num_classes=81,
                 hm_weight=1.,
                 wh_weight=5.,
                 wh_offset_base=16.,
                 planes=(256, 128, 64),
                 shortcut_num=(1, 2, 3),
                 wh_head_conv_num=2,
                 hm_head_conv_num=2,
                 wh_conv=64,
                 score_thresh=0.01,
                 max_per_img=100,
                 base_down_ratio=32,
                 wh_loss='GiouLoss'):
        super(TTFHead, self).__init__()
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.wh_offset_base = wh_offset_base
        self.planes = planes
        self.shortcut_num = shortcut_num
        self.shortcut_len = len(shortcut_num)
        self.wh_head_conv_num = wh_head_conv_num
        self.hm_head_conv_num = hm_head_conv_num
        self.wh_conv = wh_conv
        self.score_thresh = score_thresh
        self.max_per_img = max_per_img
        self.down_ratio = base_down_ratio // 2**len(planes)
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.wh_loss = wh_loss

    def shortcut(self, x, out_c, layer_num, kernel_size=3, padding=1,
                 name=None):
        assert layer_num > 0
        for i in range(layer_num):
            act = 'relu' if i < layer_num - 1 else None
            fan_out = x.shape[2] * x.shape[3] * out_c
            std = math.sqrt(2. / fan_out)
            param_name = name + '.layers.' + str(i * 2)
            x = fluid.layers.conv2d(
                x,
                out_c,
                kernel_size,
                padding=padding,
                act=act,
                param_attr=ParamAttr(
                    initializer=Normal(0, std), name=param_name + '.weight'),
                bias_attr=ParamAttr(
                    learning_rate=2.,
                    regularizer=L2Decay(0.),
                    name=param_name + '.bias'))
        return x

    def upsample(self, x, out_c, name=None):
        conv = DeformConv(x, out_c, 3, bias_attr=True, name=name + '.0')
        norm_name = name + '.1'
        pattr = ParamAttr(name=norm_name + '.weight', initializer=Constant(1.))
        battr = ParamAttr(
            name=norm_name + '.bias',
            initializer=Constant(0.),
            learning_rate=2.,
            regularizer=L2Decay(0.))
        bn = fluid.layers.batch_norm(
            input=conv,
            act='relu',
            param_attr=pattr,
            bias_attr=battr,
            name=norm_name + '.output.1',
            moving_mean_name=norm_name + '.running_mean',
            moving_variance_name=norm_name + '.running_var')
        up = fluid.layers.resize_bilinear(
            bn, scale=2, name=name + '.2.upsample')
        return up

    def _head(self, x, out_c, conv_num=1, head_out_c=None, name=None):
        head_out_c = self.head_conv if not head_out_c else head_out_c
        conv_w_std = 0.01 if '.hm' in name else 0.001
        conv_w_init = Normal(0, conv_w_std)
        for i in range(conv_num):
            conv_name = '{}.{}.conv'.format(name, i)
            x = fluid.layers.conv2d(
                x,
                head_out_c,
                3,
                padding=1,
                param_attr=ParamAttr(
                    initializer=conv_w_init, name=conv_name + '.weight'),
                bias_attr=ParamAttr(
                    learning_rate=2.,
                    regularizer=L2Decay(0.),
                    name=conv_name + '.bias'),
                act='relu')
        bias_init = float(-np.log((1 - 0.01) / 0.01)) if '.hm' in name else 0.
        conv_b_init = Constant(bias_init)
        x = fluid.layers.conv2d(
            x,
            out_c,
            1,
            param_attr=ParamAttr(
                initializer=conv_w_init,
                name='{}.{}.weight'.format(name, conv_num)),
            bias_attr=ParamAttr(
                learning_rate=2.,
                regularizer=L2Decay(0.),
                name='{}.{}.bias'.format(name, conv_num),
                initializer=conv_b_init))
        return x

    def hm_head(self, x, name=None):
        hm = self._head(x, self.num_classes, self.hm_head_conv_num, name=name)
        return hm

    def wh_head(self, x, name=None):
        wh_planes = 4
        wh = self._head(
            x, wh_planes, self.wh_head_conv_num, self.wh_conv, name=name)
        return fluid.layers.relu(wh)

    def get_output(self, input, name=None):
        feat = input[-1]
        for i, out_c in enumerate(self.planes):
            feat = self.upsample(
                feat, out_c, name=name + '.deconv_layers.' + str(i))
            if i < self.shortcut_len:
                shortcut = self.shortcut(
                    input[-i - 2],
                    out_c,
                    self.shortcut_num[i],
                    name=name + '.shortcut_layers.' + str(i))
                feat = fluid.layers.elementwise_add(feat, shortcut)

        hm = self.hm_head(feat, name=name + '.hm')
        wh = self.wh_head(feat, name=name + '.wh') * self.wh_offset_base
        return hm, wh

    def get_bboxes(self, heatmap, wh, scale_factor):
        heatmap = fluid.layers.sigmoid(heatmap)
        heat = SimpleNMS(heatmap)
        scores, inds, clses, ys, xs = TopK(heat, self.max_per_img)
        ys = fluid.layers.cast(ys, 'float32') * self.down_ratio
        xs = fluid.layers.cast(xs, 'float32') * self.down_ratio

        wh_t = fluid.layers.transpose(wh, [0, 2, 3, 1])
        wh = fluid.layers.reshape(wh_t, [-1, wh_t.shape[-1]])
        wh = fluid.layers.gather(wh, inds)

        scores = fluid.layers.unsqueeze(scores, [1])
        clses = fluid.layers.unsqueeze(clses, [1])
        x1 = xs - wh[:, 0:1]
        y1 = ys - wh[:, 1:2]
        x2 = xs + wh[:, 2:3]
        y2 = ys + wh[:, 3:4]
        bboxes = fluid.layers.concat([x1, y1, x2, y2], axis=1)
        bboxes = fluid.layers.elementwise_div(bboxes, scale_factor, axis=-1)
        results = fluid.layers.concat([clses, scores, bboxes], axis=1)
        # hack: append result with cls=-1 and score=1. to avoid all scores
        # are less than score_thresh which may cause error in gather.
        fill_r = fluid.layers.assign(
            np.array(
                [[-1, 1., 0, 0, 0, 0]], dtype='float32'))
        results = fluid.layers.concat([results, fill_r])
        scores = results[:, 1]
        valid_ind = fluid.layers.where(scores > self.score_thresh)
        results = fluid.layers.gather(results, valid_ind)
        return {'bbox': results}

    def ct_focal_loss(self, pred_hm, target_hm, gamma=2.0):
        fg_map = fluid.layers.cast(target_hm == 1, 'float32')
        fg_map.stop_gradient = True
        #num_pos = fluid.layers.reduce_sum(fg_map, [1, 2, 3])
        bg_map = fluid.layers.cast(target_hm < 1, 'float32')
        bg_map.stop_gradient = True

        neg_weights = fluid.layers.pow(1 - target_hm, 4) * bg_map
        pos_loss = 0 - fluid.layers.log(pred_hm) * fluid.layers.pow(
            1 - pred_hm, gamma) * fg_map
        neg_loss = 0 - fluid.layers.log(1 - pred_hm) * fluid.layers.pow(
            pred_hm, gamma) * neg_weights
        pos_loss = fluid.layers.reduce_sum(pos_loss)
        neg_loss = fluid.layers.reduce_sum(neg_loss)

        fg_num = fluid.layers.reduce_sum(fg_map)
        focal_loss = (pos_loss + neg_loss) / (
            fg_num + fluid.layers.cast(fg_num == 0, 'float32'))
        return focal_loss

    def filter_box_by_weight(self, pred, target, weight):
        index = fluid.layers.where(weight > 0)
        weight = fluid.layers.gather_nd(weight, index)
        pred = fluid.layers.gather_nd(pred, index)
        target = fluid.layers.gather_nd(target, index)
        return pred, target, weight

    def get_loss(self, pred_hm, pred_wh, target_hm, box_target, target_weight):
        pred_hm = paddle.tensor.clamp(
            fluid.layers.sigmoid(pred_hm), 1e-4, 1 - 1e-4)
        hm_loss = self.ct_focal_loss(pred_hm, target_hm) * self.hm_weight
        H, W = target_hm.shape[2:]
        mask = fluid.layers.reshape(target_weight, [-1, H, W])
        avg_factor = fluid.layers.reduce_sum(mask) + 1e-4

        base_step = self.down_ratio
        shifts_x = paddle.arange(0, W * base_step, base_step)
        shifts_y = paddle.arange(0, H * base_step, base_step)
        shift_y, shift_x = paddle.tensor.meshgrid([shifts_y, shifts_x])
        self.base_loc = fluid.layers.stack([shift_x, shift_y], axis=0)

        pred_boxes = fluid.layers.concat(
            [
                0 - pred_wh[:, 0:2, :, :] + self.base_loc,
                pred_wh[:, 2:4] + self.base_loc
            ],
            axis=1)
        pred_boxes = fluid.layers.transpose(pred_boxes, [0, 2, 3, 1])
        boxes = fluid.layers.transpose(box_target, [0, 2, 3, 1])

        pred_boxes, boxes, mask = self.filter_box_by_weight(pred_boxes, boxes,
                                                            mask)
        wh_loss = self.wh_loss(
            pred_boxes, boxes, outside_weight=mask, use_transform=False)
        wh_loss = wh_loss / avg_factor
        return {'hm_loss': hm_loss, 'wh_loss': wh_loss}
