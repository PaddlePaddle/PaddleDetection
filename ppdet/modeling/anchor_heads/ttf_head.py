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
from paddle.fluid.initializer import Normal, Constant, Uniform, Xavier
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling.ops import DeformConv, DropBlock
from ppdet.modeling.losses import GiouLoss

__all__ = ['TTFHead']


@register
class TTFHead(object):
    """
    TTFHead
    Args:
        head_conv(int): the default channel number of convolution in head. 
            128 by default.
        num_classes(int): the number of classes, 80 by default.
        hm_weight(float): the weight of heatmap branch. 1. by default.
        wh_weight(float): the weight of wh branch. 5. by default.
        wh_offset_base(flaot): the base offset of width and height. 
            16. by default.
        planes(tuple): the channel number of convolution in each upsample. 
            (256, 128, 64) by default.
        shortcut_num(tuple): the number of convolution layers in each shortcut.
            (1, 2, 3) by default.
        wh_head_conv_num(int): the number of convolution layers in wh head.
            2 by default.
        hm_head_conv_num(int): the number of convolution layers in wh head.
            2 by default.
        wh_conv(int): the channel number of convolution in wh head. 
            64 by default.
        wh_planes(int): the output channel in wh head. 4 by default.
        score_thresh(float): the score threshold to get prediction. 
            0.01 by default.
        max_per_img(int): the maximum detection per image. 100 by default.
        base_down_ratio(int): the base down_ratio, the actual down_ratio is 
            calculated by base_down_ratio and the number of upsample layers.
            16 by default.
        wh_loss(object): `GiouLoss` instance.
        dcn_upsample(bool): whether upsample by dcn. True by default.
        dcn_head(bool): whether use dcn in head. False by default.
        drop_block(bool): whether use dropblock. False by default.
        block_size(int): block_size parameter for drop_block. 3 by default.
        keep_prob(float): keep_prob parameter for drop_block. 0.9 by default.
    """

    __inject__ = ['wh_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 head_conv=128,
                 num_classes=80,
                 hm_weight=1.,
                 wh_weight=5.,
                 wh_offset_base=16.,
                 planes=(256, 128, 64),
                 shortcut_num=(1, 2, 3),
                 wh_head_conv_num=2,
                 hm_head_conv_num=2,
                 wh_conv=64,
                 wh_planes=4,
                 score_thresh=0.01,
                 max_per_img=100,
                 base_down_ratio=32,
                 wh_loss='GiouLoss',
                 dcn_upsample=True,
                 dcn_head=False,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9):
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
        self.wh_planes = wh_planes
        self.score_thresh = score_thresh
        self.max_per_img = max_per_img
        self.down_ratio = base_down_ratio // 2**len(planes)
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.wh_loss = wh_loss
        self.dcn_upsample = dcn_upsample
        self.dcn_head = dcn_head
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob

    def shortcut(self, x, out_c, layer_num, kernel_size=3, padding=1,
                 name=None):
        assert layer_num > 0
        for i in range(layer_num):
            act = 'relu' if i < layer_num - 1 else None
            fan_out = kernel_size * kernel_size * out_c
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
        fan_in = x.shape[1] * 3 * 3
        stdv = 1. / math.sqrt(fan_in)
        if self.dcn_upsample:
            conv = DeformConv(
                x,
                out_c,
                3,
                initializer=Uniform(-stdv, stdv),
                bias_attr=True,
                name=name + '.0')
        else:
            conv = fluid.layers.conv2d(
                x,
                out_c,
                3,
                padding=1,
                param_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
                bias_attr=ParamAttr(
                    learning_rate=2., regularizer=L2Decay(0.)))

        norm_name = name + '.1'
        pattr = ParamAttr(name=norm_name + '.weight', initializer=Constant(1.))
        battr = ParamAttr(name=norm_name + '.bias', initializer=Constant(0.))
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

    def _head(self,
              x,
              out_c,
              conv_num=1,
              head_out_c=None,
              name=None,
              is_test=False):
        head_out_c = self.head_conv if not head_out_c else head_out_c
        conv_w_std = 0.01 if '.hm' in name else 0.001
        conv_w_init = Normal(0, conv_w_std)
        for i in range(conv_num):
            conv_name = '{}.{}.conv'.format(name, i)
            if self.dcn_head:
                x = DeformConv(
                    x,
                    head_out_c,
                    3,
                    initializer=conv_w_init,
                    name=conv_name + '.dcn')
                x = fluid.layers.relu(x)
            else:
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
        if self.drop_block and '.hm' in name:
            x = DropBlock(
                x,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
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

    def hm_head(self, x, name=None, is_test=False):
        hm = self._head(
            x,
            self.num_classes,
            self.hm_head_conv_num,
            name=name,
            is_test=is_test)
        return hm

    def wh_head(self, x, name=None):
        planes = self.wh_planes
        wh = self._head(
            x, planes, self.wh_head_conv_num, self.wh_conv, name=name)
        return fluid.layers.relu(wh)

    def get_output(self, input, name=None, is_test=False):
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

        hm = self.hm_head(feat, name=name + '.hm', is_test=is_test)
        wh = self.wh_head(feat, name=name + '.wh') * self.wh_offset_base

        return hm, wh

    def _simple_nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = fluid.layers.pool2d(heat, kernel, 'max', pool_padding=pad)
        keep = fluid.layers.cast(hmax == heat, 'float32')
        return heat * keep

    def _topk(self, scores, k):
        cat, height, width = scores.shape[1:]
        # batch size is 1
        scores_r = fluid.layers.reshape(scores, [cat, -1])
        topk_scores, topk_inds = fluid.layers.topk(scores_r, k)
        topk_ys = topk_inds / width
        topk_xs = topk_inds % width

        topk_score_r = fluid.layers.reshape(topk_scores, [-1])
        topk_score, topk_ind = fluid.layers.topk(topk_score_r, k)
        topk_clses = fluid.layers.cast(topk_ind / k, 'float32')

        topk_inds = fluid.layers.reshape(topk_inds, [-1])
        topk_ys = fluid.layers.reshape(topk_ys, [-1, 1])
        topk_xs = fluid.layers.reshape(topk_xs, [-1, 1])
        topk_inds = fluid.layers.gather(topk_inds, topk_ind)
        topk_ys = fluid.layers.gather(topk_ys, topk_ind)
        topk_xs = fluid.layers.gather(topk_xs, topk_ind)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def get_bboxes(self, heatmap, wh, scale_factor):
        heatmap = fluid.layers.sigmoid(heatmap)
        heat = self._simple_nms(heatmap)
        scores, inds, clses, ys, xs = self._topk(heat, self.max_per_img)
        ys = fluid.layers.cast(ys, 'float32') * self.down_ratio
        xs = fluid.layers.cast(xs, 'float32') * self.down_ratio
        scores = fluid.layers.unsqueeze(scores, [1])
        clses = fluid.layers.unsqueeze(clses, [1])

        wh_t = fluid.layers.transpose(wh, [0, 2, 3, 1])
        wh = fluid.layers.reshape(wh_t, [-1, wh_t.shape[-1]])
        wh = fluid.layers.gather(wh, inds)

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
        index.stop_gradient = True
        weight = fluid.layers.gather_nd(weight, index)
        pred = fluid.layers.gather_nd(pred, index)
        target = fluid.layers.gather_nd(target, index)
        return pred, target, weight

    def get_loss(self, pred_hm, pred_wh, target_hm, box_target, target_weight):
        pred_hm = paddle.tensor.clamp(
            fluid.layers.sigmoid(pred_hm), 1e-4, 1 - 1e-4)
        hm_loss = self.ct_focal_loss(pred_hm, target_hm) * self.hm_weight
        shape = fluid.layers.shape(target_hm)
        shape.stop_gradient = True
        H, W = shape[2], shape[3]

        mask = fluid.layers.reshape(target_weight, [-1, H, W])
        avg_factor = fluid.layers.reduce_sum(mask) + 1e-4
        base_step = self.down_ratio
        zero = fluid.layers.fill_constant(shape=[1], value=0, dtype='int32')
        shifts_x = paddle.arange(zero, W * base_step, base_step, dtype='int32')
        shifts_y = paddle.arange(zero, H * base_step, base_step, dtype='int32')
        shift_y, shift_x = paddle.tensor.meshgrid([shifts_y, shifts_x])
        base_loc = fluid.layers.stack([shift_x, shift_y], axis=0)
        base_loc.stop_gradient = True

        pred_boxes = fluid.layers.concat(
            [0 - pred_wh[:, 0:2, :, :] + base_loc, pred_wh[:, 2:4] + base_loc],
            axis=1)
        pred_boxes = fluid.layers.transpose(pred_boxes, [0, 2, 3, 1])
        boxes = fluid.layers.transpose(box_target, [0, 2, 3, 1])
        boxes.stop_gradient = True

        pred_boxes, boxes, mask = self.filter_box_by_weight(pred_boxes, boxes,
                                                            mask)
        mask.stop_gradient = True
        wh_loss = self.wh_loss(
            pred_boxes, boxes, outside_weight=mask, use_transform=False)
        wh_loss = wh_loss / avg_factor

        ttf_loss = {'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return ttf_loss
