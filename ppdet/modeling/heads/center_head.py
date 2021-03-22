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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Uniform, Normal
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling.backbones.dla import ConvLayer


def focal_loss(pred, gt):
    pos_mask = gt == 1
    num_pos = paddle.sum(paddle.cast(pos_mask, dtype=np.int32))
    pos_mask = paddle.cast(pos_mask, pred.dtype)
    neg_mask = gt < 1
    neg_mask = paddle.cast(neg_mask, pred.dtype)

    neg_weights = paddle.pow(1 - gt, 4)

    loss = 0

    pos_mask.stop_gradient = True
    neg_weights.stop_gradient = True
    neg_mask.stop_gradient = True
    pos_loss = paddle.log(pred) * paddle.pow(1 - pred, 2) * pos_mask
    neg_loss = paddle.log(1 - pred) * paddle.pow(pred,
                                                 2) * neg_weights * neg_mask

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


@register
class CenterHead(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 in_channels,
                 num_classes=80,
                 head_planes=256,
                 heatmap_weight=1,
                 size_weight=0.1,
                 offset_weight=1):
        super(CenterHead, self).__init__()
        self.weights = {
            'heatmap': heatmap_weight,
            'size': size_weight,
            'offset': offset_weight
        }
        self.heatmap = nn.Sequential(
            ConvLayer(
                in_channels,
                head_planes,
                kernel_size=3,
                padding=1,
                bias=True,
                name="hm.0"),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                name="hm.2"))
        self.heatmap[2].conv.bias[:] = -2.19
        self.size = nn.Sequential(
            ConvLayer(
                in_channels,
                head_planes,
                kernel_size=3,
                padding=1,
                bias=True,
                name="wh.0"),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                4,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                name="wh.2"))
        self.offset = nn.Sequential(
            ConvLayer(
                in_channels,
                head_planes,
                kernel_size=3,
                padding=1,
                bias=True,
                name="reg.0"),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                name="reg.2"))

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self, feat, inputs):
        heatmap = self.heatmap(feat)
        #print('--------------heat map before sigmoid', np.mean(heatmap.numpy()))
        heatmap = F.sigmoid(heatmap)
        #print('---------neck feat', np.mean(feat.numpy()))
        size = self.size(feat)
        offset = self.offset(feat)
        if self.training:
            loss = self.get_loss(heatmap, size, offset, self.weights, inputs)
            return loss
        else:
            return {'heatmap': heatmap, 'size': size, 'offset': offset}

    def get_loss(self, heatmap, size, offset, weights, inputs):
        heatmap_target = inputs['heatmap']
        size_target = inputs['size']
        offset_target = inputs['offset']
        index = inputs['index']
        mask = inputs['index_mask']
        #print('------heatmap', np.mean(heatmap.numpy()))
        #print('------size', np.mean(size.numpy())) 
        #print('------offset', np.mean(offset.numpy()))
        heatmap_loss = focal_loss(heatmap, heatmap_target)

        size = paddle.transpose(size, perm=[0, 2, 3, 1])
        size_n, size_h, size_w, size_c = size.shape
        size = paddle.reshape(size, shape=[size_n, -1, size_c])
        index = paddle.unsqueeze(index, 2)
        batch_inds = list()
        for i in range(size_n):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)
        #print('gather_nd size', np.mean(size.numpy()))
        pos_size = paddle.gather_nd(size, index=index)
        mask = paddle.unsqueeze(mask, axis=2)
        size_mask = paddle.expand_as(mask, pos_size)
        size_mask = paddle.cast(size_mask, dtype=pos_size.dtype)
        pos_num = size_mask.sum()
        size_mask.stop_gradient = True
        size_target.stop_gradient = True
        #print('pos_size', np.mean(pos_size.numpy()), pos_size.numpy()[0, 0, :])
        #print('size_target', np.mean(size_target.numpy()), size_target.numpy()[0, 0, :])
        #print('mask', np.mean(mask.numpy()), mask.numpy()[0, 0])
        size_loss = F.l1_loss(
            pos_size * size_mask, size_target * size_mask, reduction='sum')
        #print('size_loss', size_loss.numpy())
        #print('pos num', np.mean(pos_num.numpy()))
        size_loss = size_loss / (pos_num + 1e-4)

        offset = paddle.transpose(offset, perm=[0, 2, 3, 1])
        offset_n, offset_h, offset_w, offset_c = offset.shape
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

        det_loss = weights['heatmap'] * heatmap_loss + weights[
            'size'] * size_loss + weights['offset'] * offset_loss
        #print('weights heatmap', weights['heatmap'], heatmap_loss.numpy().mean())
        #print('weights size', weights['size'], size_loss.numpy().mean())
        #print('weights offset', weights['offset'], offset_loss.numpy().mean())
        #print('det loss', det_loss.numpy().mean())

        return {
            'det_loss': det_loss,
            'heatmap_loss': heatmap_loss,
            'size_loss': size_loss,
            'offset_loss': offset_loss
        }
