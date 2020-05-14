# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

import paddle.fluid as fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['RetinaNet']


@register
class RetinaNet(object):
    """
    RetinaNet architecture, see https://arxiv.org/abs/1708.02002

    Args:
        backbone (object): backbone instance
        fpn (object): feature pyramid network instance
        retina_head (object): `RetinaHead` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'retina_head']

    def __init__(self, backbone, fpn, retina_head):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.retina_head = retina_head

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if mode == 'train':
            gt_bbox = feed_vars['gt_bbox']
            gt_class = feed_vars['gt_class']
            is_crowd = feed_vars['is_crowd']

        mixed_precision_enabled = mixed_precision_global_state() is not None
        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        # backbone
        body_feats = self.backbone(im)

        # cast features back to FP32
        if mixed_precision_enabled:
            body_feats = OrderedDict((k, fluid.layers.cast(v, 'float32'))
                                     for k, v in body_feats.items())

        # FPN
        body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # retinanet head
        if mode == 'train':
            loss = self.retina_head.get_loss(body_feats, spatial_scale, im_info,
                                             gt_bbox, gt_class, is_crowd)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.retina_head.get_prediction(body_feats, spatial_scale,
                                                   im_info)
            return pred

    def _inputs_def(self, image_shape):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_info':  {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_id':    {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'im_shape': {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'gt_bbox':  {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'is_crowd': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'is_difficult': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 1},
        }
        # yapf: enable
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=[
                'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd'
            ],  # for-train
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape)
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=16,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'test')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')
