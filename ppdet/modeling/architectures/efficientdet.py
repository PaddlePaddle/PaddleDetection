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

from collections import OrderedDict

import paddle.fluid as fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['EfficientDet']


@register
class EfficientDet(object):
    """
    EfficientDet architecture, see https://arxiv.org/abs/1911.09070

    Args:
        backbone (object): backbone instance
        fpn (object): feature pyramid network instance
        retina_head (object): `RetinaHead` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'efficient_head', 'anchor_grid']

    def __init__(self,
                 backbone,
                 fpn,
                 efficient_head,
                 anchor_grid,
                 box_loss_weight=50.):
        super(EfficientDet, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.efficient_head = efficient_head
        self.anchor_grid = anchor_grid
        self.box_loss_weight = box_loss_weight

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        if mode == 'train':
            gt_labels = feed_vars['gt_label']
            gt_targets = feed_vars['gt_target']
            fg_num = feed_vars['fg_num']
        else:
            im_info = feed_vars['im_info']

        mixed_precision_enabled = mixed_precision_global_state() is not None
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')
        body_feats = self.backbone(im)
        if mixed_precision_enabled:
            body_feats = [fluid.layers.cast(f, 'float32') for f in body_feats]
        body_feats = self.fpn(body_feats)

        # XXX not used for training, but the parameters are needed when
        # exporting inference model
        anchors = self.anchor_grid()

        if mode == 'train':
            loss = self.efficient_head.get_loss(body_feats, gt_labels,
                                                gt_targets, fg_num)
            loss_cls = loss['loss_cls']
            loss_bbox = loss['loss_bbox']
            total_loss = loss_cls + self.box_loss_weight * loss_bbox
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.efficient_head.get_prediction(body_feats, anchors,
                                                      im_info)
            return pred

    def _inputs_def(self, image_shape):
        im_shape = [None] + image_shape
        inputs_def = {
            'image': {
                'shape': im_shape,
                'dtype': 'float32'
            },
            'im_info': {
                'shape': [None, 3],
                'dtype': 'float32'
            },
            'im_id': {
                'shape': [None, 1],
                'dtype': 'int64'
            },
            'im_shape': {
                'shape': [None, 3],
                'dtype': 'float32'
            },
            'fg_num': {
                'shape': [None, 1],
                'dtype': 'int32'
            },
            'gt_label': {
                'shape': [None, None, 1],
                'dtype': 'int32'
            },
            'gt_target': {
                'shape': [None, None, 4],
                'dtype': 'float32'
            },
        }
        return inputs_def

    def build_inputs(self,
                     image_shape=[3, None, None],
                     fields=[
                         'image', 'im_info', 'im_id', 'fg_num', 'gt_label',
                         'gt_target'
                     ],
                     use_dataloader=True,
                     iterable=False):
        inputs_def = self._inputs_def(image_shape)
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'])) for key in fields])
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
