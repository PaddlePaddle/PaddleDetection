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

from collections import OrderedDict

from paddle import fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['TTFNet']


@register
class TTFNet(object):
    """
    TTFNet network, see https://arxiv.org/abs/1909.00700

    Args:
        backbone (object): backbone instance
        ttf_head (object): `TTFHead` instance
        num_classes (int): the number of classes, 80 by default.
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'ttf_head']
    __shared__ = ['num_classes']

    def __init__(self, backbone, ttf_head='TTFHead', num_classes=80):
        super(TTFNet, self).__init__()
        self.backbone = backbone
        self.ttf_head = ttf_head
        self.num_classes = num_classes

    def build(self, feed_vars, mode='train', exclude_nms=False):
        im = feed_vars['image']

        mixed_precision_enabled = mixed_precision_global_state() is not None

        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        body_feats = self.backbone(im)

        if isinstance(body_feats, OrderedDict):
            body_feat_names = list(body_feats.keys())
            body_feats = [body_feats[name] for name in body_feat_names]

        # cast features back to FP32
        if mixed_precision_enabled:
            body_feats = [fluid.layers.cast(v, 'float32') for v in body_feats]

        predict_hm, predict_wh = self.ttf_head.get_output(
            body_feats, 'ttf_head', is_test=mode == 'test')
        if mode == 'train':
            heatmap = feed_vars['ttf_heatmap']
            box_target = feed_vars['ttf_box_target']
            reg_weight = feed_vars['ttf_reg_weight']
            loss = self.ttf_head.get_loss(predict_hm, predict_wh, heatmap,
                                          box_target, reg_weight)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            results = self.ttf_head.get_bboxes(predict_hm, predict_wh,
                                               feed_vars['scale_factor'])
            return results

    def _inputs_def(self, image_shape, downsample):
        im_shape = [None] + image_shape
        H, W = im_shape[2:]
        target_h = None if H is None else H // downsample
        target_w = None if W is None else W // downsample
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,                 'dtype': 'float32', 'lod_level': 0},
            'scale_factor':  {'shape': [None, 4],           'dtype': 'float32',   'lod_level': 0},
            'im_id':    {'shape': [None, 1],                'dtype': 'int64',   'lod_level': 0},
            'ttf_heatmap':  {'shape': [None, self.num_classes, target_h, target_w], 'dtype': 'float32', 'lod_level': 0},
            'ttf_box_target': {'shape': [None, 4, target_h, target_w],    'dtype': 'float32',   'lod_level': 0},
            'ttf_reg_weight': {'shape': [None, 1, target_h, target_w],    'dtype': 'float32', 'lod_level': 0},
        }
        # yapf: enable

        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=[
                'image', 'ttf_heatmap', 'ttf_box_target', 'ttf_reg_weight'
            ],  # for train
            use_dataloader=True,
            iterable=False,
            downsample=4):
        inputs_def = self._inputs_def(image_shape, downsample)
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
        return self.build(feed_vars, mode='train')

    def eval(self, feed_vars):
        return self.build(feed_vars, mode='test')

    def test(self, feed_vars, exclude_nms=False):
        return self.build(feed_vars, mode='test', exclude_nms=exclude_nms)
