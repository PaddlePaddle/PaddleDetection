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

from paddle import fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['YOLOv3']


@register
class YOLOv3(object):
    """
    YOLOv3 network, see https://arxiv.org/abs/1804.02767

    Args:
        backbone (object): an backbone instance
        yolo_head (object): an `YOLOv3Head` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'yolo_head']

    def __init__(self, backbone, yolo_head='YOLOv3Head'):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.yolo_head = yolo_head

    def build(self, feed_vars, mode='train'):
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

        if mode == 'train':
            gt_bbox = feed_vars['gt_bbox']
            gt_class = feed_vars['gt_class']
            gt_score = feed_vars['gt_score']

            return {
                'loss': self.yolo_head.get_loss(body_feats, gt_bbox, gt_class,
                                                gt_score)
            }
        else:
            im_size = feed_vars['im_size']
            return self.yolo_head.get_prediction(body_feats, im_size)

    def _inputs_def(self, image_shape, num_max_boxes):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,                 'dtype': 'float32', 'lod_level': 0},
     'im_size':  {'shape': [None, 2],                'dtype': 'int32',   'lod_level': 0},
            'im_id':    {'shape': [None, 1],                'dtype': 'int32',   'lod_level': 0},
            'gt_bbox':  {'shape': [None, num_max_boxes, 4], 'dtype': 'float32', 'lod_level': 0},
            'gt_class': {'shape': [None, num_max_boxes],    'dtype': 'int32',   'lod_level': 0},
            'gt_score': {'shape': [None, num_max_boxes],    'dtype': 'float32', 'lod_level': 0},
            'is_difficult': {'shape': [None, num_max_boxes],'dtype': 'int32',   'lod_level': 0},
        }
        # yapf: enable
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, 608, 608],
            fields=['image', 'gt_bbox', 'gt_class', 'gt_score'],  # for train
            num_max_boxes=50,
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape, num_max_boxes)
        feed_vars = OrderedDict([(key, fluid.layers.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=64,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader

    def train(self, feed_vars):
        return self.build(feed_vars, mode='train')

    def eval(self, feed_vars):
        return self.build(feed_vars, mode='test')

    def test(self, feed_vars):
        return self.build(feed_vars, mode='test')
