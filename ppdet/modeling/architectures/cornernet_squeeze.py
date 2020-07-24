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

from ppdet.core.workspace import register
import numpy as np
from ppdet.utils.check import check_version

__all__ = ['CornerNetSqueeze']


def rescale_bboxes(bboxes, ratios, borders):
    x1, y1, x2, y2 = fluid.layers.split(bboxes, 4)
    x1 = x1 / ratios[:, 1] - borders[:, 2]
    x2 = x2 / ratios[:, 1] - borders[:, 2]
    y1 = y1 / ratios[:, 0] - borders[:, 0]
    y2 = y2 / ratios[:, 0] - borders[:, 0]
    return fluid.layers.concat([x1, y1, x2, y2], axis=2)


@register
class CornerNetSqueeze(object):
    """
    """
    __category__ = 'architecture'
    __inject__ = ['backbone', 'corner_head', 'fpn']
    __shared__ = ['num_classes']

    def __init__(self,
                 backbone,
                 corner_head='CornerHead',
                 num_classes=80,
                 fpn=None):
        check_version('1.8.0')
        super(CornerNetSqueeze, self).__init__()
        self.backbone = backbone
        self.corner_head = corner_head
        self.num_classes = num_classes
        self.fpn = fpn

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        body_feats = self.backbone(im)
        if self.fpn is not None:
            body_feats, _ = self.fpn.get_output(body_feats)
            body_feats = [list(body_feats.values())[-1]]
        if mode == 'train':
            target_vars = [
                'tl_heatmaps', 'br_heatmaps', 'tag_masks', 'tl_regrs',
                'br_regrs', 'tl_tags', 'br_tags'
            ]
            target = {key: feed_vars[key] for key in target_vars}
            self.corner_head.get_output(body_feats)
            loss = self.corner_head.get_loss(target)
            return loss

        elif mode == 'test':
            ratios = feed_vars['ratios']
            borders = feed_vars['borders']
            bboxes, scores, tl_scores, br_scores, clses = self.corner_head.get_prediction(
                body_feats[-1])
            bboxes = rescale_bboxes(bboxes, ratios, borders)
            detections = fluid.layers.concat([clses, scores, bboxes], axis=2)

            detections = detections[0]
            return {'bbox': detections}

    def _inputs_def(self, image_shape, output_size, max_tag_len):
        im_shape = [None] + image_shape
        C = self.num_classes
        # yapf: disable
        inputs_def = {
            'image':        {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_id':        {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'gt_bbox':       {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class':     {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'ratios':       {'shape': [None, 2],  'dtype': 'float32', 'lod_level': 0},
            'borders':      {'shape': [None, 4],  'dtype': 'float32', 'lod_level': 0},
            'tl_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'br_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'tl_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'br_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'tl_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'br_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'tag_masks':     {'shape': [None, max_tag_len], 'dtype': 'int32', 'lod_level': 0},
        }
        # yapf: enable
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=[
                'image', 'im_id', 'gt_box', 'gt_class', 'tl_heatmaps',
                'br_heatmaps', 'tl_regrs', 'br_regrs', 'tl_tags', 'br_tags',
                'tag_masks'
            ],  # for train
            output_size=64,
            max_tag_len=128,
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape, output_size, max_tag_len)
        feed_vars = OrderedDict([(key, fluid.data(
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

    def test(self, feed_vars, exclude_nms=False):
        assert not exclude_nms, "exclude_nms for {} is not support currently".format(
            self.__class__.__name__)
        return self.build(feed_vars, mode='test')
