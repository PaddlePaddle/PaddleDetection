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

__all__ = ['FCOS']


@register
class FCOS(object):
    """
    FCOS architecture, see https://arxiv.org/abs/1904.01355

    Args:
        backbone (object): backbone instance
        fpn (object): feature pyramid network instance
        fcos_head (object): `FCOSHead` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'fcos_head']

    def __init__(self, backbone, fpn, fcos_head):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.fcos_head = fcos_head

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        im_info = feed_vars['im_info']

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

        # fcosnet head
        if mode == 'train':
            tag_labels = []
            tag_bboxes = []
            tag_centerness = []
            for i in range(len(self.fcos_head.fpn_stride)):
                # reg_target, labels, scores, centerness
                k_lbl = 'labels{}'.format(i)
                if k_lbl in feed_vars:
                    tag_labels.append(feed_vars[k_lbl])
                k_box = 'reg_target{}'.format(i)
                if k_box in feed_vars:
                    tag_bboxes.append(feed_vars[k_box])
                k_ctn = 'centerness{}'.format(i)
                if k_ctn in feed_vars:
                    tag_centerness.append(feed_vars[k_ctn])
            # tag_labels, tag_bboxes, tag_centerness
            loss = self.fcos_head.get_loss(body_feats, tag_labels, tag_bboxes,
                                           tag_centerness)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.fcos_head.get_prediction(body_feats, im_info)
            return pred

    def _inputs_def(self, image_shape, fields):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_shape': {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_info':  {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_id':    {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'gt_bbox':  {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'gt_score': {'shape': [None, 1], 'dtype': 'float32', 'lod_level': 1},
            'is_crowd': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'is_difficult': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 1}
        }
        # yapf: disable
        if 'gt_bbox' in fields:
            targets_def = {
                'labels0':      {'shape': [None, None, None, 1],  'dtype': 'int32',     'lod_level': 0},
                'reg_target0':  {'shape': [None, None, None, 4],  'dtype': 'float32',   'lod_level': 0},
                'centerness0':  {'shape': [None, None, None, 1],  'dtype': 'float32',   'lod_level': 0},
                'labels1':      {'shape': [None, None, None, 1],  'dtype': 'int32',     'lod_level': 0},
                'reg_target1':  {'shape': [None, None, None, 4],  'dtype': 'float32',   'lod_level': 0},
                'centerness1':  {'shape': [None, None, None, 1],  'dtype': 'float32',   'lod_level': 0},
                'labels2':      {'shape': [None, None, None, 1],  'dtype': 'int32',     'lod_level': 0},
                'reg_target2':  {'shape': [None, None, None, 4],  'dtype': 'float32',   'lod_level': 0},
                'centerness2':  {'shape': [None, None, None, 1],  'dtype': 'float32',   'lod_level': 0},
                'labels3':      {'shape': [None, None, None, 1],  'dtype': 'int32',     'lod_level': 0},
                'reg_target3':  {'shape': [None, None, None, 4],  'dtype': 'float32',   'lod_level': 0},
                'centerness3':  {'shape': [None, None, None, 1],  'dtype': 'float32',   'lod_level': 0},
                'labels4':      {'shape': [None, None, None, 1],  'dtype': 'int32',     'lod_level': 0},
                'reg_target4':  {'shape': [None, None, None, 4],  'dtype': 'float32',   'lod_level': 0},
                'centerness4':  {'shape': [None, None, None, 1],  'dtype': 'float32',   'lod_level': 0},
            }
            # yapf: enable

            # downsample = 128
            for k, stride in enumerate(self.fcos_head.fpn_stride):
                k_lbl = 'labels{}'.format(k)
                k_box = 'reg_target{}'.format(k)
                k_ctn = 'centerness{}'.format(k)
                grid_y = image_shape[-2] // stride if image_shape[-2] else None
                grid_x = image_shape[-1] // stride if image_shape[-1] else None
                if grid_x is not None:
                    num_pts = grid_x * grid_y
                    num_dim2 = 1
                else:
                    num_pts = None
                    num_dim2 = None
                targets_def[k_lbl]['shape'][1] = num_pts
                targets_def[k_box]['shape'][1] = num_pts
                targets_def[k_ctn]['shape'][1] = num_pts
                targets_def[k_lbl]['shape'][2] = num_dim2
                targets_def[k_box]['shape'][2] = num_dim2
                targets_def[k_ctn]['shape'][2] = num_dim2
            inputs_def.update(targets_def)
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=[
                'image', 'im_shape', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd'
            ],  # for-train
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape, fields)
        if "gt_bbox" in fields:
            for i in range(len(self.fcos_head.fpn_stride)):
                fields.extend(
                    ['labels%d' % i, 'reg_target%d' % i, 'centerness%d' % i])
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
