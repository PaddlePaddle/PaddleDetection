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

__all__ = ['SOLOv2']


@register
class SOLOv2(object):
    """
    SOLOv2 network, see https://arxiv.org/abs/2003.10152

    Args:
        backbone (object): an backbone instance
        fpn (object): feature pyramid network instance
        bbox_head (object): an `SOLOv2Head` instance
        mask_head (object): an `SOLOv2MaskHead` instance
        batch_size (int): batch size.
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'bbox_head', 'mask_head']

    def __init__(self,
                 backbone,
                 fpn=None,
                 bbox_head='SOLOv2Head',
                 mask_head='SOLOv2MaskHead',
                 batch_size=1):
        super(SOLOv2, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.bbox_head = bbox_head
        self.mask_head = mask_head
        self.batch_size = batch_size

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']

        mixed_precision_enabled = mixed_precision_global_state() is not None

        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        body_feats = self.backbone(im)

        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        if isinstance(body_feats, OrderedDict):
            body_feat_names = list(body_feats.keys())
            body_feats = [body_feats[name] for name in body_feat_names]

        # cast features back to FP32
        if mixed_precision_enabled:
            body_feats = [fluid.layers.cast(v, 'float32') for v in body_feats]

        if not mode == 'train':
            self.batch_size = 1

        mask_feat_pred = self.mask_head.get_output(body_feats, self.batch_size)

        if mode == 'train':
            ins_labels = []
            cate_labels = []
            grid_orders = []
            fg_num = feed_vars['fg_num']
            grid_offset = feed_vars['grid_offset']

            for i in range(5):
                ins_label = 'ins_label{}'.format(i)
                if ins_label in feed_vars:
                    ins_labels.append(feed_vars[ins_label])
                cate_label = 'cate_label{}'.format(i)
                if cate_label in feed_vars:
                    cate_labels.append(feed_vars[cate_label])
                grid_order = 'grid_order{}'.format(i)
                if grid_order in feed_vars:
                    grid_orders.append(feed_vars[grid_order])

            cate_preds, kernel_preds = self.bbox_head.get_outputs(
                body_feats, batch_size=self.batch_size)

            losses = self.bbox_head.get_loss(
                cate_preds, kernel_preds, mask_feat_pred, ins_labels,
                cate_labels, grid_orders, fg_num, grid_offset, self.batch_size)
            total_loss = fluid.layers.sum(list(losses.values()))
            losses.update({'loss': total_loss})
            return losses
        else:
            im_info = feed_vars['im_info']
            outs = self.bbox_head.get_outputs(
                body_feats, is_eval=True, batch_size=self.batch_size)
            seg_inputs = outs + (mask_feat_pred, im_info)
            return self.bbox_head.get_prediction(*seg_inputs)

    def _inputs_def(self, image_shape, fields):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,   'dtype': 'float32', 'lod_level': 0},
            'im_info':  {'shape': [None, 3],  'dtype': 'float32', 'lod_level': 0},
            'im_id':    {'shape': [None, 1],  'dtype': 'int64',   'lod_level': 0},
            'im_shape': {'shape': [None, 3],  'dtype': 'float32', 'lod_level': 0},
        }

        if 'gt_segm' in fields:
            targets_def = {
                'ins_label0':  {'shape': [None, None, None], 'dtype': 'int32', 'lod_level': 1},
                'ins_label1':  {'shape': [None, None, None], 'dtype': 'int32', 'lod_level': 1},
                'ins_label2':  {'shape': [None, None, None], 'dtype': 'int32', 'lod_level': 1},
                'ins_label3':  {'shape': [None, None, None], 'dtype': 'int32', 'lod_level': 1},
                'ins_label4':  {'shape': [None, None, None], 'dtype': 'int32', 'lod_level': 1},
                'cate_label0': {'shape': [None],       'dtype': 'int32', 'lod_level': 1},
                'cate_label1': {'shape': [None],       'dtype': 'int32', 'lod_level': 1},
                'cate_label2': {'shape': [None],       'dtype': 'int32', 'lod_level': 1},
                'cate_label3': {'shape': [None],       'dtype': 'int32', 'lod_level': 1},
                'cate_label4': {'shape': [None],       'dtype': 'int32', 'lod_level': 1},
                'grid_order0': {'shape': [None], 'dtype': 'int32', 'lod_level': 1},
                'grid_order1': {'shape': [None], 'dtype': 'int32', 'lod_level': 1},
                'grid_order2': {'shape': [None], 'dtype': 'int32', 'lod_level': 1},
                'grid_order3': {'shape': [None], 'dtype': 'int32', 'lod_level': 1},
                'grid_order4': {'shape': [None], 'dtype': 'int32', 'lod_level': 1},
                'fg_num':      {'shape': [None],             'dtype': 'int32', 'lod_level': 0},
                'grid_offset': {'shape': [None, 5], 'dtype': 'int32', 'lod_level': 0},
            }
            # yapf: enable
            inputs_def.update(targets_def)
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=['image', 'im_id', 'gt_segm'],  # for train
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape, fields)
        if 'gt_segm' in fields:
            fields.remove('gt_segm')
            fields.extend(['fg_num', 'grid_offset'])
            for i in range(5):
                fields.extend([
                    'ins_label%d' % i, 'cate_label%d' % i, 'grid_order%d' % i
                ])

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

    def test(self, feed_vars):
        return self.build(feed_vars, mode='test')
