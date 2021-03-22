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
import copy

from paddle import fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

from .input_helper import multiscale_def

__all__ = ['FasterRCNN']


@register
class FasterRCNN(object):
    """
    Faster R-CNN architecture, see https://arxiv.org/abs/1506.01497
    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNhead` instance
        bbox_assigner (object): `BBoxAssigner` instance
        roi_extractor (object): ROI extractor instance
        bbox_head (object): `BBoxHead` instance
        fpn (object): feature pyramid network instance
    """

    __category__ = 'architecture'
    __inject__ = [
        'backbone', 'rpn_head', 'bbox_assigner', 'roi_extractor', 'bbox_head',
        'fpn'
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_extractor,
                 bbox_head='BBoxHead',
                 bbox_assigner='BBoxAssigner',
                 rpn_only=False,
                 fpn=None):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_assigner = bbox_assigner
        self.roi_extractor = roi_extractor
        self.bbox_head = bbox_head
        self.fpn = fpn
        self.rpn_only = rpn_only

    def build(self, feed_vars, mode='train'):
        if mode == 'train':
            required_fields = ['gt_class', 'gt_bbox', 'is_crowd', 'im_info']
        else:
            required_fields = ['im_shape', 'im_info']
        self._input_check(required_fields, feed_vars)

        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if mode == 'train':
            gt_bbox = feed_vars['gt_bbox']
            is_crowd = feed_vars['is_crowd']
        else:
            im_shape = feed_vars['im_shape']

        mixed_precision_enabled = mixed_precision_global_state() is not None

        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        body_feats = self.backbone(im)
        body_feat_names = list(body_feats.keys())

        # cast features back to FP32
        if mixed_precision_enabled:
            body_feats = OrderedDict((k, fluid.layers.cast(v, 'float32'))
                                     for k, v in body_feats.items())

        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        rois = self.rpn_head.get_proposals(body_feats, im_info, mode=mode)

        if mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, gt_bbox, is_crowd)
            # sampled rpn proposals
            for var in ['gt_class', 'is_crowd', 'gt_bbox', 'im_info']:
                assert var in feed_vars, "{} has no {}".format(feed_vars, var)
            outs = self.bbox_assigner(
                rpn_rois=rois,
                gt_classes=feed_vars['gt_class'],
                is_crowd=feed_vars['is_crowd'],
                gt_boxes=feed_vars['gt_bbox'],
                im_info=feed_vars['im_info'])

            rois = outs[0]
            labels_int32 = outs[1]
            bbox_targets = outs[2]
            bbox_inside_weights = outs[3]
            bbox_outside_weights = outs[4]
        else:
            if self.rpn_only:
                im_scale = fluid.layers.slice(
                    im_info, [1], starts=[2], ends=[3])
                im_scale = fluid.layers.sequence_expand(im_scale, rois)
                rois = rois / im_scale
                return {'proposal': rois}
        if self.fpn is None:
            # in models without FPN, roi extractor only uses the last level of
            # feature maps. And body_feat_names[-1] represents the name of
            # last feature map.
            body_feat = body_feats[body_feat_names[-1]]
            roi_feat = self.roi_extractor(body_feat, rois)
        else:
            roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

        if mode == 'train':
            loss = self.bbox_head.get_loss(roi_feat, labels_int32, bbox_targets,
                                           bbox_inside_weights,
                                           bbox_outside_weights)
            loss.update(rpn_loss)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.bbox_head.get_prediction(roi_feat, rois, im_info,
                                                 im_shape)
            return pred

    def build_multi_scale(self, feed_vars):
        required_fields = ['image', 'im_info', 'im_shape']
        self._input_check(required_fields, feed_vars)

        result = {}
        im_shape = feed_vars['im_shape']
        result['im_shape'] = im_shape
        for i in range(len(self.im_info_names) // 2):
            im = feed_vars[self.im_info_names[2 * i]]
            im_info = feed_vars[self.im_info_names[2 * i + 1]]
            body_feats = self.backbone(im)
            body_feat_names = list(body_feats.keys())

            if self.fpn is not None:
                body_feats, spatial_scale = self.fpn.get_output(body_feats)

            rois = self.rpn_head.get_proposals(body_feats, im_info, mode='test')

            if self.fpn is None:
                # in models without FPN, roi extractor only uses the last level of
                # feature maps. And body_feat_names[-1] represents the name of
                # last feature map.
                body_feat = body_feats[body_feat_names[-1]]
                roi_feat = self.roi_extractor(body_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

            pred = self.bbox_head.get_prediction(
                roi_feat, rois, im_info, im_shape, return_box_score=True)
            bbox_name = 'bbox_' + str(i)
            score_name = 'score_' + str(i)
            if 'flip' in im.name:
                bbox_name += '_flip'
                score_name += '_flip'
            result[bbox_name] = pred['bbox']
            result[score_name] = pred['score']
        return result

    def _input_check(self, require_fields, feed_vars):
        for var in require_fields:
            assert var in feed_vars, \
                "{} has no {} field".format(feed_vars, var)

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
            ],  # for train
            multi_scale=False,
            num_scales=-1,
            use_flip=None,
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape)
        fields = copy.deepcopy(fields)
        if multi_scale:
            ms_def, ms_fields = multiscale_def(image_shape, num_scales,
                                               use_flip)
            inputs_def.update(ms_def)
            fields += ms_fields
            self.im_info_names = ['image', 'im_info'] + ms_fields

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

    def eval(self, feed_vars, multi_scale=None):
        if multi_scale:
            return self.build_multi_scale(feed_vars)
        return self.build(feed_vars, 'test')

    def test(self, feed_vars, exclude_nms=False):
        assert not exclude_nms, "exclude_nms for {} is not support currently".format(
            self.__class__.__name__)
        return self.build(feed_vars, 'test')
