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

import numpy as np
import sys

from collections import OrderedDict
import copy

import paddle.fluid as fluid
from ppdet.core.workspace import register
from .input_helper import multiscale_def

__all__ = ['CascadeRCNNClsAware']


@register
class CascadeRCNNClsAware(object):
    """
    Cascade R-CNN architecture, see https://arxiv.org/abs/1712.00726
    This is a kind of modification of Cascade R-CNN.
    Specifically, it predicts bboxes for all classes with different weights,
    while the standard vesion just predicts bboxes for foreground
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
        'backbone', 'fpn', 'rpn_head', 'bbox_assigner', 'roi_extractor',
        'bbox_head'
    ]

    def __init__(
            self,
            backbone,
            rpn_head,
            roi_extractor='FPNRoIAlign',
            bbox_head='CascadeBBoxHead',
            bbox_assigner='CascadeBBoxAssigner',
            fpn='FPN', ):
        super(CascadeRCNNClsAware, self).__init__()
        assert fpn is not None, "cascade RCNN requires FPN"
        self.backbone = backbone
        self.fpn = fpn
        self.rpn_head = rpn_head
        self.bbox_assigner = bbox_assigner
        self.roi_extractor = roi_extractor
        self.bbox_head = bbox_head
        self.bbox_clip = np.log(1000. / 16.)
        # Cascade local cfg
        (brw0, brw1, brw2) = self.bbox_assigner.bbox_reg_weights
        self.cascade_bbox_reg_weights = [
            [1. / brw0, 1. / brw0, 2. / brw0, 2. / brw0],
            [1. / brw1, 1. / brw1, 2. / brw1, 2. / brw1],
            [1. / brw2, 1. / brw2, 2. / brw2, 2. / brw2]
        ]
        self.cascade_rcnn_loss_weight = [1.0, 0.5, 0.25]

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if mode == 'train':
            gt_bbox = feed_vars['gt_bbox']
            is_crowd = feed_vars['is_crowd']
            gt_class = feed_vars['gt_class']
        else:
            im_shape = feed_vars['im_shape']

        # backbone
        body_feats = self.backbone(im)

        # FPN
        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # rpn proposals
        rpn_rois = self.rpn_head.get_proposals(body_feats, im_info, mode=mode)

        if mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, gt_bbox, is_crowd)

        proposal_list = []
        roi_feat_list = []
        rcnn_pred_list = []
        rcnn_target_list = []

        bbox_pred = None

        self.cascade_var_v = []
        for stage in range(3):
            var_v = np.array(
                self.cascade_bbox_reg_weights[stage], dtype="float32")
            prior_box_var = fluid.layers.create_tensor(dtype="float32")
            fluid.layers.assign(input=var_v, output=prior_box_var)
            self.cascade_var_v.append(prior_box_var)

        self.cascade_decoded_box = []
        self.cascade_cls_prob = []

        for stage in range(3):
            if stage > 0:
                pool_rois = decoded_assign_box
            else:
                pool_rois = rpn_rois
            if mode == "train":
                self.cascade_var_v[stage].stop_gradient = True
                outs = self.bbox_assigner(
                    input_rois=pool_rois, feed_vars=feed_vars, curr_stage=stage)
                pool_rois = outs[0]
                rcnn_target_list.append(outs)

            # extract roi features
            roi_feat = self.roi_extractor(body_feats, pool_rois, spatial_scale)
            roi_feat_list.append(roi_feat)

            # bbox head
            cls_score, bbox_pred = self.bbox_head.get_output(
                roi_feat,
                cls_agnostic_bbox_reg=self.bbox_head.num_classes,
                wb_scalar=1.0 / self.cascade_rcnn_loss_weight[stage],
                name='_' + str(stage + 1))

            cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)

            decoded_box, decoded_assign_box = fluid.layers.box_decoder_and_assign(
                pool_rois, self.cascade_var_v[stage], bbox_pred, cls_prob,
                self.bbox_clip)

            if mode == "train":
                decoded_box.stop_gradient = True
                decoded_assign_box.stop_gradient = True
            else:
                self.cascade_cls_prob.append(cls_prob)
                self.cascade_decoded_box.append(decoded_box)

            rcnn_pred_list.append((cls_score, bbox_pred))

        # out loop
        if mode == 'train':
            loss = self.bbox_head.get_loss(rcnn_pred_list, rcnn_target_list,
                                           self.cascade_rcnn_loss_weight)
            loss.update(rpn_loss)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.bbox_head.get_prediction_cls_aware(
                im_info, im_shape, self.cascade_cls_prob,
                self.cascade_decoded_box, self.cascade_bbox_reg_weights)
            return pred

    def build_multi_scale(self, feed_vars):
        required_fields = ['image', 'im_shape', 'im_info']
        self._input_check(required_fields, feed_vars)

        result = {}
        im_shape = feed_vars['im_shape']
        result['im_shape'] = im_shape

        for i in range(len(self.im_info_names) // 2):
            im = feed_vars[self.im_info_names[2 * i]]
            im_info = feed_vars[self.im_info_names[2 * i + 1]]

            # backbone
            body_feats = self.backbone(im)
            result.update(body_feats)
            # FPN
            if self.fpn is not None:
                body_feats, spatial_scale = self.fpn.get_output(body_feats)

            # rpn proposals
            rpn_rois = self.rpn_head.get_proposals(
                body_feats, im_info, mode="test")

            proposal_list = []
            roi_feat_list = []
            rcnn_pred_list = []
            rcnn_target_list = []

            bbox_pred = None

            self.cascade_var_v = []
            for stage in range(3):
                var_v = np.array(
                    self.cascade_bbox_reg_weights[stage], dtype="float32")
                prior_box_var = fluid.layers.create_tensor(dtype="float32")
                fluid.layers.assign(input=var_v, output=prior_box_var)
                self.cascade_var_v.append(prior_box_var)

            self.cascade_decoded_box = []
            self.cascade_cls_prob = []

            for stage in range(3):
                if stage > 0:
                    pool_rois = decoded_assign_box
                else:
                    pool_rois = rpn_rois

                # extract roi features
                roi_feat = self.roi_extractor(body_feats, pool_rois,
                                              spatial_scale)
                roi_feat_list.append(roi_feat)

                # bbox head
                cls_score, bbox_pred = self.bbox_head.get_output(
                    roi_feat,
                    cls_agnostic_bbox_reg=self.bbox_head.num_classes,
                    wb_scalar=1.0 / self.cascade_rcnn_loss_weight[stage],
                    name='_' + str(stage + 1))

                cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)

                decoded_box, decoded_assign_box = fluid.layers.box_decoder_and_assign(
                    pool_rois, self.cascade_var_v[stage], bbox_pred, cls_prob,
                    self.bbox_clip)

                self.cascade_cls_prob.append(cls_prob)
                self.cascade_decoded_box.append(decoded_box)

                rcnn_pred_list.append((cls_score, bbox_pred))

            pred = self.bbox_head.get_prediction_cls_aware(
                im_info,
                im_shape,
                self.cascade_cls_prob,
                self.cascade_decoded_box,
                self.cascade_bbox_reg_weights,
                return_box_score=True)

            bbox_name = 'bbox_' + str(i)
            score_name = 'score_' + str(i)
            if 'flip' in im.name:
                bbox_name += '_flip'
                score_name += '_flip'
            result[bbox_name] = pred['bbox']
            result[score_name] = pred['score']

        return result

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

    def build_inputs(self,
                     image_shape=[3, None, None],
                     fields=[
                         'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class',
                         'is_crowd', 'gt_mask'
                     ],
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

    def _input_check(self, require_fields, feed_vars):
        for var in require_fields:
            assert var in feed_vars, \
                "{} has no {} field".format(feed_vars, var)

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
