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

import copy
from collections import OrderedDict

import paddle.fluid as fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register
from ppdet.utils.check import check_version
from .input_helper import multiscale_def

__all__ = ['CascadeRCNN']


@register
class CascadeRCNN(object):
    """
    Cascade R-CNN architecture, see https://arxiv.org/abs/1712.00726

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

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_extractor='FPNRoIAlign',
                 bbox_head='CascadeBBoxHead',
                 bbox_assigner='CascadeBBoxAssigner',
                 rpn_only=False,
                 fpn='FPN'):
        super(CascadeRCNN, self).__init__()
        check_version('2.0.0-rc0')
        assert fpn is not None, "cascade RCNN requires FPN"
        self.backbone = backbone
        self.fpn = fpn
        self.rpn_head = rpn_head
        self.bbox_assigner = bbox_assigner
        self.roi_extractor = roi_extractor
        self.bbox_head = bbox_head
        self.rpn_only = rpn_only
        # Cascade local cfg
        self.cls_agnostic_bbox_reg = 2
        (brw0, brw1, brw2) = self.bbox_assigner.bbox_reg_weights
        self.cascade_bbox_reg_weights = [
            [1. / brw0, 1. / brw0, 2. / brw0, 2. / brw0],
            [1. / brw1, 1. / brw1, 2. / brw1, 2. / brw1],
            [1. / brw2, 1. / brw2, 2. / brw2, 2. / brw2]
        ]
        self.cascade_rcnn_loss_weight = [1.0, 0.5, 0.25]

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
        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # rpn proposals
        rpn_rois = self.rpn_head.get_proposals(body_feats, im_info, mode=mode)

        if mode == 'train':
            #fluid.layers.Print(gt_bbox)
            #fluid.layers.Print(is_crowd)
            rpn_loss = self.rpn_head.get_loss(im_info, gt_bbox, is_crowd)
        else:
            if self.rpn_only:
                im_scale = fluid.layers.slice(
                    im_info, [1], starts=[2], ends=[3])
                im_scale = fluid.layers.sequence_expand(im_scale, rpn_rois)
                rois = rpn_rois / im_scale
                return {'proposal': rois}

        proposal_list = []
        roi_feat_list = []
        rcnn_pred_list = []
        rcnn_target_list = []

        proposals = None
        bbox_pred = None
        max_overlap = None
        for i in range(3):
            if i > 0:
                refined_bbox = self._decode_box(
                    proposals,
                    bbox_pred,
                    curr_stage=i - 1, )
            else:
                refined_bbox = rpn_rois

            if mode == 'train':
                outs = self.bbox_assigner(
                    input_rois=refined_bbox,
                    feed_vars=feed_vars,
                    curr_stage=i,
                    max_overlap=max_overlap)

                proposals = outs[0]
                max_overlap = outs[-1]
                rcnn_target_list.append(outs[:-1])
            else:
                proposals = refined_bbox
            proposal_list.append(proposals)

            # extract roi features
            roi_feat = self.roi_extractor(body_feats, proposals, spatial_scale)
            roi_feat_list.append(roi_feat)

            # bbox head
            cls_score, bbox_pred = self.bbox_head.get_output(
                roi_feat,
                wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i],
                name='_' + str(i + 1) if i > 0 else '')
            rcnn_pred_list.append((cls_score, bbox_pred))

        if mode == 'train':
            loss = self.bbox_head.get_loss(rcnn_pred_list, rcnn_target_list,
                                           self.cascade_rcnn_loss_weight)
            loss.update(rpn_loss)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.bbox_head.get_prediction(
                im_info, feed_vars['im_shape'], roi_feat_list, rcnn_pred_list,
                proposal_list, self.cascade_bbox_reg_weights,
                self.cls_agnostic_bbox_reg)
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
                body_feats, im_info, mode='test')

            proposal_list = []
            roi_feat_list = []
            rcnn_pred_list = []

            proposals = None
            bbox_pred = None
            for i in range(3):
                if i > 0:
                    refined_bbox = self._decode_box(
                        proposals,
                        bbox_pred,
                        curr_stage=i - 1, )
                else:
                    refined_bbox = rpn_rois

                proposals = refined_bbox
                proposal_list.append(proposals)

                # extract roi features
                roi_feat = self.roi_extractor(body_feats, proposals,
                                              spatial_scale)
                roi_feat_list.append(roi_feat)

                # bbox head
                cls_score, bbox_pred = self.bbox_head.get_output(
                    roi_feat,
                    wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i],
                    name='_' + str(i + 1) if i > 0 else '')
                rcnn_pred_list.append((cls_score, bbox_pred))

            # get mask rois
            rois = proposal_list[2]

            if self.fpn is None:
                last_feat = body_feats[list(body_feats.keys())[-1]]
                roi_feat = self.roi_extractor(last_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

            pred = self.bbox_head.get_prediction(
                im_info,
                im_shape,
                roi_feat_list,
                rcnn_pred_list,
                proposal_list,
                self.cascade_bbox_reg_weights,
                self.cls_agnostic_bbox_reg,
                return_box_score=True)
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

    def _decode_box(self, proposals, bbox_pred, curr_stage):
        rcnn_loc_delta_r = fluid.layers.reshape(
            bbox_pred, (-1, self.cls_agnostic_bbox_reg, 4))
        # only use fg box delta to decode box
        rcnn_loc_delta_s = fluid.layers.slice(
            rcnn_loc_delta_r, axes=[1], starts=[1], ends=[2])
        refined_bbox = fluid.layers.box_coder(
            prior_box=proposals,
            prior_box_var=self.cascade_bbox_reg_weights[curr_stage],
            target_box=rcnn_loc_delta_s,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1, )
        refined_bbox = fluid.layers.reshape(refined_bbox, shape=[-1, 4])

        return refined_bbox

    def _inputs_def(self, image_shape):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_info':  {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_shape': {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_id':    {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
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
                         'is_crowd'
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
