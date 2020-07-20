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
import copy
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

from .input_helper import multiscale_def

__all__ = ['HybridTaskCascade']


@register
class HybridTaskCascade(object):
    """
    Hybrid Task Cascade  Mask R-CNN architecture, see https://arxiv.org/abs/1901.07518

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNhead` instance
        bbox_assigner (object): `BBoxAssigner` instance
        roi_extractor (object): ROI extractor instance
        bbox_head (object): `HTCBBoxHead` instance
        mask_assigner (object): `MaskAssigner` instance
        mask_head (object): `HTCMaskHead` instance
        fpn (object): feature pyramid network instance
        semantic_roi_extractor(object): ROI extractor instance 
        fused_semantic_head (object): `FusedSemanticHead` instance 
    """

    __category__ = 'architecture'
    __inject__ = [
        'backbone', 'rpn_head', 'bbox_assigner', 'roi_extractor', 'bbox_head',
        'mask_assigner', 'mask_head', 'fpn', 'semantic_roi_extractor',
        'fused_semantic_head'
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_extractor='FPNRoIAlign',
                 semantic_roi_extractor='RoIAlign',
                 fused_semantic_head='FusedSemanticHead',
                 bbox_head='HTCBBoxHead',
                 bbox_assigner='CascadeBBoxAssigner',
                 mask_assigner='MaskAssigner',
                 mask_head='HTCMaskHead',
                 rpn_only=False,
                 fpn='FPN'):
        super(HybridTaskCascade, self).__init__()
        assert fpn is not None, "HTC requires FPN"
        self.backbone = backbone
        self.fpn = fpn
        self.rpn_head = rpn_head
        self.bbox_assigner = bbox_assigner
        self.roi_extractor = roi_extractor
        self.semantic_roi_extractor = semantic_roi_extractor
        self.fused_semantic_head = fused_semantic_head
        self.bbox_head = bbox_head
        self.mask_assigner = mask_assigner
        self.mask_head = mask_head
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
        self.num_stage = 3
        self.with_mask = True
        self.interleaved = True
        self.mask_info_flow = True
        self.with_semantic = True
        self.use_bias_scalar = True

    def build(self, feed_vars, mode='train'):
        if mode == 'train':
            required_fields = [
                'gt_class', 'gt_bbox', 'gt_mask', 'is_crowd', 'im_info',
                'semantic'
            ]
        else:
            required_fields = ['im_shape', 'im_info']
        self._input_check(required_fields, feed_vars)

        im = feed_vars['image']
        if mode == 'train':
            gt_bbox = feed_vars['gt_bbox']
            is_crowd = feed_vars['is_crowd']

        im_info = feed_vars['im_info']

        # backbone
        body_feats = self.backbone(im)

        loss = {}
        # FPN
        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        if self.with_semantic:
            # TODO: use cfg
            semantic_feat, seg_pred = self.fused_semantic_head.get_out(
                body_feats)
            if mode == 'train':
                s_label = feed_vars['semantic']
                semantic_loss = self.fused_semantic_head.get_loss(seg_pred,
                                                                  s_label) * 0.2
                loss.update({"semantic_loss": semantic_loss})
        else:
            semantic_feat = None

        # rpn proposals
        rpn_rois = self.rpn_head.get_proposals(body_feats, im_info, mode=mode)
        if mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, gt_bbox, is_crowd)
            loss.update(rpn_loss)
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
        mask_logits_list = []
        mask_target_list = []
        proposals = None
        bbox_pred = None
        outs = None
        refined_bbox = rpn_rois
        for i in range(self.num_stage):
            # BBox Branch
            if mode == 'train':
                outs = self.bbox_assigner(
                    input_rois=refined_bbox, feed_vars=feed_vars, curr_stage=i)
                proposals = outs[0]
                rcnn_target_list.append(outs)
            else:
                proposals = refined_bbox
            proposal_list.append(proposals)

            # extract roi features
            roi_feat = self.roi_extractor(body_feats, proposals, spatial_scale)
            if self.with_semantic:
                semantic_roi_feat = self.semantic_roi_extractor(semantic_feat,
                                                                proposals)
                if semantic_roi_feat is not None:
                    semantic_roi_feat = fluid.layers.pool2d(
                        semantic_roi_feat,
                        pool_size=2,
                        pool_stride=2,
                        pool_padding='SAME')
                    roi_feat = fluid.layers.sum([roi_feat, semantic_roi_feat])
            roi_feat_list.append(roi_feat)

            # bbox head
            cls_score, bbox_pred = self.bbox_head.get_output(
                roi_feat,
                wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i],
                name='_' + str(i))
            rcnn_pred_list.append((cls_score, bbox_pred))

            # Mask Branch 
            if self.with_mask:
                if mode == 'train':
                    labels_int32 = outs[1]
                    if self.interleaved:
                        refined_bbox = self._decode_box(
                            proposals, bbox_pred, curr_stage=i)
                        proposals = refined_bbox

                    mask_rois, roi_has_mask_int32, mask_int32 = self.mask_assigner(
                        rois=proposals,
                        gt_classes=feed_vars['gt_class'],
                        is_crowd=feed_vars['is_crowd'],
                        gt_segms=feed_vars['gt_mask'],
                        im_info=feed_vars['im_info'],
                        labels_int32=labels_int32)
                    mask_target_list.append(mask_int32)

                    mask_feat = self.roi_extractor(
                        body_feats, mask_rois, spatial_scale, is_mask=True)

                    if self.with_semantic:
                        semantic_roi_feat = self.semantic_roi_extractor(
                            semantic_feat, mask_rois)
                        if semantic_roi_feat is not None:
                            mask_feat = fluid.layers.sum(
                                [mask_feat, semantic_roi_feat])

                    if self.mask_info_flow:
                        last_feat = None
                        for j in range(i):
                            last_feat = self.mask_head.get_output(
                                mask_feat,
                                last_feat,
                                return_logits=False,
                                return_feat=True,
                                wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i]
                                if self.use_bias_scalar else 1.0,
                                name='_' + str(i) + '_' + str(j))
                        mask_logits = self.mask_head.get_output(
                            mask_feat,
                            last_feat,
                            return_logits=True,
                            return_feat=False,
                            wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i]
                            if self.use_bias_scalar else 1.0,
                            name='_' + str(i))
                    else:
                        mask_logits = self.mask_head.get_output(
                            mask_feat,
                            return_logits=True,
                            wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i]
                            if self.use_bias_scalar else 1.0,
                            name='_' + str(i))
                    mask_logits_list.append(mask_logits)

            if i < self.num_stage - 1 and not self.interleaved:
                refined_bbox = self._decode_box(
                    proposals, bbox_pred, curr_stage=i)
            elif i < self.num_stage - 1 and mode != 'train':
                refined_bbox = self._decode_box(
                    proposals, bbox_pred, curr_stage=i)

        if mode == 'train':
            bbox_loss = self.bbox_head.get_loss(
                rcnn_pred_list, rcnn_target_list, self.cascade_rcnn_loss_weight)
            loss.update(bbox_loss)
            mask_loss = self.mask_head.get_loss(mask_logits_list,
                                                mask_target_list,
                                                self.cascade_rcnn_loss_weight)
            loss.update(mask_loss)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            mask_name = 'mask_pred'
            mask_pred, bbox_pred = self.single_scale_eval(
                body_feats,
                spatial_scale,
                im_info,
                mask_name,
                bbox_pred,
                roi_feat_list,
                rcnn_pred_list,
                proposal_list,
                feed_vars['im_shape'],
                semantic_feat=semantic_feat if self.with_semantic else None)
            return {'bbox': bbox_pred, 'mask': mask_pred}

    def single_scale_eval(self,
                          body_feats,
                          spatial_scale,
                          im_info,
                          mask_name,
                          bbox_pred,
                          roi_feat_list=None,
                          rcnn_pred_list=None,
                          proposal_list=None,
                          im_shape=None,
                          use_multi_test=False,
                          semantic_feat=None):

        if not use_multi_test:
            bbox_pred = self.bbox_head.get_prediction(
                im_info, im_shape, roi_feat_list, rcnn_pred_list, proposal_list,
                self.cascade_bbox_reg_weights)
            bbox_pred = bbox_pred['bbox']

        # share weight
        bbox_shape = fluid.layers.shape(bbox_pred)
        bbox_size = fluid.layers.reduce_prod(bbox_shape)
        bbox_size = fluid.layers.reshape(bbox_size, [1, 1])
        size = fluid.layers.fill_constant([1, 1], value=6, dtype='int32')
        cond = fluid.layers.less_than(x=bbox_size, y=size)

        mask_pred = fluid.layers.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=False,
            name=mask_name)

        def noop():
            fluid.layers.assign(input=bbox_pred, output=mask_pred)

        def process_boxes():
            bbox = fluid.layers.slice(bbox_pred, [1], starts=[2], ends=[6])

            im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
            im_scale = fluid.layers.sequence_expand(im_scale, bbox)

            bbox = fluid.layers.cast(bbox, dtype='float32')
            im_scale = fluid.layers.cast(im_scale, dtype='float32')
            mask_rois = bbox * im_scale

            mask_feat = self.roi_extractor(
                body_feats, mask_rois, spatial_scale, is_mask=True)

            if self.with_semantic:
                semantic_roi_feat = self.semantic_roi_extractor(semantic_feat,
                                                                mask_rois)
                if semantic_roi_feat is not None:
                    mask_feat = fluid.layers.sum([mask_feat, semantic_roi_feat])

            mask_logits_list = []
            mask_pred_list = []
            for i in range(self.num_stage):
                if self.mask_info_flow:
                    last_feat = None
                    for j in range(i):
                        last_feat = self.mask_head.get_output(
                            mask_feat,
                            last_feat,
                            return_logits=False,
                            return_feat=True,
                            wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i]
                            if self.use_bias_scalar else 1.0,
                            name='_' + str(i) + '_' + str(j))
                    mask_logits = self.mask_head.get_output(
                        mask_feat,
                        last_feat,
                        return_logits=True,
                        return_feat=False,
                        wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i]
                        if self.use_bias_scalar else 1.0,
                        name='_' + str(i))
                    mask_logits_list.append(mask_logits)
                else:
                    mask_logits = self.mask_head.get_output(
                        mask_feat,
                        return_logits=True,
                        return_feat=False,
                        name='_' + str(i))
                mask_pred_out = self.mask_head.get_prediction(mask_logits, bbox)
                mask_pred_list.append(mask_pred_out)

            mask_pred_out = fluid.layers.sum(mask_pred_list) / float(
                len(mask_pred_list))
            fluid.layers.assign(input=mask_pred_out, output=mask_pred)

        fluid.layers.cond(cond, noop, process_boxes)
        return mask_pred, bbox_pred

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
            'im_id':    {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'im_shape': {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'gt_bbox':  {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'is_crowd': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'gt_mask':  {'shape': [None, 2], 'dtype': 'float32', 'lod_level': 3}, # polygon coordinates
            'semantic': {'shape': [None, 1, None, None], 'dtype': 'int32', 'lod_level': 0},
        }
        # yapf: enable
        return inputs_def

    def build_inputs(self,
                     image_shape=[3, None, None],
                     fields=[
                         'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class',
                         'is_crowd', 'gt_mask', 'semantic'
                     ],
                     multi_scale=False,
                     num_scales=-1,
                     use_flip=None,
                     use_dataloader=True,
                     iterable=False,
                     mask_branch=False):
        inputs_def = self._inputs_def(image_shape)
        fields = copy.deepcopy(fields)
        if multi_scale:
            ms_def, ms_fields = multiscale_def(image_shape, num_scales,
                                               use_flip)
            inputs_def.update(ms_def)
            fields += ms_fields
            self.im_info_names = ['image', 'im_info'] + ms_fields
            if mask_branch:
                box_fields = ['bbox', 'bbox_flip'] if use_flip else ['bbox']
                for key in box_fields:
                    inputs_def[key] = {
                        'shape': [6],
                        'dtype': 'float32',
                        'lod_level': 1
                    }
                fields += box_fields
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        use_dataloader = use_dataloader and not mask_branch
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=64,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars, multi_scale=None, mask_branch=False):
        if multi_scale:
            return self.build_multi_scale(feed_vars, mask_branch)
        return self.build(feed_vars, 'test')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')
