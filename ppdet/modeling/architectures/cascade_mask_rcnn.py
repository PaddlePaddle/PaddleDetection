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

import paddle.fluid as fluid

from ppdet.core.workspace import register

__all__ = ['CascadeMaskRCNN']


@register
class CascadeMaskRCNN(object):
    """
    Cascade Mask R-CNN architecture, see https://arxiv.org/abs/1712.00726

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNhead` instance
        bbox_assigner (object): `BBoxAssigner` instance
        roi_extractor (object): ROI extractor instance
        bbox_head (object): `BBoxHead` instance
        mask_assigner (object): `MaskAssigner` instance
        mask_head (object): `MaskHead` instance
        fpn (object): feature pyramid network instance
    """

    __category__ = 'architecture'
    __inject__ = [
        'backbone', 'rpn_head', 'bbox_assigner', 'roi_extractor', 'bbox_head',
        'mask_assigner', 'mask_head', 'fpn'
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_extractor='FPNRoIAlign',
                 bbox_head='CascadeBBoxHead',
                 bbox_assigner='CascadeBBoxAssigner',
                 mask_assigner='MaskAssigner',
                 mask_head='MaskHead',
                 rpn_only=False,
                 fpn='FPN'):
        super(CascadeMaskRCNN, self).__init__()
        assert fpn is not None, "cascade RCNN requires FPN"
        self.backbone = backbone
        self.fpn = fpn
        self.rpn_head = rpn_head
        self.bbox_assigner = bbox_assigner
        self.roi_extractor = roi_extractor
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

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        assert mode in ['train', 'test'], \
            "only 'train' and 'test' mode is supported"

        if mode == 'train':
            required_fields = [
                'gt_label', 'gt_box', 'gt_mask', 'is_crowd', 'im_info'
            ]
        else:
            required_fields = ['im_shape', 'im_info']

        for var in required_fields:
            assert var in feed_vars, \
                "{} has no {} field".format(feed_vars, var)

        if mode == 'train':
            gt_box = feed_vars['gt_box']
            is_crowd = feed_vars['is_crowd']

        im_info = feed_vars['im_info']

        # backbone
        body_feats = self.backbone(im)

        # FPN
        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # rpn proposals
        rpn_rois = self.rpn_head.get_proposals(body_feats, im_info, mode=mode)

        if mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, gt_box, is_crowd)
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
                    input_rois=refined_bbox, feed_vars=feed_vars, curr_stage=i)

                proposals = outs[0]
                rcnn_target_list.append(outs)
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

        # get mask rois
        rois = proposal_list[2]

        if mode == 'train':
            loss = self.bbox_head.get_loss(rcnn_pred_list, rcnn_target_list,
                                           self.cascade_rcnn_loss_weight)
            loss.update(rpn_loss)

            labels_int32 = rcnn_target_list[2][1]

            mask_rois, roi_has_mask_int32, mask_int32 = self.mask_assigner(
                rois=rois,
                gt_classes=feed_vars['gt_label'],
                is_crowd=feed_vars['is_crowd'],
                gt_segms=feed_vars['gt_mask'],
                im_info=feed_vars['im_info'],
                labels_int32=labels_int32)

            if self.fpn is None:
                bbox_head_feat = self.bbox_head.get_head_feat()
                feat = fluid.layers.gather(bbox_head_feat, roi_has_mask_int32)
            else:
                feat = self.roi_extractor(
                    body_feats, mask_rois, spatial_scale, is_mask=True)
            mask_loss = self.mask_head.get_loss(feat, mask_int32)
            loss.update(mask_loss)

            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            if self.fpn is None:
                last_feat = body_feats[list(body_feats.keys())[-1]]
                roi_feat = self.roi_extractor(last_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

            bbox_pred = self.bbox_head.get_prediction(
                im_info, feed_vars['im_shape'], roi_feat_list, rcnn_pred_list,
                proposal_list, self.cascade_bbox_reg_weights,
                self.cls_agnostic_bbox_reg)

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
                name='mask_pred')

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(input=bbox_pred, output=mask_pred)
                with switch.default():
                    bbox = fluid.layers.slice(
                        bbox_pred, [1], starts=[2], ends=[6])

                    im_scale = fluid.layers.slice(
                        im_info, [1], starts=[2], ends=[3])
                    im_scale = fluid.layers.sequence_expand(im_scale, bbox)

                    mask_rois = bbox * im_scale
                    if self.fpn is None:
                        mask_feat = self.roi_extractor(last_feat, mask_rois)
                        mask_feat = self.bbox_head.get_head_feat(mask_feat)
                    else:
                        mask_feat = self.roi_extractor(
                            body_feats, mask_rois, spatial_scale, is_mask=True)

                    mask_out = self.mask_head.get_prediction(mask_feat, bbox)
                    fluid.layers.assign(input=mask_out, output=mask_pred)
            return {'bbox': bbox_pred, 'mask': mask_pred}

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

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'test')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')
