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

from paddle import fluid
from ppdet.core.workspace import register

__all__ = ['MaskRCNN']


@register
class MaskRCNN(object):
    """
    Mask R-CNN architecture, see https://arxiv.org/abs/1703.06870
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
                 bbox_head='BBoxHead',
                 bbox_assigner='BBoxAssigner',
                 roi_extractor='RoIAlign',
                 mask_assigner='MaskAssigner',
                 mask_head='MaskHead',
                 rpn_only=False,
                 fpn=None):
        super(MaskRCNN, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_assigner = bbox_assigner
        self.roi_extractor = roi_extractor
        self.bbox_head = bbox_head
        self.mask_assigner = mask_assigner
        self.mask_head = mask_head
        self.rpn_only = rpn_only
        self.fpn = fpn

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
        im_info = feed_vars['im_info']

        body_feats = self.backbone(im)

        # FPN
        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # RPN proposals
        rois = self.rpn_head.get_proposals(body_feats, im_info, mode=mode)

        if mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, feed_vars['gt_box'],
                                              feed_vars['is_crowd'])

            outs = self.bbox_assigner(
                rpn_rois=rois,
                gt_classes=feed_vars['gt_label'],
                is_crowd=feed_vars['is_crowd'],
                gt_boxes=feed_vars['gt_box'],
                im_info=feed_vars['im_info'])
            rois = outs[0]
            labels_int32 = outs[1]

            if self.fpn is None:
                last_feat = body_feats[list(body_feats.keys())[-1]]
                roi_feat = self.roi_extractor(last_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

            loss = self.bbox_head.get_loss(roi_feat, labels_int32, *outs[2:])
            loss.update(rpn_loss)

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
            if self.rpn_only:
                im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
                im_scale = fluid.layers.sequence_expand(im_scale, rois)
                rois = rois / im_scale
                return {'proposal': rois}
            if self.fpn is None:
                last_feat = body_feats[list(body_feats.keys())[-1]]
                roi_feat = self.roi_extractor(last_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

            bbox_pred = self.bbox_head.get_prediction(roi_feat, rois, im_info,
                                                      feed_vars['im_shape'])
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

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'test')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')
