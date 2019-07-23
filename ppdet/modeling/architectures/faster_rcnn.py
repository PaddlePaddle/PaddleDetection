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
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if mode == 'train':
            gt_box = feed_vars['gt_box']
            is_crowd = feed_vars['is_crowd']
        else:
            im_shape = feed_vars['im_shape']
        body_feats = self.backbone(im)
        body_feat_names = list(body_feats.keys())

        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        rois = self.rpn_head.get_proposals(body_feats, im_info, mode=mode)

        if mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, gt_box, is_crowd)
            # sampled rpn proposals
            for var in ['gt_label', 'is_crowd', 'gt_box', 'im_info']:
                assert var in feed_vars, "{} has no {}".format(feed_vars, var)
            outs = self.bbox_assigner(
                rpn_rois=rois,
                gt_classes=feed_vars['gt_label'],
                is_crowd=feed_vars['is_crowd'],
                gt_boxes=feed_vars['gt_box'],
                im_info=feed_vars['im_info'])

            rois = outs[0]
            labels_int32 = outs[1]
            bbox_targets = outs[2]
            bbox_inside_weights = outs[3]
            bbox_outside_weights = outs[4]
        else:
            if self.rpn_only:
                im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
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

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'test')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')
