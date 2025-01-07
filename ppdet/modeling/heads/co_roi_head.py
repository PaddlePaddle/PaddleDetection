# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
this code is base on https://github.com/Sense-X/Co-DETR/blob/main/projects/models/co_roi_head.py
"""
import paddle
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.heads.bbox_head import BBoxHead
from .roi_extractor import RoIAlign
from ..cls_utils import _get_class_default_kwargs

__all__ = ['Co_RoiHead']


@register
class Co_RoiHead(BBoxHead):
    __shared__ = ['num_classes', 'use_cot']
    __inject__ = ['bbox_assigner', 'bbox_loss', 'loss_cot']
    """
    RCNN bbox head

    Args:
        head (nn.Layer): Extract feature in bbox head
        in_channel (int): Input channel after RoI extractor
        roi_extractor (object): The module of RoI Extractor
        bbox_assigner (object): The module of Box Assigner, label and sample the 
            box.
        with_pool (bool): Whether to use pooling for the RoI feature.
        num_classes (int): The number of classes
        bbox_weight (List[float]): The weight to get the decode box
        cot_classes (int): The number of base classes
        loss_cot (object): The module of Label-cotuning
        use_cot(bool): whether to use Label-cotuning 
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=_get_class_default_kwargs(RoIAlign),
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.],
                 bbox_loss=None,
                 cls_loss_weight=1.,
                 loss_normalize_pos=False,
                 cot_classes=None,
                 loss_cot='COTLoss',
                 use_cot=False):
        super(Co_RoiHead, self).__init__(
            head=head,
            in_channel=in_channel,
            roi_extractor=roi_extractor,
            bbox_assigner=bbox_assigner,
            with_pool=with_pool,
            num_classes=num_classes,
            bbox_weight=bbox_weight,
            bbox_loss =bbox_loss,
            loss_normalize_pos=loss_normalize_pos,
            cot_classes=cot_classes,
            loss_cot=loss_cot,
            use_cot=use_cot
            )
        self.head=head
        self.cls_loss_weight = cls_loss_weight
    
    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None, cot=False):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        if self.with_pool:
            feat = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            feat = paddle.squeeze(feat, axis=[2, 3])
        else:
            feat = bbox_feat
        if self.use_cot:
            scores = self.cot_bbox_score(feat)
            cot_scores = self.bbox_score(feat)
        else:
            scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        if self.training:
            loss = self.get_loss(
                scores,
                deltas,
                targets,
                rois,
                self.bbox_weight,
                loss_normalize_pos=self.loss_normalize_pos)
            loss['loss_bbox_cls'] *= self.cls_loss_weight

            if self.cot_relation is not None:
                loss_cot = self.loss_cot(cot_scores, targets, self.cot_relation)
                loss.update(loss_cot)
                
            target_labels, target_bboxs, _ = targets
            # get pos_coords
            ori_proposals, ori_labels, ori_bbox_targets, ori_bbox_feats = [], [], [], []
            roi_start_idx = 0
            for i in range(len(rois)):
                ori_proposal = rois[i].unsqueeze(0)
                ori_label = target_labels[i].unsqueeze(0)
                ori_bbox_target = target_bboxs[i].unsqueeze(0)
                ori_bbox_feat = rois_feat[roi_start_idx:roi_start_idx+rois_num[i]].mean(-1).mean(-1)
                ori_bbox_feat = ori_bbox_feat.unsqueeze(0)

                ori_proposals.append(ori_proposal) 
                ori_labels.append(ori_label)
                ori_bbox_targets.append(ori_bbox_target)
                ori_bbox_feats.append(ori_bbox_feat)

                roi_start_idx += rois_num[i]
                
            ori_coords = paddle.concat(ori_proposals, axis=0)
            ori_labels = paddle.concat(ori_labels, axis=0)
            ori_bbox_targets = paddle.concat(ori_bbox_targets, axis=0)
            ori_bbox_feats = paddle.concat(ori_bbox_feats, axis=0)
            pos_coords = (ori_coords, ori_labels, ori_bbox_targets, ori_bbox_feats, 'rcnn')
            loss.update(pos_coords=pos_coords)
            return loss, bbox_feat
        else:
            if cot:
                pred = self.get_prediction(cot_scores, deltas)
            else:
                pred = self.get_prediction(scores, deltas)
            return pred, self.head
