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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, create
from ppdet.modeling import ops

from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import bbox2delta, delta2bbox, clip_bbox, nonempty_bbox


@register
class CascadeTwoFCHead(nn.Layer):
    __shared__ = ['num_cascade_stage']

    def __init__(self,
                 in_dim=256,
                 mlp_dim=1024,
                 resolution=7,
                 num_cascade_stage=3):
        super(CascadeTwoFCHead, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        fan = in_dim * resolution * resolution

        self.fc6_list = []
        self.fc7_list = []
        for i in range(num_cascade_stage):
            fc6_name = 'fc6_{}'.format(i)
            fc7_name = 'fc7_{}'.format(i)
            fc6 = self.add_sublayer(
                fc6_name,
                nn.Linear(
                    in_dim * resolution * resolution,
                    mlp_dim,
                    weight_attr=paddle.ParamAttr(
                        initializer=XavierUniform(fan_out=fan))))

            fc7 = self.add_sublayer(
                fc7_name,
                nn.Linear(
                    mlp_dim,
                    mlp_dim,
                    weight_attr=paddle.ParamAttr(initializer=XavierUniform())))
            self.fc6_list.append(fc6)
            self.fc7_list.append(fc7)

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_dim': s.channels}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.mlp_dim, )]

    def forward(self, rois_feat, stage=0):
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6_list[stage](rois_feat)
        fc6 = F.relu(fc6)
        fc7 = self.fc7_list[stage](fc6)
        fc7 = F.relu(fc7)
        return fc7


@register
class CascadeHead(nn.Layer):
    __shared__ = ['num_classes', 'num_cascade_stages']
    __inject__ = ['bbox_assigner']
    """
    head (nn.Layer): Extract feature in bbox head
    in_channel (int): Input channel after RoI extractor
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 num_classes=80,
                 bbox_weight=[[10., 10., 5., 5.], [20.0, 20.0, 10.0, 10.0],
                              [30.0, 30.0, 15.0, 15.0]],
                 num_cascade_stages=3):
        super(CascadeHead, self).__init__()
        self.head = head
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner

        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.num_cascade_stages = num_cascade_stages

        self.bbox_score_list = []
        self.bbox_delta_list = []
        for i in range(num_cascade_stages):
            score_name = 'bbox_score_stage{}'.format(i)
            delta_name = 'bbox_delta_stage{}'.format(i)
            bbox_score = self.add_sublayer(
                score_name,
                nn.Linear(
                    in_channel,
                    self.num_classes + 1,
                    weight_attr=paddle.ParamAttr(initializer=Normal(
                        mean=0.0, std=0.01))))

            bbox_delta = self.add_sublayer(
                delta_name,
                nn.Linear(
                    in_channel,
                    4,
                    weight_attr=paddle.ParamAttr(initializer=Normal(
                        mean=0.0, std=0.001))))
            self.bbox_score_list.append(bbox_score)
            self.bbox_delta_list.append(bbox_delta)
        self.assigned_label = None
        self.assigned_rois = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
            'in_channel': head.out_shape[0].channels
        }

    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (Tensor): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)
            targets_list = [targets]
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        pred_bbox = None
        head_out_list = []
        for i in range(self.num_cascade_stages):
            if i > 0:
                rois, rois_num = self._get_rois_from_boxes(pred_bbox,
                                                           inputs['im_shape'])
                if self.training:
                    rois, rois_num, targets = self.bbox_assigner(
                        rois, rois_num, inputs, i, is_cascade=True)
                    targets_list.append(targets)

            rois_feat = self.roi_extractor(body_feats, rois, rois_num)
            bbox_feat = self.head(rois_feat, i)
            scores = self.bbox_score_list[i](bbox_feat)
            deltas = self.bbox_delta_list[i](bbox_feat)
            head_out_list.append([scores, deltas, rois])
            pred_bbox = self._get_pred_bbox(deltas, rois, self.bbox_weight[i])

        if self.training:
            loss = {}
            for stage, (
                (scores, deltas, rois),
                    targets) in enumerate(zip(head_out_list, targets_list)):
                loss_stage = self.get_loss(scores, deltas, targets, rois,
                                           self.bbox_weight[stage])
                loss.update({
                    k + "_stage{}".format(stage): v
                    for k, v in loss_stage.items()
                })
            return loss, bbox_feat
        else:
            scores, deltas, self.refined_rois = self.get_prediction(
                head_out_list)
            return (deltas, scores), self.head

    def _get_rois_from_boxes(self, boxes, im_shape):
        rois = []
        for i, boxes_per_image in enumerate(boxes):
            clip_box = clip_bbox(boxes_per_image, im_shape[i])
            if self.training:
                keep = nonempty_bbox(clip_box)
                clip_box = paddle.gather(clip_box, keep)
            rois.append(clip_box)
        rois_num = paddle.concat([paddle.shape(r)[0] for r in rois])
        return rois, rois_num

    def _get_pred_bbox(self, deltas, proposals, weights):
        pred_proposals = paddle.concat(proposals) if len(
            proposals) > 1 else proposals[0]
        pred_bbox = delta2bbox(deltas, pred_proposals, weights)
        num_prop = [p.shape[0] for p in proposals]
        return pred_bbox.split(num_prop)

    def get_loss(self, scores, deltas, targets, rois, bbox_weight):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        # TODO: better pass args
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets
        tgt_labels = paddle.concat(tgt_labels) if len(
            tgt_labels) > 1 else tgt_labels[0]
        tgt_labels = tgt_labels.cast('int64')
        tgt_labels.stop_gradient = True
        loss_bbox_cls = F.cross_entropy(
            input=scores, label=tgt_labels, reduction='mean')

        fg_inds = paddle.nonzero(
            paddle.logical_and(tgt_labels >= 0, tgt_labels <
                               self.num_classes)).flatten()

        reg_delta = paddle.gather(deltas, fg_inds)
        rois = paddle.concat(rois) if len(rois) > 1 else rois[0]
        tgt_bboxes = paddle.concat(tgt_bboxes) if len(
            tgt_bboxes) > 1 else tgt_bboxes[0]

        reg_target = bbox2delta(rois, tgt_bboxes, bbox_weight)
        reg_target = paddle.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True

        loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum(
        ) / tgt_labels.shape[0]

        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}
        loss_bbox[cls_name] = loss_bbox_cls / self.num_cascade_stages
        loss_bbox[reg_name] = loss_bbox_reg / self.num_cascade_stages

        return loss_bbox

    def get_prediction(self, head_out_list):
        """
        head_out_list(List[Tensor]): scores, deltas, rois
        """
        pred_list = []
        scores_list = [F.softmax(head[0]) for head in head_out_list]
        scores = paddle.add_n(scores_list) / self.num_cascade_stages
        # Get deltas and rois from the last stage
        _, deltas, rois = head_out_list[-1]
        return scores, deltas, rois

    def get_head(self, ):
        return self.head

    def get_assigned_targets(self, ):
        return self.assigned_targets

    def get_assigned_rois(self, ):
        return self.assigned_rois

    def get_refined_rois(self, ):
        return self.refined_rois
