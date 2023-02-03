# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..ssod_utils import QFLv2
from ..losses import GIoULoss

__all__ = ['PPYOLOE', 'PPYOLOEWithAuxHead']
# PP-YOLOE and PP-YOLOE+ are recommended to use this architecture, especially when use distillation or aux head
# PP-YOLOE and PP-YOLOE+ can also use the same architecture of YOLOv3 in yolo.py when not use distillation or aux head


@register
class PPYOLOE(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['for_distill']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='CSPResNet',
                 neck='CustomCSPPAN',
                 yolo_head='PPYOLOEHead',
                 post_process='BBoxPostProcess',
                 for_distill=False,
                 feat_distill_place='neck_feats',
                 for_mot=False):
        """
        PPYOLOE network, see https://arxiv.org/abs/2203.16250

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            post_process (object): `BBoxPostProcess` instance
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(PPYOLOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot

        # semi-det
        self.is_teacher = False
        self.queue_ptr = 0
        self.queue_size = 100 * 672
        self.queue_feats = paddle.zeros([self.queue_size, 80])
        self.queue_probs = paddle.zeros([self.queue_size, 80])
        self.iter = 0

        # distill
        self.for_distill = for_distill
        self.feat_distill_place = feat_distill_place
        if for_distill:
            assert feat_distill_place in ['backbone_feats', 'neck_feats']

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        self.is_teacher = self.inputs.get('is_teacher', False)  # for semi-det
        if self.training or self.is_teacher:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)

            if self.for_distill:
                if self.feat_distill_place == 'backbone_feats':
                    self.yolo_head.distill_pairs['backbone_feats'] = body_feats
                elif self.feat_distill_place == 'neck_feats':
                    self.yolo_head.distill_pairs['neck_feats'] = neck_feats
                else:
                    raise ValueError
            return yolo_losses
        else:
            cam_data = {}  # record bbox scores and index before nms
            yolo_head_outs = self.yolo_head(neck_feats)
            cam_data['scores'] = yolo_head_outs[0]

            if self.post_process is not None:
                bbox, bbox_num, before_nms_indexes = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])
                cam_data['before_nms_indexes'] = before_nms_indexes
            else:
                bbox, bbox_num, before_nms_indexes = self.yolo_head.post_process(
                    yolo_head_outs, self.inputs['scale_factor'])
                # data for cam
                cam_data['before_nms_indexes'] = before_nms_indexes
            output = {'bbox': bbox, 'bbox_num': bbox_num, 'cam_data': cam_data}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_iou', 'loss_dfl', 'loss_contrast']

    def get_distill_loss(self, student_head_outs, teacher_head_outs,
                         ratio=0.01):
        # for semi-det distill
        # student_probs: already sigmoid
        student_probs, student_deltas, student_dfl = student_head_outs
        teacher_probs, teacher_deltas, teacher_dfl = teacher_head_outs
        nc = student_probs.shape[-1]
        student_probs = student_probs.reshape([-1, nc])
        teacher_probs = teacher_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_deltas = teacher_deltas.reshape([-1, 4])
        student_dfl = student_dfl.reshape([-1, 4, self.yolo_head.reg_channels])
        teacher_dfl = teacher_dfl.reshape([-1, 4, self.yolo_head.reg_channels])

        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_probs.shape[0] * ratio)
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals,
                                                   teacher_probs.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

            # for contrast loss
            probs = teacher_probs[b_mask].detach()
            temperature = 0.2
            alpha = 0.9
            if self.iter > 100:
                A = paddle.exp(
                    paddle.mm(teacher_probs[b_mask], self.queue_probs.t()) /
                    temperature)
                A = A / A.sum(1, keepdim=True)
                probs = alpha * probs + (1 - alpha) * paddle.mm(
                    A, self.queue_probs)
            n = student_probs[b_mask].shape[0]
        sim = paddle.exp(
            paddle.mm(student_probs[b_mask], teacher_probs[b_mask].t()) / 0.2)
        sim_probs = sim / sim.sum(1, keepdim=True)
        Q = paddle.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= 0.5).astype('float32')
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        self.queue_feats[self.queue_ptr:self.queue_ptr + n, :] = teacher_probs[
            b_mask].detach()
        self.queue_probs[self.queue_ptr:self.queue_ptr + n, :] = teacher_probs[
            b_mask].detach()
        self.queue_ptr = (self.queue_ptr + n) % self.queue_size
        self.iter += 1
        loss_contrast = -(paddle.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()

        # loss_cls
        loss_cls = QFLv2(
            student_probs, teacher_probs, weight=mask, reduction="sum") / fg_num

        # loss_iou
        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            -1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            -1)
        iou_loss = GIoULoss(reduction='mean')
        loss_iou = iou_loss(inputs, targets)

        # loss_dfl
        loss_dfl = F.cross_entropy(
            student_dfl[b_mask].reshape([-1, self.yolo_head.reg_channels]),
            teacher_dfl[b_mask].reshape([-1, self.yolo_head.reg_channels]),
            soft_label=True,
            reduction='mean')

        return {
            "distill_loss_cls": loss_cls,
            "distill_loss_iou": loss_iou,
            "distill_loss_dfl": loss_dfl,
            "distill_loss_contrast": loss_contrast,
            "fg_sum": fg_num,
        }


@register
class PPYOLOEWithAuxHead(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='CSPResNet',
                 neck='CustomCSPPAN',
                 yolo_head='PPYOLOEHead',
                 aux_head='SimpleConvHead',
                 post_process='BBoxPostProcess',
                 for_mot=False,
                 detach_epoch=5):
        """
        PPYOLOE network, see https://arxiv.org/abs/2203.16250

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            post_process (object): `BBoxPostProcess` instance
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(PPYOLOEWithAuxHead, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.aux_neck = copy.deepcopy(self.neck)

        self.yolo_head = yolo_head
        self.aux_head = aux_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.detach_epoch = detach_epoch

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)
        aux_neck = copy.deepcopy(neck)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)
        aux_head = create(cfg['aux_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
            'aux_head': aux_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            if self.inputs['epoch_id'] >= self.detach_epoch:
                aux_neck_feats = self.aux_neck([f.detach() for f in body_feats])
                dual_neck_feats = (paddle.concat(
                    [f.detach(), aux_f], axis=1) for f, aux_f in
                                   zip(neck_feats, aux_neck_feats))
            else:
                aux_neck_feats = self.aux_neck(body_feats)
                dual_neck_feats = (paddle.concat(
                    [f, aux_f], axis=1) for f, aux_f in
                                   zip(neck_feats, aux_neck_feats))
            aux_cls_scores, aux_bbox_preds = self.aux_head(dual_neck_feats)
            loss = self.yolo_head(
                neck_feats,
                self.inputs,
                aux_pred=[aux_cls_scores, aux_bbox_preds])
            return loss
        else:
            cam_data = {}  # record bbox scores and index before nms
            yolo_head_outs = self.yolo_head(neck_feats)
            cam_data['scores'] = yolo_head_outs[0]

            if self.post_process is not None:
                bbox, bbox_num, before_nms_indexes = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])
                cam_data['before_nms_indexes'] = before_nms_indexes
            else:
                bbox, bbox_num, before_nms_indexes = self.yolo_head.post_process(
                    yolo_head_outs, self.inputs['scale_factor'])
                # data for cam
                cam_data['before_nms_indexes'] = before_nms_indexes
            output = {'bbox': bbox, 'bbox_num': bbox_num, 'cam_data': cam_data}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
