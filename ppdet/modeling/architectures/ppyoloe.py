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
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['PPYOLOE', 'PPYOLOEWithAuxHead']
# PP-YOLOE and PP-YOLOE+ are recommended to use this architecture, especially when use distillation or aux head
# PP-YOLOE and PP-YOLOE+ can also use the same architecture of YOLOv3 in yolo.py when not use distillation or aux head


@register
class PPYOLOE(BaseArch):
    """
    PPYOLOE network, see https://arxiv.org/abs/2203.16250

    Args:
        backbone (nn.Layer): backbone instance
        neck (nn.Layer): neck instance
        yolo_head (nn.Layer): anchor_head instance
        post_process (object): `BBoxPostProcess` instance
        ssod_loss (object): 'SSODPPYOLOELoss' instance, only used for semi-det(ssod)
        for_distill (bool): whether for distillation
        feat_distill_place (str): distill which feature for distillation
        for_mot (bool): whether return other features for multi-object tracking
            models, default False in pure object detection models.
    """

    __category__ = 'architecture'
    __shared__ = ['for_distill']
    __inject__ = ['post_process', 'ssod_loss']

    def __init__(self,
                 backbone='CSPResNet',
                 neck='CustomCSPPAN',
                 yolo_head='PPYOLOEHead',
                 post_process='BBoxPostProcess',
                 ssod_loss='SSODPPYOLOELoss',
                 for_distill=False,
                 feat_distill_place='neck_feats',
                 with_mask=False,
                 for_mot=False):
        super(PPYOLOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.with_mask = with_mask

        # for ssod, semi-det
        self.is_teacher = False
        self.ssod_loss = ssod_loss

        # distill
        self.for_distill = for_distill
        self.feat_distill_place = feat_distill_place
        if for_distill:
            assert feat_distill_place in ['backbone_feats', 'neck_feats']

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

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

            yolo_head_outs = self.yolo_head(neck_feats)

            if self.post_process is not None:
                bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])

            else:
                if not self.with_mask:
                    bbox, bbox_num, nms_keep_idx = self.yolo_head.post_process(
                        yolo_head_outs, self.inputs['scale_factor'])
                else:
                    bbox, bbox_num, mask, nms_keep_idx = self.yolo_head.post_process(
                        yolo_head_outs,
                        im_shape=self.inputs['im_shape'],
                        scale_factor=self.inputs['scale_factor'],
                        infer_shape=self.inputs['image'].shape[2:])

            if not self.with_mask:
                output = {'bbox': bbox, 'bbox_num': bbox_num}
            else:
                output = {'bbox': bbox, 'bbox_num': bbox_num, 'mask': mask}

            if self.with_mask:
                output['mask'] = mask

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_iou', 'loss_dfl', 'loss_contrast']

    def get_ssod_loss(self, student_head_outs, teacher_head_outs, train_cfg):
        ssod_losses = self.ssod_loss(student_head_outs, teacher_head_outs,
                                     train_cfg)
        return ssod_losses


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
            yolo_head_outs = self.yolo_head(neck_feats)
            if self.post_process is not None:
                bbox, bbox_num = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])
            else:
                bbox, bbox_num = self.yolo_head.post_process(
                    yolo_head_outs, self.inputs['scale_factor'])
            output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()