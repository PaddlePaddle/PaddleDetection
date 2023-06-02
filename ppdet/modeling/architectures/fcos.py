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

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['FCOS', 'ARSL_FCOS']


@register
class FCOS(BaseArch):
    """
    FCOS network, see https://arxiv.org/abs/1904.01355

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        fcos_head (object): 'FCOSHead' instance
        ssod_loss (object): 'SSODFCOSLoss' instance, only used for semi-det(ssod) by DenseTeacher
    """

    __category__ = 'architecture'
    __inject__ = ['ssod_loss']

    def __init__(self,
                 backbone='ResNet',
                 neck='FPN',
                 fcos_head='FCOSHead',
                 ssod_loss='SSODFCOSLoss'):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fcos_head = fcos_head

        # for ssod, semi-det
        self.is_teacher = False
        self.ssod_loss = ssod_loss

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        fcos_head = create(cfg['fcos_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "fcos_head": fcos_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)

        self.is_teacher = self.inputs.get('is_teacher', False)
        if self.training or self.is_teacher:
            losses = self.fcos_head(fpn_feats, self.inputs)
            return losses
        else:
            fcos_head_outs = self.fcos_head(fpn_feats)
            bbox_pred, bbox_num = self.fcos_head.post_process(
                fcos_head_outs, self.inputs['scale_factor'])
            return {'bbox': bbox_pred, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_box', 'loss_quality']

    def get_ssod_loss(self, student_head_outs, teacher_head_outs, train_cfg):
        ssod_losses = self.ssod_loss(student_head_outs, teacher_head_outs,
                                     train_cfg)
        return ssod_losses


@register
class ARSL_FCOS(BaseArch):
    """
    FCOS ARSL network, see https://arxiv.org/abs/

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        fcos_head (object): 'FCOSHead_ARSL' instance
        fcos_cr_loss (object): 'FCOSLossCR' instance, only used for semi-det(ssod) by ARSL
    """

    __category__ = 'architecture'
    __inject__ = ['fcos_cr_loss']

    def __init__(self,
                 backbone,
                 neck,
                 fcos_head='FCOSHead_ARSL',
                 fcos_cr_loss='FCOSLossCR'):
        super(ARSL_FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fcos_head = fcos_head
        self.fcos_cr_loss = fcos_cr_loss

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        fcos_head = create(cfg['fcos_head'], **kwargs)

        # consistency regularization loss
        fcos_cr_loss = create(cfg['fcos_cr_loss'])

        return {
            'backbone': backbone,
            'neck': neck,
            'fcos_head': fcos_head,
            'fcos_cr_loss': fcos_cr_loss,
        }

    def forward(self, inputs, branch="supervised", teacher_prediction=None):
        assert branch in ['supervised', 'semi_supervised'], \
            print('In ARSL, type must be supervised or semi_supervised.')

        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = paddle.transpose(image, [0, 2, 3, 1])
        self.inputs = inputs

        if self.training:
            if branch == "supervised":
                out = self.get_loss()
            else:
                out = self.get_pseudo_loss(teacher_prediction)
        else:
            # norm test
            if branch == "supervised":
                out = self.get_pred()
                # predict pseudo labels
            else:
                out = self.get_pseudo_pred()
        return out

    # model forward 
    def model_forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        fcos_head_outs = self.fcos_head(fpn_feats)
        return fcos_head_outs

    # supervised loss for labeled data
    def get_loss(self):
        loss = {}
        tag_labels, tag_bboxes, tag_centerness = [], [], []
        for i in range(len(self.fcos_head.fpn_stride)):
            # labels, reg_target, centerness
            k_lbl = 'labels{}'.format(i)
            if k_lbl in self.inputs:
                tag_labels.append(self.inputs[k_lbl])
            k_box = 'reg_target{}'.format(i)
            if k_box in self.inputs:
                tag_bboxes.append(self.inputs[k_box])
            k_ctn = 'centerness{}'.format(i)
            if k_ctn in self.inputs:
                tag_centerness.append(self.inputs[k_ctn])
        fcos_head_outs = self.model_forward()
        loss_fcos = self.fcos_head.get_loss(fcos_head_outs, tag_labels,
                                            tag_bboxes, tag_centerness)
        loss.update(loss_fcos)
        return loss

    # unsupervised loss for unlabeled data
    def get_pseudo_loss(self, teacher_prediction):
        loss = {}
        fcos_head_outs = self.model_forward()
        unsup_loss = self.fcos_cr_loss(fcos_head_outs, teacher_prediction)
        for k in unsup_loss.keys():
            loss[k + '_pseudo'] = unsup_loss[k]
        return loss

    # get detection results for test, decode and rescale the results to original size
    def get_pred(self):
        fcos_head_outs = self.model_forward()
        scale_factor = self.inputs['scale_factor']
        bbox_pred, bbox_num = self.fcos_head.post_process(fcos_head_outs,
                                                          scale_factor)
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output

    # generate pseudo labels to guide student
    def get_pseudo_pred(self):
        fcos_head_outs = self.model_forward()
        pred_cls, pred_loc, pred_iou = fcos_head_outs[1:]  # 0 is locations
        for lvl, _ in enumerate(pred_loc):
            pred_loc[lvl] = pred_loc[lvl] / self.fcos_head.fpn_stride[lvl]

        return [pred_cls, pred_loc, pred_iou, self.fcos_head.fpn_stride]
