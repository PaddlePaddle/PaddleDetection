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
from typing import KeysView
import copy
from ppdet.core.workspace import register, create, merge_config
from .meta_arch import BaseArch
from ppdet.data.reader import transform
import paddle
import os

import numpy as np
from operator import itemgetter
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.fluid import framework
from ppdet.core.workspace import register, create
from ppdet.modeling.bbox_utils import delta2bbox
from ppdet.data.transform.atss_assigner import bbox_overlaps
from ppdet.utils.logger import setup_logger
from ppdet.modeling.ssod.utils import filter_invalid, weighted_loss
from .multi_stream_detector import MultiSteamDetector
logger = setup_logger(__name__)

__all__ = ['DETR_SSOD']
@register
class DETR_SSOD(MultiSteamDetector):
    def __init__(self, teacher, student, train_cfg=None, test_cfg=None, PPDETRTransformer=None):
        super(DETR_SSOD, self).__init__(
            dict(teacher=teacher, student=student),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.semi_start_iters=train_cfg['semi_start_iters']
        self.ema_start_iters=train_cfg['ema_start_iters']
        self.momentum=0.9996
        self.cls_thr=None
        self.cls_thr_ig=None
        # self.id=0
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg['unsup_weight']
            self.sup_weight = self.train_cfg['sup_weight']
            self._teacher = None
            self._student = None
            self._transformer = None

    @classmethod
    def from_config(cls, cfg):
        teacher = create(cfg['teacher'])
        merge_config(cfg)
        student = create(cfg['student'])
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        PPDETRTransformer = cfg['PPDETRTransformer']
        return {
            'teacher': teacher,
            'student': student,
            'train_cfg': train_cfg,
            'test_cfg' : test_cfg,
            'PPDETRTransformer': PPDETRTransformer
        }
    def forward_train(self, inputs, **kwargs):
        if isinstance(inputs,dict):
            iter_id=inputs['iter_id']
        elif isinstance(inputs,list):
            iter_id=inputs[-1]
        if iter_id==self.semi_start_iters:
            self.update_ema_model(momentum=0)
        elif iter_id>self.semi_start_iters:
            self.update_ema_model(momentum=self.momentum)
        # elif iter_id<self.semi_start_iters:
        #     self.update_ema_model(momentum=0)
        if iter_id>=self.semi_start_iters:
            if iter_id==self.semi_start_iters:
                print('***********************')
                print('******semi start*******')
                print('***********************')
            data_sup_w, data_sup_s, data_unsup_w, data_unsup_s,_=inputs
            
            if data_sup_w['image'].shape != data_sup_s['image'].shape:
                data_sup_w, data_sup_s = align_weak_strong_shape(data_sup_w,data_sup_s)
                                                                                    
            if  'gt_bbox' in data_unsup_s.keys():
                del data_unsup_s['gt_bbox']
            if  'gt_class' in data_unsup_s.keys():
                del data_unsup_s['gt_class']  
            if  'gt_class' in data_unsup_w.keys():
                del data_unsup_w['gt_class']  
            if  'gt_bbox' in data_unsup_w.keys():
                del data_unsup_w['gt_bbox']    
            for k, v in data_sup_s.items():
                if k in ['epoch_id']:
                    continue
                elif k in ['gt_class','gt_bbox','is_crowd']:
                    data_sup_s[k].extend(data_sup_w[k])
                else:
                    data_sup_s[k] = paddle.concat([v,data_sup_w[k]])

            loss={}
            body_feats=self.student.backbone(data_sup_s)
            if self.student.neck is not None:
                    body_feats = self.student.neck(body_feats)
            out_transformer = self.student.transformer(body_feats,None,data_sup_s)
            sup_loss = self.student.detr_head(out_transformer, body_feats, data_sup_s)
            sup_loss.update({
                'loss':
                paddle.add_n([v for k, v in sup_loss.items() if 'log' not in k])
            })
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}

            loss.update(**sup_loss)   
            unsup_loss =  self.foward_unsup_train(data_unsup_w, data_unsup_s)
            unsup_loss.update({
            'loss':
            paddle.add_n([v for k, v in unsup_loss.items() if 'log' not in k])
        })
            # print(unsup_loss)
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            unsup_loss.update({
                'loss':
                paddle.add_n([v for k, v in unsup_loss.items() if 'log' not in k])
            })
            loss.update(**unsup_loss)      
            loss.update({'loss':  loss['sup_loss']+loss['unsup_loss'] })
        else:
            if iter_id==self.semi_start_iters-1:
                print('********************')
                print('******sup ing*******')
                print('********************')
            loss = {}
            sup_loss=self.student(inputs)
            sup_loss = {k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)      

        return loss
    def foward_unsup_train(self, data_unsup_w, data_unsup_s):

        with paddle.no_grad():
            body_feats=self.teacher.backbone(data_unsup_w)
            if self.teacher.neck is not None:
                body_feats = self.teacher.neck(body_feats,is_teacher=True)
            out_transformer = self.teacher.transformer(body_feats,None,data_unsup_w,is_teacher=True)
            preds = self.teacher.detr_head(out_transformer, body_feats)
            bbox, bbox_num = self.teacher.post_process_semi(preds)
        self.place=body_feats[0].place

        proposal_bbox_list = bbox[:, -4:]
        proposal_bbox_list = proposal_bbox_list.split(tuple(np.array(bbox_num)), 0)
        
        proposal_label_list = paddle.cast(bbox[:, :1], np.float32)
        proposal_label_list = proposal_label_list.split(tuple(np.array(bbox_num)), 0)
        proposal_score_list = paddle.cast(bbox[:, 1:81], np.float32)
        proposal_score_list = proposal_score_list.split(tuple(np.array(bbox_num)), 0)        
        proposal_bbox_list = [paddle.to_tensor(p, place=self.place) for p in proposal_bbox_list]
        proposal_label_list = [paddle.to_tensor(p, place=self.place) for p in proposal_label_list]
        # print(bbox[:,1].max())
        # filter invalid box roughly
        if isinstance(self.train_cfg['pseudo_label_initial_score_thr'], float):
            thr = self.train_cfg['pseudo_label_initial_score_thr']
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        # print("thr0.5 :",sum([len(bbox) for bbox in proposal_list]), "\tscore:",[proposal[:, -1] for proposal in proposal_list])
        proposal_bbox_list, proposal_label_list, proposal_score_list = list(
            zip(
                *[
                    filter_invalid(
                        proposal[:,:4],
                        proposal_label,
                        proposal_score,
                        thr=0.9,
                        min_size=self.train_cfg['min_pseduo_box_size'],
                    )
                    for proposal, proposal_label ,proposal_score in zip(
                        proposal_bbox_list, proposal_label_list, proposal_score_list
                    )
                ]
            )
        )

        teacher_bboxes = list(proposal_bbox_list )
        teacher_labels = proposal_label_list
        teacher_info=[teacher_bboxes,teacher_labels]
        student_unsup=data_unsup_s
        return self.compute_pseudo_label_loss(student_unsup, teacher_info,proposal_score_list)

    def compute_pseudo_label_loss(self,student_unsup, teacher_info,proposal_score_list):                                 

        pseudo_bboxes=list(teacher_info[0])
        pseudo_labels=list(teacher_info[1])
        losses = dict()
        for i in range(len(pseudo_bboxes)):
            if pseudo_labels[i].shape[0]==0:
                pseudo_bboxes[i]=paddle.zeros([0,4]).numpy()
                pseudo_labels[i]=paddle.zeros([0,1]).numpy()
            else:
                pseudo_bboxes[i]=pseudo_bboxes[i][:,:4].numpy()
                pseudo_labels[i]=pseudo_labels[i].numpy()
        for i in range(len(pseudo_bboxes)):
            pseudo_labels[i]= paddle.to_tensor(pseudo_labels[i],dtype=paddle.int32,place=self.place)
            pseudo_bboxes[i]= paddle.to_tensor(pseudo_bboxes[i],dtype=paddle.float32,place=self.place)         
        # print(pseudo_bboxes[0].shape[0])
        student_unsup.update({'gt_bbox':pseudo_bboxes,'gt_class':pseudo_labels})
        # student_data.update(gt_bbox=pseudo_bboxes,gt_class=pseudo_labels)
        pseudo_sum=0
        # self.id+=1
        for i in range(len(pseudo_bboxes)):
            pseudo_sum+=pseudo_bboxes[i].sum()
        # print(self.id)
        if pseudo_sum==0:
            # print('pseudo_sum=0')
            pseudo_bboxes[0]=paddle.ones([1,4])-0.5
            pseudo_labels[0]=paddle.ones([1,1]).astype('int32')
            student_unsup.update({'gt_bbox':pseudo_bboxes,'gt_class':pseudo_labels})
            body_feats=self.student.backbone(student_unsup)
            if self.student.neck is not None:
                    body_feats = self.student.neck(body_feats)
            out_transformer = self.student.transformer(body_feats,None,student_unsup)
            losses = self.student.detr_head(out_transformer, body_feats, student_unsup)
  
            losses['loss_class']*=0
            losses['loss_bbox']*=0
            losses['loss_giou']*=0
            losses['loss_class_aux']*=0
            losses['loss_bbox_aux']*=0
            losses['loss_giou_aux']*=0
            losses['loss_class_dn']*=0
            losses['loss_bbox_dn']*=0
            losses['loss_giou_dn']*=0
            losses['loss_class_aux_dn']*=0
            losses['loss_bbox_aux_dn']*=0
            losses['loss_giou_aux_dn']*=0
        else:
            gt_bbox=[]
            gt_class=[]
            images=[]
            proposal_score=[]
            for i in range(len(pseudo_bboxes)):
                if pseudo_labels[i].shape[0]==0:
                    continue
                else:
                    proposal_score.append(proposal_score_list[i].max(-1).unsqueeze(-1))
                    gt_class.append(pseudo_labels[i])
                    gt_bbox.append(pseudo_bboxes[i])
                    images.append(student_unsup['image'][i])
            images=paddle.stack(images)
            student_unsup.update({'image':images,'gt_bbox':gt_bbox,'gt_class':gt_class})
            body_feats=self.student.backbone(student_unsup)
            if self.student.neck is not None:
                    body_feats = self.student.neck(body_feats)
            out_transformer = self.student.transformer(body_feats,None,student_unsup)
            student_unsup.update({'gt_score':proposal_score})
            losses = self.student.detr_head(out_transformer, body_feats, student_unsup)
        return losses

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return paddle.stack(b, axis=-1)

def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (ow, oh)

def align_weak_strong_shape(data_weak, data_strong):
    shape_x = data_strong['image'].shape[2]
    shape_y = data_strong['image'].shape[3]
    
    target_size = [shape_x, shape_y]
    data_weak['image'] = F.interpolate(
        data_weak['image'],
        size=target_size,
        mode='bilinear',
        align_corners=False)
    return data_weak, data_strong
