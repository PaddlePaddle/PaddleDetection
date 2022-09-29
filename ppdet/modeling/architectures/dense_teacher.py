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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['DenseTeacher']


@register
class DenseTeacher(BaseArch):
    __category__ = 'architecture'
    """
    DenseTeacher network, see 
    Args:
        teacher (object): teacher detector model instance
        student (object): student detector model instance
    """

    def __init__(self,
                 teacher='FCOS',
                 student='FCOS',
                 train_cfg=None,
                 test_cfg=None):
        super(DenseTeacher, self).__init__()
        self.teacher = teacher
        self.student = student
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        teacher = create(cfg['teacher'])
        student = create(cfg['student'])
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        return {
            'teacher': teacher,
            'student': student,
            'train_cfg': train_cfg,
            'test_cfg': test_cfg
        }

    def _forward(self):
        body_feats = self.teacher.backbone(self.inputs)
        fpn_feats = self.teacher.neck(body_feats)

        if not self.training:
            fcos_head_outs = self.teacher.fcos_head(fpn_feats)
            bbox_pred, bbox_num = self.teacher.fcos_head.post_process(
                fcos_head_outs, self.inputs['scale_factor'])
            return {'bbox': bbox_pred, 'bbox_num': bbox_num}

        return True

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
