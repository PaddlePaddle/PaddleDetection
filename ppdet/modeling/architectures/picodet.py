# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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
from paddlecv.ppcv.register import MODEL, BACKBONE, NECK, HEAD

from .meta_arch import BaseArch

__all__ = ['PicoDet']


@MODEL.register()
class PicoDet(BaseArch):
    """
    Generalized Focal Loss network, see https://arxiv.org/abs/2006.04388

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        head (object): 'PicoHead' instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, neck, head='PicoHead', **kwargs):
        super(PicoDet, self).__init__()
        self.backbone = BACKBONE.build(backbone)
        kwargs = {'in_channels': [i.channels for i in self.backbone.out_shape]}
        self.neck = NECK.build(neck, **kwargs)
        kwargs = {'in_channels': [i.channels for i in self.neck.out_shape]}
        self.head = HEAD.build(head, **kwargs)
        self.export_post_process = True
        self.export_nms = True

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        head_outs = self.head(fpn_feats, self.export_post_process)
        if self.training or not self.export_post_process:
            return head_outs, None
        else:
            scale_factor = self.inputs['scale_factor']
            bboxes, bbox_num = self.head.post_process(
                head_outs, scale_factor, export_nms=self.export_nms)
            return bboxes, bbox_num

    def get_loss(self, ):
        loss = {}

        head_outs, _ = self._forward()
        loss_gfl = self.head.get_loss(head_outs, self.inputs)
        loss.update(loss_gfl)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        if not self.export_post_process:
            return {'picodet': self._forward()[0]}
        elif self.export_nms:
            bbox_pred, bbox_num = self._forward()
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
            return output
        else:
            bboxes, mlvl_scores = self._forward()
            output = {'bbox': bboxes, 'scores': mlvl_scores}
            return output
