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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle
import paddle.nn.functional as F

__all__ = ['BlazeFace']


@register
class BlazeFace(BaseArch):
    """
    BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs,
               see https://arxiv.org/abs/1907.05047

    Args:
        backbone (nn.Layer): backbone instance
        neck (nn.Layer): neck instance
        blaze_head (nn.Layer): `blazeHead` instance
        post_process (object): `BBoxPostProcess` instance
    """

    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self, backbone, blaze_head, neck, post_process):
        super(BlazeFace, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.blaze_head = blaze_head
        self.post_process = post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)
        # head
        kwargs = {'input_shape': neck.out_shape}
        blaze_head = create(cfg['blaze_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            'blaze_head': blaze_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        # neck
        neck_feats = self.neck(body_feats)
        # blaze Head
        if self.training:
            return self.blaze_head(neck_feats, self.inputs['image'],
                                   self.inputs['gt_bbox'],
                                   self.inputs['gt_class'])
        else:
            preds, anchors = self.blaze_head(neck_feats, self.inputs['image'])
            bbox, bbox_num, nms_keep_idx = self.post_process(
                preds, anchors, self.inputs['im_shape'],
                self.inputs['scale_factor'])
            if self.use_extra_data:
                extra_data = {}  # record the bbox output before nms, such like scores and nms_keep_idx
                """extra_data:{
                            'scores': predict scores,
                            'nms_keep_idx': bbox index before nms,
                           }
                           """
                preds_logits = preds[1]  # [[1xNumBBoxNumClass]]
                extra_data['scores'] = F.softmax(paddle.concat(
                    preds_logits, axis=1)).transpose([0, 2, 1])
                extra_data['logits'] = paddle.concat(
                    preds_logits, axis=1).transpose([0, 2, 1])
                extra_data['nms_keep_idx'] = nms_keep_idx  # bbox index before nms
                return bbox, bbox_num, extra_data
            else:
                return bbox, bbox_num

    def get_loss(self, ):
        return {"loss": self._forward()}

    def get_pred(self):
        if self.use_extra_data:
            bbox_pred, bbox_num, extra_data = self._forward()
            output = {
                "bbox": bbox_pred,
                "bbox_num": bbox_num,
                "extra_data": extra_data
            }
        else:
            bbox_pred, bbox_num = self._forward()
            output = {
                "bbox": bbox_pred,
                "bbox_num": bbox_num,
            }

        return output
