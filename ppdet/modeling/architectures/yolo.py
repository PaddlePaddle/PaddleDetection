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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..post_process import JDEBBoxPostProcess

__all__ = ['YOLOv3']
# YOLOv3,PP-YOLO,PP-YOLOv2,PP-YOLOE,PP-YOLOE+ use the same architecture as YOLOv3
# PP-YOLOE and PP-YOLOE+ are recommended to use PPYOLOE architecture in ppyoloe.py, especially when use distillation or aux head


@register
class YOLOv3(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DarkNet',
                 neck='YOLOv3FPN',
                 yolo_head='YOLOv3Head',
                 post_process='BBoxPostProcess',
                 data_format='NCHW',
                 for_mot=False):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            bbox_post_process (object): `BBoxPostProcess` instance
            data_format (str): data format, NCHW or NHWC
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv3, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.return_idx = isinstance(post_process, JDEBBoxPostProcess)

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
        if self.for_mot:
            neck_feats = self.neck(body_feats, self.for_mot)
        else:
            neck_feats = self.neck(body_feats)

        if isinstance(neck_feats, dict):
            assert self.for_mot == True
            emb_feats = neck_feats['emb_feats']
            neck_feats = neck_feats['yolo_feats']

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)

            if self.for_mot:
                return {'det_losses': yolo_losses, 'emb_feats': emb_feats}
            else:
                return yolo_losses

        else:
            yolo_head_outs = self.yolo_head(neck_feats)

            if self.for_mot:
                # the detection part of JDE MOT model
                boxes_idx, bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors)
                output = {
                    'bbox': bbox,
                    'bbox_num': bbox_num,
                    'boxes_idx': boxes_idx,
                    'nms_keep_idx': nms_keep_idx,
                    'emb_feats': emb_feats,
                }
            else:
                if self.return_idx:
                    # the detection part of JDE MOT model
                    _, bbox, bbox_num, nms_keep_idx = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors)
                elif self.post_process is not None:
                    # anchor based YOLOs: YOLOv3,PP-YOLO,PP-YOLOv2 use mask_anchors
                    bbox, bbox_num, nms_keep_idx = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors,
                        self.inputs['im_shape'], self.inputs['scale_factor'])
                else:
                    # anchor free YOLOs: PP-YOLOE, PP-YOLOE+
                    bbox, bbox_num, nms_keep_idx = self.yolo_head.post_process(
                        yolo_head_outs, self.inputs['scale_factor'])

                if self.use_extra_data:
                    extra_data = {}  # record the bbox output before nms, such like scores and nms_keep_idx
                    """extra_data:{
                                'scores': predict scores,
                                'nms_keep_idx': bbox index before nms,
                               }
                    """
                    extra_data['scores'] = yolo_head_outs[0]  # predict scores (probability)
                    # Todo: get logits output
                    extra_data['nms_keep_idx'] = nms_keep_idx
                    # Todo support for mask_anchors yolo
                    output = {'bbox': bbox, 'bbox_num': bbox_num, 'extra_data': extra_data}
                else:
                    output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
