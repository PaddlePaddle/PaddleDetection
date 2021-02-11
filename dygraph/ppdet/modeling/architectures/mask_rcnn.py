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

__all__ = ['MaskRCNN']


@register
class MaskRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 mask_head,
                 bbox_post_process,
                 mask_post_process,
                 neck=None):
        """
        backbone (nn.Layer): backbone instance.
        rpn_head (nn.Layer): generates proposals using backbone features.
        bbox_head (nn.Layer): a head that performs per-region computation.
        mask_head (nn.Layer): generates mask from bbox and backbone features.
        """
        super(MaskRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head

        self.bbox_post_process = bbox_post_process
        self.mask_post_process = mask_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)

        out_shape = neck and out_shape or bbox_head.get_head().out_shape
        kwargs = {'input_shape': out_shape}
        mask_head = create(cfg['mask_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
            "mask_head": mask_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)
            bbox_loss, bbox_feat = self.bbox_head(body_feats, rois, rois_num,
                                                  self.inputs)
            rois, rois_num = self.bbox_head.get_assigned_rois()
            bbox_targets = self.bbox_head.get_assigned_targets()
            # Mask Head needs bbox_feat in Mask RCNN
            mask_loss = self.mask_head(body_feats, rois, rois_num, self.inputs,
                                       bbox_targets, bbox_feat)
            return rpn_loss, bbox_loss, mask_loss
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            preds, feat_func = self.bbox_head(body_feats, rois, rois_num, None)

            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']

            bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
                                                    im_shape, scale_factor)
            mask_out = self.mask_head(
                body_feats, bbox, bbox_num, self.inputs, feat_func=feat_func)

            # rescale the prediction back to origin image
            bbox_pred = self.bbox_post_process.get_pred(bbox, bbox_num,
                                                        im_shape, scale_factor)
            origin_shape = self.bbox_post_process.get_origin_shape()
            mask_pred = self.mask_post_process(mask_out[:, 0, :, :], bbox_pred,
                                               bbox_num, origin_shape)
            return bbox_pred, bbox_num, mask_pred

    def get_loss(self, ):
        bbox_loss, mask_loss, rpn_loss = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        loss.update(mask_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num, mask_pred = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num, 'mask': mask_pred}
        return output
