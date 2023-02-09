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
import numpy as np

__all__ = ['FasterRCNN']


@register
class FasterRCNN(BaseArch):
    """
    Faster R-CNN network, see https://arxiv.org/abs/1506.01497

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        neck (object): 'FPN' instance
    """
    __category__ = 'architecture'
    __inject__ = ['bbox_post_process']

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 neck=None):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process

    def init_cot_head(self, relationship):
        self.bbox_head.init_cot_head(relationship)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)
            bbox_loss, _ = self.bbox_head(body_feats, rois, rois_num,
                                          self.inputs)
            return rpn_loss, bbox_loss
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            preds, _ = self.bbox_head(body_feats, rois, rois_num, None)
            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bbox, bbox_num, nms_keep_idx = self.bbox_post_process(preds, (rois, rois_num),
                                                    im_shape, scale_factor)

            # rescale the prediction back to origin image
            bboxes, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
                bbox, bbox_num, im_shape, scale_factor)

            if self.use_extra_data:
                extra_data = {}  # record the bbox output before nms, such like scores and nms_keep_idx
                """extra_data:{
                            'scores': predict scores,
                            'nms_keep_idx': bbox index before nms,
                           }
                """
                extra_data['scores'] = preds[1]  # predict scores (probability)
                # Todo: get logits output
                extra_data['nms_keep_idx'] = nms_keep_idx  # bbox index before nms
                return bbox_pred, bbox_num, extra_data
            else:
                return bbox_pred, bbox_num


    def get_loss(self, ):
        rpn_loss, bbox_loss = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        if self.use_extra_data:
            bbox_pred, bbox_num, extra_data = self._forward()
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num, 'extra_data': extra_data}
        else:
            bbox_pred, bbox_num = self._forward()
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output

    def target_bbox_forward(self, data):
        body_feats = self.backbone(data)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        rois = [roi for roi in data['gt_bbox']]
        rois_num = paddle.concat([paddle.shape(roi)[0] for roi in rois])

        preds, _ = self.bbox_head(body_feats, rois, rois_num, None, cot=True)
        return preds

    def relationship_learning(self, loader, num_classes_novel):
        print('computing relationship')
        train_labels_list = []
        label_list = []

        for step_id, data in enumerate(loader):
            _, bbox_prob = self.target_bbox_forward(data)      
            batch_size = data['im_id'].shape[0]
            for i in range(batch_size):
                num_bbox = data['gt_class'][i].shape[0]           
                train_labels = data['gt_class'][i]
                train_labels_list.append(train_labels.numpy().squeeze(1))
            base_labels = bbox_prob.detach().numpy()[:,:-1]
            label_list.append(base_labels)

        labels = np.concatenate(train_labels_list, 0)
        probabilities = np.concatenate(label_list, 0)
        N_t = np.max(labels) + 1
        conditional = []
        for i in range(N_t):
            this_class = probabilities[labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)
        return np.concatenate(conditional)