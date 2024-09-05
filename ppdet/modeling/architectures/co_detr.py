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
import numpy as np
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['CO_DETR']
# Collaborative DETR, DINO use the same architecture as DETR

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (paddle.Tensor | np.ndarray): shape (n, 5)
        labels (paddle.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, paddle.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]
    

@register
class CO_DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['bbox_head']

    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 rpn_head=None,
                 roi_head=None,
                 bbox_head=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 ):
        super(CO_DETR, self).__init__()
        self.backbone = backbone
        if neck is not None:
            self.with_neck = True
        self.neck = neck
        self.query_head = query_head
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.bbox_head = bbox_head
        self.deploy = False
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
    
    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)
        # out_shape = neck and neck.out_shape or backbone.out_shape
        query_head = create(cfg['query_head'])
        out_shape = query_head.transformer.encoder.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        roi_head = create(cfg['roi_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            'query_head': query_head,
            'rpn_head': rpn_head,
            'roi_head':roi_head,
        }
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def get_inputs(self):
        img_metas = []
        gt_bboxes = []
        gt_labels = []

        for idx, im_shape in enumerate(self.inputs['im_shape']):
            img_meta = {
                'img_shape': im_shape.astype("int32").tolist() + [1, ],
                'batch_input_shape': self.inputs['image'].shape[-2:],
                'pad_mask': self.inputs['pad_mask'][idx],
            }
            img_metas.append(img_meta)

            gt_labels.append(self.inputs['gt_class'][idx])
            gt_bboxes.append(self.inputs['gt_bbox'][idx])

        return img_metas, gt_bboxes, gt_labels
    

    def get_pred(self):
        img = self.inputs['image']
        batch_size, _, height, width = img.shape
        img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3),
                scale_factor=self.inputs['scale_factor'][i])
            for i in range(batch_size)
        ]
        
        x = self.extract_feat(self.inputs)
        # from reprod_log import ReprodLogger
        # reprod_log_1 = ReprodLogger()
        # reprod_log_1.add("demo_test_1", x[0].cpu().detach().numpy())
        # reprod_log_1.save("result_1_paddle.npy")
        # breakpoint()
        bbox = self.query_head.simple_test(
            x, img_metas, rescale=True)
        bbox_num=[]
        for i in range(len(bbox)):
            bbox_num.append(bbox[i].shape[0])
        bbox_num = paddle.to_tensor(bbox_num)
        bbox = paddle.concat(bbox, axis=0)

        return {'bbox': bbox, 'bbox_num': bbox_num}

    def get_loss(self):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            gt_areas (list[Tensor]): mask areas corresponding to each box.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img = self.inputs['image']
        batch_size, _, height, width = img.shape
        img_metas, gt_bboxes, gt_labels = self.get_inputs()
        gt_bboxes_ignore = getattr(self.inputs, 'gt_bboxes_ignore', None)
        x = self.extract_feat(self.inputs)
        losses = dict()
        # DETR encoder and decoder forward
        if self.query_head is not None:
            bbox_losses, x = self.query_head.forward_train(x, img_metas, gt_bboxes,
                                                          gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)

        if self.rpn_head is not None:
            rois, rois_num, rpn_loss = self.rpn_head(x, self.inputs)
            losses.update(rpn_loss)
            
        positive_coords = []
        if self.roi_head is not None:
            roi_losses, _ = self.roi_head(x, rois, rois_num,
                                self.inputs)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else: 
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')  
            losses.update(roi_losses)

        # if self.bbox_head is not None:
        #     bbox_losses = self.bbox_head.forward_train(x,img_metas,gt_bboxes,gt_labels,)
        #     if self.with_pos_coord:
        #         positive_coords.append(bbox_losses.pop('pos_coords'))
        #     else: 
        #         if 'pos_coords' in bbox_losses.keys():
        #             tmp = bbox_losses.pop('pos_coords')  
        #     losses.update(bbox_losses)
        
        if self.with_pos_coord and len(positive_coords)>0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                if bbox_losses is not None:                    
                    losses.update(bbox_losses)
        loss = 0
        for k, v in losses.items():
            if isinstance(v, list):
                loss += sum(v)
            else: 
                loss += v
        losses={}
        losses['loss'] = loss
        return losses
    