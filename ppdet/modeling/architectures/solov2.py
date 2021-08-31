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

__all__ = ['SOLOv2']


@register
class SOLOv2(BaseArch):
    """
    SOLOv2 network, see https://arxiv.org/abs/2003.10152

    Args:
        backbone (object): an backbone instance
        solov2_head (object): an `SOLOv2Head` instance
        mask_head (object): an `SOLOv2MaskHead` instance
        neck (object): neck of network, such as feature pyramid network instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, solov2_head, mask_head, neck=None):
        super(SOLOv2, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.solov2_head = solov2_head
        self.mask_head = mask_head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        solov2_head = create(cfg['solov2_head'], **kwargs)
        mask_head = create(cfg['mask_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            'solov2_head': solov2_head,
            'mask_head': mask_head,
        }

    def model_arch(self):
        body_feats = self.backbone(self.inputs)

        body_feats = self.neck(body_feats)

        self.seg_pred = self.mask_head(body_feats)

        self.cate_pred_list, self.kernel_pred_list = self.solov2_head(
            body_feats)

    def get_loss(self, ):
        loss = {}
        # get gt_ins_labels, gt_cate_labels, etc.
        gt_ins_labels, gt_cate_labels, gt_grid_orders = [], [], []
        fg_num = self.inputs['fg_num']
        for i in range(len(self.solov2_head.seg_num_grids)):
            ins_label = 'ins_label{}'.format(i)
            if ins_label in self.inputs:
                gt_ins_labels.append(self.inputs[ins_label])
            cate_label = 'cate_label{}'.format(i)
            if cate_label in self.inputs:
                gt_cate_labels.append(self.inputs[cate_label])
            grid_order = 'grid_order{}'.format(i)
            if grid_order in self.inputs:
                gt_grid_orders.append(self.inputs[grid_order])

        loss_solov2 = self.solov2_head.get_loss(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            gt_ins_labels, gt_cate_labels, gt_grid_orders, fg_num)
        loss.update(loss_solov2)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        seg_masks, cate_labels, cate_scores, bbox_num = self.solov2_head.get_prediction(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            self.inputs['im_shape'], self.inputs['scale_factor'])
        outs = {
            "segm": seg_masks,
            "bbox_num": bbox_num,
            'cate_label': cate_labels,
            'cate_score': cate_scores
        }
        return outs
