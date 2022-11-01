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
import numpy as np
import copy
import paddle

__all__ = ['YOLOv3']


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

    def init_cot_head(self, relationship, cot_lambda, cot_scale):
        self.yolo_head.init_cot_head(relationship, cot_lambda, cot_scale)

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
                    _, bbox, bbox_num, _ = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors)
                elif self.post_process is not None:
                    bbox, bbox_num = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors,
                        self.inputs['im_shape'], self.inputs['scale_factor'])
                else:
                    bbox, bbox_num = self.yolo_head.post_process(
                        yolo_head_outs, self.inputs['scale_factor'])
                output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def relationship_learning(self, loader, num_classes_novel, cot_scale, coco_labels, novel_labels):
            # self.model.eval()
        print('computing relationship')

        for i in range(10):
            train_labels_list = []
            labels_list = []
            for step_id, data in enumerate(loader):
                bbox_prob, cur_label, cot_head_dict = self.target_bbox_forward(data)
                bbox_prob.detach().numpy()

                train_labels_list.append(cur_label.numpy())
                labels_list.append(bbox_prob)
                # inputs = data
                # batch_size = inputs['im_id'].shape[0]
                # offset_bbox = 0
                # #print('batch results', inputs['gt_class'])
                # for i in range(batch_size):
                #     num_bbox = inputs['gt_class'][i].shape[0]
                #     #for j in range(num_bbox):
                #     #    print('gt predict', inputs['im_id'][i].tolist()[0], inputs['gt_class'][i][j].tolist()[0], inputs['gt_bbox'][i][j].tolist(), preds[offset_bbox+j].tolist()[0])            
                #     offset_bbox += num_bbox
                #     train_labels = inputs['gt_class'][i]
                #     train_labels_list.append(train_labels.numpy().squeeze(1))
            labels = np.concatenate(train_labels_list, 0)
            if len(np.unique(labels)) == num_classes_novel:
                logits = np.concatenate(labels_list, 0)
                break
            else:
                print(labels)
                continue
        #print('debug cot1', labels.shape, logits.shape)
        # print(labels, logits)
        def softmax_np(x, dim):
            max_el = np.max(x, axis=dim, keepdims=True)
            x = x - max_el
            x = np.exp(x)
            s = np.sum(x, axis=dim, keepdims=True)
            return x / s

        # probabilities = softmax_np(logits * cot_scale, dim=1)
        probabilities = logits
        N_t = np.max(labels) + 1
        conditional = []
        for i in range(N_t):
            this_class = probabilities[labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)

        ### debug co-tuning relationship begin ###
        print('------- co-tuning relationship -------------')
        for i in range(num_classes_novel):
            average = conditional[i]
            values, indices = paddle.topk(paddle.to_tensor(average), k=80)
            print(indices)
            print('Top related novel class ', i, novel_labels[i])
            for k in range(3):
                indice = int(indices.tolist()[0][k])
                print('\t', 'coco class:', indice, coco_labels[indice], 'prob:', values.tolist()[0][k])
            print('Least related novel class ', i, novel_labels[i])
            for k in range(3):
                indice = int(indices.tolist()[0][79-k])
                print('\t', 'coco class:', indice, coco_labels[indice], 'prob:', values.tolist()[0][79-k])
        ### debug co-tuning relationship end ###

        return np.concatenate(conditional), cot_head_dict


    def target_bbox_forward(self, inputs):
        self.backbone.eval()
        self.neck.eval()
        # self.yolo_head.eval()
        body_feats = self.backbone(inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        yolo_head_outs = self.yolo_head.forward_targets(neck_feats, inputs)
        p = yolo_head_outs[0]
        t = yolo_head_outs[1]
        pcls = p.reshape([-1, p.shape[-1]])
        tcls = t.reshape([-1])
        mask = (tcls < 4)    
        cot_head_dict = copy.deepcopy(self.yolo_head.pred_cls.state_dict())
        
        return pcls[mask], tcls[mask], cot_head_dict