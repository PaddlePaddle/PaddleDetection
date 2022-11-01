from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

__all__ = ['COTLoss']

@register
class COTLoss(nn.Layer):
    __shared__ = ['num_classes']
    def __init__(self,
                 num_classes=80, 
                 cot_scale=1,
                 cot_lambda=1):
        super(COTLoss, self).__init__()
        self.cot_scale = cot_scale
        self.cot_lambda = cot_lambda    
        self.num_classes = num_classes    

    def forward(self, pred_cot, assigned_labels, cot_relation):    
        cls_name = 'loss_cot'
        loss_bbox = {}
        pred_cot = pred_cot.reshape([-1, pred_cot.shape[-1]])
        assigned_labels = assigned_labels.reshape([-1])
        mask = (assigned_labels < self.num_classes)
        if mask.sum() == 0:
            loss_bbox[cls_name] = paddle.zeros([1], dtype='float32')
        else:
            pred_cot = pred_cot[mask]
            assigned_labels = assigned_labels[mask]
            valid_cot_targets = []
            for i in range(assigned_labels.shape[0]):
                train_label = assigned_labels[i]
                valid_cot_targets.append(cot_relation[train_label])
            coco_targets = paddle.to_tensor(valid_cot_targets)
            coco_targets.stop_gradient = True
            coco_loss = - coco_targets * F.log_softmax(pred_cot * self.cot_scale)
            loss_bbox[cls_name] = self.cot_lambda * paddle.mean(paddle.sum(coco_loss, axis=-1))
        return loss_bbox



        targets = targets.transpose((0, 1, 3, 4, 2))
        targets= targets[:, :, :, :, 6:]
        tgt_labels = targets.reshape([-1, targets.shape[-1]])
        mask = tgt_labels.sum(-1) > 0
        if mask.sum() == 0:
            loss_bbox[cls_name] = paddle.zeros([1], dtype='float32')            
        else:
            scores = scores[mask]
            tgt_labels = tgt_labels[mask].argmax(-1)
            valid_cot_targets = []
            for i in range(tgt_labels.shape[0]):
                train_label = tgt_labels[i]
                if train_label < self.num_classes:
                    valid_cot_targets.append(cot_relation[train_label])
                else:
                    valid_cot_targets.append(np.zeros(80))
            coco_targets = paddle.to_tensor(valid_cot_targets)
            coco_targets.stop_gradient = True
            #print('debug coco_targets', coco_targets)
            #print('debug coco_scores', F.log_softmax(scores))
            coco_loss = - coco_targets * F.log_softmax(scores * self.cot_scale)
            #print('debug cot gradient scores', scores.stop_gradient)
            #print('debug cot gradient loss', coco_loss.stop_gradient)
            loss_bbox[cls_name] = self.cot_lambda * paddle.mean(paddle.sum(coco_loss, axis=-1))
        return loss_bbox