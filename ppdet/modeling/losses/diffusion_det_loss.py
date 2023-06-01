# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is based on https://github.com/PeizeSun/SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/loss.py
Ths copyright of PeizeSun/SparseR-CNN is as follows:
MIT License [see LICENSE for details]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.optimize import linear_sum_assignment
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling.transformers.utils import bbox_cxcywh_to_xyxy, boxes_iou
from ppdet.modeling.transformers.matchers import HungarianMatcherDynamicK
from ppdet.modeling.losses.sparsercnn_loss import get_bboxes_giou

__all__ = ["DiffusionDetSparseRCNNLoss"]


@register
class DiffusionDetSparseRCNNLoss(nn.Layer):
    """ This class computes the loss for DiffusionDet(Based on SparseRCNNLoss).
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __shared__ = ['num_classes']
    # __inject__ = ["matcher"]

    def __init__(self,
                 losses,
                 focal_loss_alpha,
                 focal_loss_gamma,
                 eos_coef,
                 use_focal=True,
                 use_fed_loss=False,
                 num_classes=80,
                 class_weight=2.,
                 l1_weight=5.,
                 giou_weight=2.):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            matcher: module able to compute a matching between targets and proposals
        """
        super().__init__()
        self.num_classes = num_classes
        weight_dict = {
            "loss_ce": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight
        }
        self.weight_dict = weight_dict
        self.matcher = HungarianMatcherDynamicK(focal_loss_alpha,
                                                focal_loss_gamma, class_weight,
                                                l1_weight, giou_weight)

        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss

        if self.use_focal:
            self.focal_loss_alpha = focal_loss_alpha
            self.focal_loss_gamma = focal_loss_gamma
        else:
            empty_weight = paddle.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    
    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes,
                             num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = paddle.unique(gt_classes)
        prob = paddle.ones([num_classes + 1], dtype="float32")
        
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.cast("float32").clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = paddle.multinomial(
                prob,
                num_fed_loss_classes - len(unique_gt_classes),
                replacement=False)
            fed_loss_classes = paddle.concat(
                [unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes
    

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size = len(targets)

        target_classes = paddle.full(
            src_logits.shape[:2], self.num_classes, dtype="int64")
        src_logits_list = []
        target_classes_o_list = []
        
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[
                gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if self.use_focal or self.use_fed_loss:
            num_boxes = sum(t.shape[0] for t in target_classes_o_list) if len(
                target_classes_o_list) != 0 else 1
            
            eye_class_add1 = paddle.eye(self.num_classes + 1,
                                        dtype=src_logits.dtype)
            target_classes_onehot = eye_class_add1[target_classes]

            gt_classes = target_classes
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            src_logits = src_logits.flatten(0, 1)
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            if self.use_focal:
                cls_loss = F.sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(
                    src_logits, target_classes_onehot, reduction="none")

            # self.fed_loss_num_classes = 50
            # self.fed_loss_cls_weights = paddle.ones([80])
            # self.use_fed_loss = True
            
            if self.use_fed_loss:
                K = self.num_classes
                N = src_logits.shape[0]
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights, )
                
                fed_loss_classes_mask = paddle.zeros([K + 1], dtype=fed_loss_classes.dtype)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.reshape([1, K]).expand([N, K]).cast("float32")

                loss_ce = paddle.sum(cls_loss * weight) / num_boxes
            else:
                loss_ce = paddle.sum(cls_loss) / num_boxes

            losses = {'loss_ce': loss_ce}
        else:
            raise NotImplementedError

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs  
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx][
                "boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx][
                "boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query].reshape([-1, 4]))
            pred_norm_box_list.append(bz_src_boxes[valid_query].reshape(
                [-1, 4]) / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx].reshape([-1, 4]))
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx].reshape(
                [-1, 4]))

        if len(pred_box_list) != 0:
            src_boxes = paddle.concat(pred_box_list)
            src_boxes_norm = paddle.concat(
                pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = paddle.concat(tgt_box_list)
            target_boxes_abs_xyxy = paddle.concat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(
                src_boxes_norm,
                bbox_cxcywh_to_xyxy(target_boxes),
                reduction='none')  # TODO ?
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - paddle.diag(
                get_bboxes_giou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {
                'loss_bbox': outputs['pred_boxes'].sum() * 0,
                'loss_giou': outputs['pred_boxes'].sum() * 0
            }

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = paddle.to_tensor(
            [num_boxes],
            dtype="float32",
            place=next(iter(outputs.values())).place)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_boxes, **kwargs)

                    w_dict = {}
                    for k in l_dict.keys():
                        if k in self.weight_dict:
                            w_dict[k + f'_{i}'] = l_dict[k] * self.weight_dict[
                                k]
                        else:
                            w_dict[k + f'_{i}'] = l_dict[k]
                    losses.update(w_dict)

        return losses

