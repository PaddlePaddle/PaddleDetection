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
from paddle.metric import accuracy
from ppdet.core.workspace import register
from ppdet.modeling.losses.iou_loss import GIoULoss

__all__ = ["SparseRCNNLoss"]


@register
class SparseRCNNLoss(nn.Layer):
    """ This class computes the loss for SparseRCNN.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 losses,
                 focal_loss_alpha,
                 focal_loss_gamma,
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
        self.losses = losses
        self.giou_loss = GIoULoss(reduction="sum")

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        self.matcher = HungarianMatcher(focal_loss_alpha, focal_loss_gamma,
                                        class_weight, l1_weight, giou_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat([
            paddle.gather(
                t["labels"], J, axis=0) for t, (_, J) in zip(targets, indices)
        ])
        target_classes = paddle.full(
            src_logits.shape[:2], self.num_classes, dtype="int32")
        for i, ind in enumerate(zip(idx[0], idx[1])):
            target_classes[int(ind[0]), int(ind[1])] = target_classes_o[i]
        target_classes.stop_gradient = True

        src_logits = src_logits.flatten(start_axis=0, stop_axis=1)

        # prepare one_hot target.
        target_classes = target_classes.flatten(start_axis=0, stop_axis=1)
        class_ids = paddle.arange(0, self.num_classes)
        labels = (target_classes.unsqueeze(-1) == class_ids).astype("float32")
        labels.stop_gradient = True

        # comp focal loss.
        class_loss = sigmoid_focal_loss(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum", ) / num_boxes
        losses = {'loss_ce': class_loss}

        if log:
            label_acc = target_classes_o.unsqueeze(-1)
            src_idx = [src for (src, _) in indices]

            pred_list = []
            for i in range(outputs["pred_logits"].shape[0]):
                pred_list.append(
                    paddle.gather(
                        outputs["pred_logits"][i], src_idx[i], axis=0))

            pred = F.sigmoid(paddle.concat(pred_list, axis=0))
            acc = accuracy(pred, label_acc.astype("int64"))
            losses["acc"] = acc

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs  # [batch_size, num_proposals, 4]
        src_idx = [src for (src, _) in indices]
        src_boxes_list = []

        for i in range(outputs["pred_boxes"].shape[0]):
            src_boxes_list.append(
                paddle.gather(
                    outputs["pred_boxes"][i], src_idx[i], axis=0))

        src_boxes = paddle.concat(src_boxes_list, axis=0)

        target_boxes = paddle.concat(
            [
                paddle.gather(
                    t['boxes'], I, axis=0)
                for t, (_, I) in zip(targets, indices)
            ],
            axis=0)
        target_boxes.stop_gradient = True
        losses = {}

        losses['loss_giou'] = self.giou_loss(src_boxes,
                                             target_boxes) / num_boxes

        image_size = paddle.concat([v["img_whwh_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='sum')
        losses['loss_bbox'] = loss_bbox / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = paddle.concat(
            [paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = paddle.concat(
            [paddle.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

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
        indices = self.matcher(outputs_without_aux, targets)

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
                indices = self.matcher(aux_outputs, targets)
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


class HungarianMatcher(nn.Layer):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 focal_loss_alpha,
                 focal_loss_gamma,
                 cost_class: float=1,
                 cost_bbox: float=1,
                 cost_giou: float=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @paddle.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Args:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 eg. outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 eg. targets = [{"labels":labels, "boxes": boxes}, ...,{"labels":labels, "boxes": boxes}]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = F.sigmoid(outputs["pred_logits"].flatten(
            start_axis=0, stop_axis=1))
        out_bbox = outputs["pred_boxes"].flatten(start_axis=0, stop_axis=1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat([v["labels"] for v in targets])
        assert (tgt_ids > -1).all()
        tgt_bbox = paddle.concat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.

        # Compute the classification cost.
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma

        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(
            1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob)
                                  **gamma) * (-(out_prob + 1e-8).log())

        cost_class = paddle.gather(
            pos_cost_class, tgt_ids, axis=1) - paddle.gather(
                neg_cost_class, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        image_size_out = paddle.concat(
            [v["img_whwh"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).tile(
            [1, num_queries, 1]).flatten(
                start_axis=0, stop_axis=1)
        image_size_tgt = paddle.concat([v["img_whwh_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = F.l1_loss(
            out_bbox_.unsqueeze(-2), tgt_bbox_,
            reduction='none').sum(-1)  # [batch_size * num_queries, num_tgts]

        # Compute the giou cost betwen boxes
        cost_giou = -get_bboxes_giou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.reshape([bs, num_queries, -1])

        sizes = [len(v["boxes"]) for v in targets]

        indices = [
            linear_sum_assignment(c[i].numpy())
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(paddle.to_tensor(
            i, dtype="int32"), paddle.to_tensor(
                j, dtype="int32")) for i, j in indices]


def box_area(boxes):
    assert (boxes[:, 2:] >= boxes[:, :2]).all()
    wh = boxes[:, 2:] - boxes[:, :2]
    return wh[:, 0] * wh[:, 1]


def boxes_iou(boxes1, boxes2):
    '''
    Compute iou

    Args:
        boxes1 (paddle.tensor) shape (N, 4)
        boxes2 (paddle.tensor) shape (M, 4)

    Return:
        (paddle.tensor) shape (N, M)
    '''
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = paddle.maximum(boxes1.unsqueeze(-2)[:, :, :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1.unsqueeze(-2)[:, :, 2:], boxes2[:, 2:])

    wh = (rb - lt).astype("float32").clip(min=1e-9)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1.unsqueeze(-1) + area2 - inter + 1e-9

    iou = inter / union
    return iou, union


def get_bboxes_giou(boxes1, boxes2, eps=1e-9):
    """calculate the ious of boxes1 and boxes2

    Args:
        boxes1 (Tensor): shape [N, 4]
        boxes2 (Tensor): shape [M, 4]
        eps (float): epsilon to avoid divide by zero

    Return:
        ious (Tensor): ious of boxes1 and boxes2, with the shape [N, M]
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = boxes_iou(boxes1, boxes2)

    lt = paddle.minimum(boxes1.unsqueeze(-2)[:, :, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1.unsqueeze(-2)[:, :, 2:], boxes2[:, 2:])

    wh = (rb - lt).astype("float32").clip(min=eps)
    enclose_area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (enclose_area - union) / enclose_area

    return giou


def sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="sum"):

    assert reduction in ["sum", "mean"
                         ], f'do not support this {reduction} reduction?'

    p = F.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
