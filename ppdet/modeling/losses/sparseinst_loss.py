# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from scipy.optimize import linear_sum_assignment

from ..mask_utils import nested_masks_from_list

__all__ = ['SparseInstLoss']


def _dice_score(inputs, targets):
    inputs = F.sigmoid(inputs)
    numerator = 2 * paddle.matmul(inputs, targets.t())
    denominator = (inputs * inputs).sum(-1)[:, None] + (targets * targets
                                                        ).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def _compute_mask_iou(inputs, targets):
    inputs = F.sigmoid(inputs)
    # thresholding 
    binarized_inputs = (inputs >= 0.4).astype(paddle.float32)
    targets = (targets > 0.5).astype(paddle.float32)
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


class SparseInstMatcher(nn.Layer):
    def __init__(self, alpha=0.8, beta=0.2):
        super(SparseInstMatcher, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets, input_shape):
        with paddle.no_grad():
            B, N, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']
            pred_logits = F.sigmoid(outputs['pred_logits'])

            tgt_ids = paddle.concat([v["labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(paddle.to_tensor(
                    [], dtype=pred_logits.dtype), paddle.to_tensor(
                        [], dtype=pred_logits.dtype))] * B
            tgt_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
            tgt_masks = tgt_masks.astype(pred_masks.dtype)

            tgt_masks = F.interpolate(
                tgt_masks[:, None],
                size=pred_masks.shape[-2:],
                mode="bilinear",
                align_corners=False).squeeze(1)

            pred_masks = pred_masks.reshape((B * N, -1))
            tgt_masks = tgt_masks.flatten(1)

            mask_score = _dice_score(pred_masks, tgt_masks)
            # Nx(Number of gts)
            matching_prob = pred_logits.reshape((B * N, -1))[:, tgt_ids]
            C = (mask_score**self.alpha) * (matching_prob**self.beta)
            C = C.reshape((B, N, -1))
            # hungarian matching
            sizes = [len(v["masks"]) for v in targets]
            indices = [
                linear_sum_assignment(
                    c[i], maximize=True)
                for i, c in enumerate(C.split(sizes, -1))
            ]
            indices = [(paddle.to_tensor(
                i, dtype=paddle.int64), paddle.to_tensor(
                    j, dtype=paddle.int64)) for i, j in indices]
            return indices


@register
@serializable
class SparseInstLoss(object):
    """
    SOLOv2Loss
    Args:
        ins_loss_weight (float): Weight of instance loss.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
    """

    def __init__(self,
                 class_loss_weight=2.0,
                 mask_pixel_loss_weight=5.0,
                 mask_dice_loss_weight=2.0,
                 objectness_loss_weight=1.0,
                 matcher_alpha=0.8,
                 matcher_beta=0.2):
        self.class_loss_weight = class_loss_weight
        self.mask_pixel_loss_weight = mask_pixel_loss_weight
        self.mask_dice_loss_weight = mask_dice_loss_weight
        self.objectness_loss_weight = objectness_loss_weight

        self.matcher = SparseInstMatcher(0.8, 0.2)

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

    def loss_labels(self, outputs, targets, indices, num_instances):
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.full(
            src_logits.shape[:2], self.num_classes, dtype=paddle.int64)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = paddle.nonzero(
            target_classes != self.num_classes, as_tuple=True)[0]
        labels = paddle.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = F.sigmoid_focal_loss(
            src_logits,
            labels,
            alpha=0.25,
            gamma=2.0,
            reduction="sum", ) / num_instances

        losses = {'loss_ce': class_loss}
        return losses

    def loss_masks_with_iou_objectness(self, outputs, targets, indices,
                                       num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # Bx100xHxW
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        with paddle.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        num_masks = [len(t["masks"]) for t in targets]
        target_masks = target_masks.astype(src_masks.dtype)
        if len(target_masks) == 0:
            losses = {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None],
            size=src_masks.shape[-2:],
            mode='bilinear',
            align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        # FIXME: tgt_idx
        mix_tgt_idx = paddle.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum:cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        with paddle.no_grad():
            ious = _compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(
                src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(
                src_masks, target_masks, reduction='mean')
        }
        return losses

    def __call__(self, outputs, targets, input_shape):
        """
        Get loss of network of SOLOv2.
        Args:
            ins_pred_list (list): Variable list of instance branch output.
            ins_label_list (list): List of instance labels pre batch.
            cate_preds (list): Concat Variable list of categroy branch output.
            cate_labels (list): Concat list of categroy labels pre batch.
            num_ins (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Variable): The instance loss Variable of SOLOv2 network.
            loss_cate (Variable): The category loss Variable of SOLOv2 network.
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = paddle.to_tensor([num_instances], dtype=paddle.float32)

        #if is_dist_avail_and_initialized():
        #    paddle.distributed.all_reduce(num_instances)
        #num_instances = paddle.clamp(num_instances / paddle.distributed.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        losses.update(
            self.loss_labels(
                loss,
                outputs,
                targets,
                indices,
                num_instances,
                input_shape=input_shape))
        losses.update(
            self.loss_masks_with_iou_objectness(
                loss,
                outputs,
                targets,
                indices,
                num_instances,
                input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses
