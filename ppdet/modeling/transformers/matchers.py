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
#
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ppdet.core.workspace import register, serializable
from .utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, boxes_iou
from ppdet.modeling.losses.sparsercnn_loss import get_bboxes_giou

__all__ = ['HungarianMatcher', 'HungarianMatcherDynamicK']


@register
@serializable
class HungarianMatcher(nn.Layer):
    __shared__ = ['use_focal_loss', 'with_mask', 'num_sample_points']

    def __init__(self,
                 matcher_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 1,
                     'dice': 1
                 },
                 use_focal_loss=False,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma


    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor|None): [b, query, h, w]
            gt_mask (List(Tensor)): list[[n, H, W]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = [len(a) for a in gt_class]
        if sum(num_gts) == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        logits = logits.detach()
        out_prob = F.sigmoid(logits.flatten(
            0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.detach().flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        out_prob = paddle.gather(out_prob, tgt_ids, axis=1)
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
                1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # Compute the L1 cost between boxes
        cost_bbox = (
            out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + \
            self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        # Compute the mask cost and dice cost
        if self.with_mask:
            assert (masks is not None and gt_mask is not None,
                    'Make sure the input has `mask` and `gt_mask`')
            # all masks share the same set of points for efficient matching
            sample_points = paddle.rand([bs, 1, self.num_sample_points, 2])
            sample_points = 2.0 * sample_points - 1.0

            out_mask = F.grid_sample(
                masks.detach(), sample_points, align_corners=False).squeeze(-2)
            out_mask = out_mask.flatten(0, 1)

            tgt_mask = paddle.concat(gt_mask).unsqueeze(1)
            sample_points = paddle.concat([
                a.tile([b, 1, 1, 1]) for a, b in zip(sample_points, num_gts)
                if b > 0
            ])
            tgt_mask = F.grid_sample(
                tgt_mask, sample_points, align_corners=False).squeeze([1, 2])

            with paddle.amp.auto_cast(enable=False):
                # binary cross entropy cost
                pos_cost_mask = F.binary_cross_entropy_with_logits(
                    out_mask, paddle.ones_like(out_mask), reduction='none')
                neg_cost_mask = F.binary_cross_entropy_with_logits(
                    out_mask, paddle.zeros_like(out_mask), reduction='none')
                cost_mask = paddle.matmul(
                    pos_cost_mask, tgt_mask, transpose_y=True) + paddle.matmul(
                        neg_cost_mask, 1 - tgt_mask, transpose_y=True)
                cost_mask /= self.num_sample_points

                # dice cost
                out_mask = F.sigmoid(out_mask)
                numerator = 2 * paddle.matmul(
                    out_mask, tgt_mask, transpose_y=True)
                denominator = out_mask.sum(
                    -1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
                cost_dice = 1 - (numerator + 1) / (denominator + 1)

                C = C + self.matcher_coeff['mask'] * cost_mask + \
                    self.matcher_coeff['dice'] * cost_dice

        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]
        sizes = [a.shape[0] for a in gt_bbox]
        indices = [
            linear_sum_assignment(c.split(sizes, -1)[i].numpy())
            for i, c in enumerate(C)
        ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
                j, dtype=paddle.int64)) for i, j in indices]


def paddle_cdist(x, y, p=2):
    y_len = y.shape[0]
    out = paddle.concat(
        [
            paddle.linalg.norm(
                x - y[i], p=p, axis=1, keepdim=True) for i in range(y_len)
        ],
        axis=1)
    return out


@register
class HungarianMatcherDynamicK(nn.Layer):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    __inject__ = ["giou_loss"]
    def __init__(
            self,
            cost_class: float=1,
            cost_bbox: float=1,
            cost_giou: float=1,
            cost_mask: float=1,
            use_focal: bool=True,
            use_fed_loss: bool=False,
            ota_k: int=5,
            # giou_loss: str="GIoULoss",
            focal_loss_alpha: float=0.25,
            focal_loss_gamma: float=2.0, ):
        """Creates the matche
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        self.ota_k = ota_k
        # self.giou_loss = GIoULoss()
        if self.use_focal:
            self.focal_loss_alpha = focal_loss_alpha
            self.focal_loss_gamma = focal_loss_gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @paddle.no_grad()
    def forward(self, outputs, targets):
        """ simOTA for detr"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        if self.use_focal or self.use_fed_loss:
            out_prob = F.sigmoid(outputs[
                "pred_logits"])  # [batch_size, num_queries, num_classes]
            out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 4]
        else:
            out_prob = F.softmax(
                outputs["pred_logits"],
                axis=-1)  # [batch_size, num_queries, num_classes]
            out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]

        indices = []
        matched_ids = []
        assert bs == len(targets)
        for batch_idx in range(bs):
            bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
            bz_out_prob = out_prob[batch_idx]
            bz_tgt_ids = targets[batch_idx]["labels"]
            num_insts = len(bz_tgt_ids)
            if num_insts == 0:  # empty object in key frame
                non_valid = paddle.zeros([bz_out_prob.shape[0]]) > 0
                indices_batchi = (non_valid, paddle.to_tensor(
                    []).astype("int64"))
                matched_qidx = paddle.to_tensor([]).astype("int64")
                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)
                continue

            bz_gtboxs = targets[batch_idx][
                'boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
            bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']
            fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                bbox_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                bbox_xyxy_to_cxcywh(
                    bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h)
                expanded_strides=32)

            pair_wise_ious, _ = boxes_iou(bz_boxes, bz_gtboxs_abs_xyxy)

            # Compute the classification cost.
            if self.use_focal:
                alpha = self.focal_loss_alpha
                gamma = self.focal_loss_gamma
                neg_cost_class = (1 - alpha) * (bz_out_prob**gamma) * (-(
                    1 - bz_out_prob + 1e-8).log())
                pos_cost_class = alpha * (
                    (1 - bz_out_prob)**gamma) * (-(bz_out_prob + 1e-8).log())
                # cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                cost_class = pos_cost_class.gather(index=bz_tgt_ids, axis=1) - \
                             neg_cost_class.gather(index=bz_tgt_ids, axis=1)
            elif self.use_fed_loss:
                # focal loss degenerates to naive one
                neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                pos_cost_class = (-(bz_out_prob + 1e-8).log())
                # cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                cost_class = pos_cost_class.gather(index=bz_tgt_ids, axis=1) - \
                             neg_cost_class.gather(index=bz_tgt_ids, axis=1)
            else:
                cost_class = -bz_out_prob[:, bz_tgt_ids]

            # Compute the L1 cost between boxes
            bz_image_size_out = targets[batch_idx]['image_size_xyxy']
            bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']

            bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2) # pred
            bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # normalize (x1, y1, x2, y2) # GT
            cost_bbox = paddle_cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

            # cost_giou = self.giou_loss(bz_boxes, bz_gtboxs_abs_xyxy.T) - 1
            cost_giou = -get_bboxes_giou(bz_boxes, bz_gtboxs_abs_xyxy)

            # Final cost matrix
            cost = self.cost_bbox * cost_bbox + \
                   self.cost_class * cost_class + \
                   self.cost_giou * cost_giou + \
                   100.0 * (~is_in_boxes_and_center)
            
            cost[~fg_mask] = cost[~fg_mask] + 10000.0

            # if bz_gtboxs.shape[0]>0:
            indices_batchi, matched_qidx = self.dynamic_k_matching(
                cost, pair_wise_ious, bz_gtboxs.shape[0])

            indices.append(indices_batchi)
            matched_ids.append(matched_qidx)

        return indices, matched_ids
    
    @paddle.no_grad()
    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = bbox_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.cast("int64") + b_r.cast("int64") + \
                        b_t.cast("int64") + b_b.cast("int64")) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (
            xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (
            xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (
            xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (
            xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.cast("int64") + b_r.cast("int64") + \
                          b_t.cast("int64") + b_b.cast("int64")) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    @paddle.no_grad()
    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = paddle.zeros_like(cost)  # [pred_box_num, num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = paddle.topk(ious_in_boxes_matrix, n_candidate_k, axis=0)
        dynamic_ks = paddle.clip(topk_ious.sum(0).cast("int64"), min=1)

        for gt_idx in range(num_gt): # For each GT Box
            _, pos_idx = paddle.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            # matching_matrix[:, gt_idx][pos_idx] = 1.0
            matching_matrix[pos_idx, gt_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            # _, cost_argmin = paddle.min(cost[anchor_matching_gt > 1], dim=1)
            cost_argmin = paddle.argmin(cost[anchor_matching_gt > 1], axis=1)
            matching_matrix[anchor_matching_gt > 1] *= 0

            ma_ma_idx = paddle.where(anchor_matching_gt > 1)[0].flatten()
            matching_matrix[ma_ma_idx, cost_argmin] = 1

        while (matching_matrix.sum(0) == 0).any():
            # num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = paddle.nonzero(
                matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = paddle.argmin(cost[:, gt_idx])
                # matching_matrix[:, gt_idx][pos_idx] = 1.0
                matching_matrix[pos_idx, gt_idx] = 1.0
            if (matching_matrix.sum(1) > 1
                ).sum() > 0:  # If a query matches more than one gt
                # _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                #                            dim=1)  # find gt for these queries with minimal cost
                cost_argmin = paddle.argmin(
                    cost[anchor_matching_gt > 1],
                    axis=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt >
                                1] *= 0  # reset mapping relationship

                ma_ma_idx = paddle.where(anchor_matching_gt > 1)[0].flatten()
                matching_matrix[ma_ma_idx, cost_argmin,
                                ] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].argmax(1)
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = paddle.argmin(cost, axis=0)

        return (selected_query, gt_indices), matched_query_id # [query Bool Idx], [GT Idx], each GT has a queryBox