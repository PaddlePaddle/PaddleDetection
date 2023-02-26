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

__all__ = ["DiffusionDetSparseRCNNLoss"]


@register
class DiffusionDetSparseRCNNLoss(nn.Layer):
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

        self.matcher = HungarianMatcherDynamicK(focal_loss_alpha, focal_loss_gamma,
                                        class_weight, l1_weight, giou_weight)
        
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
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
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
        prob = unique_gt_classes.new_ones(num_classes + 1).cast("float32")
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.cast("float32").clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = paddle.concat([unique_gt_classes, sampled_negative_classes])
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

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.full(src_logits.shape[:2], self.num_classes, dtype="int64")
        src_logits_list = []
        target_classes_o_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if self.use_focal or self.use_fed_loss:
            num_boxes = paddle.concat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

            # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
            #                                     dtype=src_logits.dtype, layout=src_logits.layout,
            #                                     device=src_logits.device)
            # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            eye_class_add1 = paddle.eye(self.num_classes + 1, dtype=src_logits.dtype)
            target_classes_onehot = eye_class_add1[target_classes]
            

            # gt_classes = paddle.argmax(target_classes_onehot, axis=-1)
            gt_classes = target_classes
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            src_logits = src_logits.flatten(0, 1)
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            if self.use_focal:
                cls_loss = F.sigmoid_focal_loss(src_logits, 
                                                target_classes_onehot, 
                                                alpha=self.focal_loss_alpha, 
                                                gamma=self.focal_loss_gamma, 
                                                reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
                
            if self.use_fed_loss:
                K = self.num_classes
                N = src_logits.shape[0]
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights,
                )
                fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()

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
        # idx = self._get_src_permutation_idx(indices)
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
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query].reshape([-1, 4]))
            pred_norm_box_list.append(bz_src_boxes[valid_query].reshape([-1, 4]) / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx].reshape([-1, 4]))
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx].reshape([-1, 4]))

        if len(pred_box_list) != 0:
            src_boxes = paddle.concat(pred_box_list)
            src_boxes_norm = paddle.concat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = paddle.concat(tgt_box_list)
            target_boxes_abs_xyxy = paddle.concat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, bbox_cxcywh_to_xyxy(target_boxes), reduction='none') # TODO ?
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - paddle.diag(get_bboxes_giou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

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



class HungarianMatcherDynamicK(nn.Layer):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,
                 cost_class: float = 1, 
                 cost_bbox: float = 1, 
                 cost_giou: float = 1, 
                 cost_mask: float = 1, 
                 use_focal: bool = True,
                 use_fed_loss: bool =False,
                 ota_k: int = 5,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 ):
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
        if self.use_focal:
            self.focal_loss_alpha = focal_loss_alpha
            self.focal_loss_gamma = focal_loss_gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    @paddle.no_grad()
    def forward(self, outputs, targets):
        """ simOTA for detr"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        if self.use_focal or self.use_fed_loss:
            out_prob = F.sigmoid(outputs["pred_logits"])  # [batch_size, num_queries, num_classes]
            out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 4]
        else:
            out_prob = F.softmax(outputs["pred_logits"], axis=-1)  # [batch_size, num_queries, num_classes]
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
                indices_batchi = (non_valid, paddle.to_tensor([]).astype("int64"))
                matched_qidx = paddle.to_tensor([]).astype("int64")
                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)
                continue

            bz_gtboxs = targets[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
            bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']
            fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                bbox_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                bbox_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h)
                expanded_strides=32
            )

            pair_wise_ious, _ = boxes_iou(bz_boxes, bz_gtboxs_abs_xyxy)

            # Compute the classification cost.
            if self.use_focal:
                alpha = self.focal_loss_alpha
                gamma = self.focal_loss_gamma
                neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
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
            # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
            # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
            # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

            bz_image_size_out = targets[batch_idx]['image_size_xyxy']
            bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']

            bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2)
            bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # normalize (x1, y1, x2, y2)
            cost_bbox = paddle_cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

            cost_giou = -get_bboxes_giou(bz_boxes, bz_gtboxs_abs_xyxy)

            # Final cost matrix
            cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
            # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
            cost[~fg_mask] = cost[~fg_mask] + 10000.0

            # if bz_gtboxs.shape[0]>0:
            indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

            indices.append(indices_batchi)
            matched_ids.append(matched_qidx)

        return indices, matched_ids

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
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.cast("int64") + b_r.cast("int64") + \
                          b_t.cast("int64") + b_b.cast("int64")) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = paddle.zeros_like(cost)  # [300, num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = paddle.topk(ious_in_boxes_matrix, n_candidate_k, axis=0)
        dynamic_ks = paddle.clip(topk_ious.sum(0).cast("int64"), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = paddle.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
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
            unmatch_id = paddle.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = paddle.argmin(cost[:, gt_idx])
                # matching_matrix[:, gt_idx][pos_idx] = 1.0
                matching_matrix[pos_idx, gt_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                # _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                #                            dim=1)  # find gt for these queries with minimal cost
                cost_argmin = paddle.argmin(cost[anchor_matching_gt > 1],
                                            axis=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                
                ma_ma_idx = paddle.where(anchor_matching_gt > 1)[0].flatten()
                matching_matrix[ma_ma_idx, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        # gt_indices = matching_matrix[selected_query].max(1)[1]
        gt_indices = matching_matrix[selected_query].argmax(1)
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        # matched_query_id = torch.min(cost, dim=0)[1]
        matched_query_id = paddle.argmin(cost, axis=0)

        return (selected_query, gt_indices), matched_query_id







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


# def sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="sum"):

#     assert reduction in ["sum", "mean"
#                          ], f'do not support this {reduction} reduction?'

#     p = F.sigmoid(inputs)
#     ce_loss = F.binary_cross_entropy_with_logits(
#         inputs, targets, reduction="none")
#     p_t = p * targets + (1 - p) * (1 - targets)
#     loss = ce_loss * ((1 - p_t)**gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

#     if reduction == "mean":
#         loss = loss.mean()
#     elif reduction == "sum":
#         loss = loss.sum()

#     return loss


def bbox_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)


def bbox_xyxy_to_cxcywh(x):
    x1, y1, x2, y2 = x.split(4, axis=-1)
    return paddle.concat(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=-1)
    

def paddle_cdist(x, y, p=2):
    y_len = y.shape[0]
    out = paddle.concat(
        [paddle.linalg.norm(x-y[i], p=p, axis=1, keepdim=True) for i in range(y_len)],
        axis=1
    )
    return out
