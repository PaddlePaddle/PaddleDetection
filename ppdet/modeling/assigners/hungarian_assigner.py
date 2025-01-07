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

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

import paddle

from ppdet.core.workspace import register
from ppdet.modeling.assigners.pose_utils import bbox_cxcywh_to_xyxy

__all__ = ["PoseHungarianAssigner", "PseudoSampler", "HungarianAssigner"]


class AssignResult:
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_inds": self.gt_inds,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info


@register
class PoseHungarianAssigner:
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression oks cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt.
    - positive integer: positive sample, index (1-based) of assigned gt.

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        kpt_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        oks_weight (int | float, optional): The scale factor for regression
            oks cost. Default 1.0.
    """

    __inject__ = ["cls_cost", "kpt_cost", "oks_cost"]

    def __init__(
        self, cls_cost="ClassificationCost", kpt_cost="KptL1Cost", oks_cost="OksCost"
    ):
        self.cls_cost = cls_cost
        self.kpt_cost = kpt_cost
        self.oks_cost = oks_cost

    def assign(
        self, cls_pred, kpt_pred, gt_labels, gt_keypoints, gt_areas, img_meta, eps=1e-7
    ):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K*2].
            gt_labels (Tensor): Label of `gt_keypoints`, shape (num_gt,).
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v]. Shape [num_gt, K*3].
            gt_areas (Tensor): Ground truth mask areas, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_kpts = gt_keypoints.shape[0], kpt_pred.shape[0]
        if not gt_keypoints.astype("bool").any():
            num_gts = 0

        # 1. assign -1 by default
        assigned_gt_inds = paddle.full((num_kpts,), -1, dtype="int64")
        assigned_labels = paddle.full((num_kpts,), -1, dtype="int64")
        if num_gts == 0 or num_kpts == 0:
            # No ground truth or keypoints, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta["img_shape"]
        factor = paddle.to_tensor(
            [img_w, img_h, img_w, img_h], dtype=gt_keypoints.dtype
        ).reshape((1, -1))

        # 2. compute the weighted costs
        # classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        # keypoint regression L1 cost
        gt_keypoints_reshape = gt_keypoints.reshape((gt_keypoints.shape[0], -1, 3))
        valid_kpt_flag = gt_keypoints_reshape[..., -1]
        kpt_pred_tmp = kpt_pred.clone().detach().reshape((kpt_pred.shape[0], -1, 2))
        normalize_gt_keypoints = gt_keypoints_reshape[..., :2] / factor[
            :, :2
        ].unsqueeze(0)
        kpt_cost = self.kpt_cost(kpt_pred_tmp, normalize_gt_keypoints, valid_kpt_flag)
        # keypoint OKS cost
        kpt_pred_tmp = kpt_pred.clone().detach().reshape((kpt_pred.shape[0], -1, 2))
        kpt_pred_tmp = kpt_pred_tmp * factor[:, :2].unsqueeze(0)
        oks_cost = self.oks_cost(
            kpt_pred_tmp, gt_keypoints_reshape[..., :2], valid_kpt_flag, gt_areas
        )
        # weighted sum of above three costs
        cost = cls_cost + kpt_cost + oks_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = paddle.to_tensor(matched_row_inds)
        matched_col_inds = paddle.to_tensor(matched_col_inds)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds][..., 0].astype(
            "int64"
        )
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


class SamplingResult:
    """Bbox sampling result."""

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = paddle.zeros(
                gt_bboxes.shape, dtype=gt_bboxes.dtype
            ).reshape((-1, 4))
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.reshape((-1, 4))

            self.pos_gt_bboxes = paddle.index_select(
                gt_bboxes, self.pos_assigned_gt_inds.astype("int64"), axis=0
            )

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        """paddle.Tensor: concatenated positive and negative boxes"""
        return paddle.concat([self.pos_bboxes, self.neg_bboxes])

    def __nice__(self):
        data = self.info.copy()
        data["pos_bboxes"] = data.pop("pos_bboxes").shape
        data["neg_bboxes"] = data.pop("neg_bboxes").shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = "    " + ",\n    ".join(parts)
        return "{\n" + body + "\n}"

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            "pos_inds": self.pos_inds,
            "neg_inds": self.neg_inds,
            "pos_bboxes": self.pos_bboxes,
            "neg_bboxes": self.neg_bboxes,
            "pos_is_gt": self.pos_is_gt,
            "num_gts": self.num_gts,
            "pos_assigned_gt_inds": self.pos_assigned_gt_inds,
        }


@register
class PseudoSampler:
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (paddle.Tensor): Bounding boxes
            gt_bboxes (paddle.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """

        pos_inds = paddle.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
        neg_inds = paddle.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1)
        gt_flags = paddle.zeros([bboxes.shape[0]], dtype="int32")
        sampling_result = SamplingResult(
            pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags
        )
        return sampling_result


@register
class HungarianAssigner:
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    __inject__ = ["cls_cost", "reg_cost", "iou_cost"]

    def __init__(
        self, cls_cost="ClassificationCost", reg_cost="BBoxL1Cost", iou_cost="IoUCost"
    ):
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.iou_cost = iou_cost

    def assign(
        self,
        bbox_pred,
        cls_pred,
        gt_bboxes,
        gt_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_bboxes.shape[0], bbox_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = paddle.full((num_bboxes,), -1, dtype="int64")
        assigned_labels = paddle.full((num_bboxes,), -1, dtype="int64")
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta["img_shape"]
        factor = paddle.to_tensor([img_w, img_h, img_w, img_h
                                   ], dtype=gt_bboxes.dtype).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = paddle.to_tensor(matched_row_inds)
        matched_col_inds = paddle.to_tensor(matched_col_inds)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds][
            ..., 0].astype("int64")
        
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
