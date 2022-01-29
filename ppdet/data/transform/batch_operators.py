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

import typing

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import cv2
import math
import numpy as np
from .operators import register_op, BaseOperator, Resize
from .op_helper import jaccard_overlap, gaussian2D, gaussian_radius, draw_umich_gaussian
from .atss_assigner import ATSSAssigner
from scipy import ndimage

from ppdet.modeling import bbox_utils
from ppdet.utils.logger import setup_logger
from ppdet.modeling.keypoint_utils import get_affine_transform, affine_transform
logger = setup_logger(__name__)

__all__ = [
    'PadBatch',
    'BatchRandomResize',
    'Gt2YoloTarget',
    'Gt2FCOSTarget',
    'Gt2TTFTarget',
    'Gt2Solov2Target',
    'Gt2SparseRCNNTarget',
    'PadMaskBatch',
    'Gt2GFLTarget',
    'Gt2CenterNetTarget',
    'PadGT',
]


@register_op
class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        # multi scale input is nested list
        if isinstance(samples,
                      typing.Sequence) and len(samples) > 0 and isinstance(
                          samples[0], typing.Sequence):
            inner_samples = samples[0]
        else:
            inner_samples = samples

        max_shape = np.array(
            [data['image'].shape for data in inner_samples]).max(axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in inner_samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm

            if 'gt_rbox2poly' in data and data['gt_rbox2poly'] is not None:
                # ploy to rbox
                polys = data['gt_rbox2poly']
                rbox = bbox_utils.poly2rbox(polys)
                data['gt_rbox'] = rbox

        return samples


@register_op
class BatchRandomResize(BaseOperator):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio,
                 interp=cv2.INTER_NEAREST,
                 random_size=True,
                 random_interp=False):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.interp = interp
        assert isinstance(target_size, (
            int, Sequence)), "target_size must be int, list or tuple"
        if random_size and not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples, context=None):
        if self.random_size:
            index = np.random.choice(len(self.target_size))
            target_size = self.target_size[index]
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, keep_ratio=self.keep_ratio, interp=interp)
        return resizer(samples, context=context)


@register_op
class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            if 'gt_score' not in sample:
                sample['gt_score'] = np.ones(
                    (gt_bbox.shape[0], 1), dtype=np.float32)
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match 
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou 
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh and target[idx, 5, gj,
                                                                gi] == 0.:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target

            # remove useless gt_class and gt_score after target calculated
            sample.pop('gt_class')
            sample.pop('gt_score')

        return samples


@register_op
class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTarget, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in locations]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2
        beg = 0
        clipped_box = bboxes.copy()
        for lvl, stride in enumerate(self.downsample_ratios):
            end = beg + num_points_each_level[lvl]
            stride_exp = self.center_sampling_radius * stride
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)
            beg = end
        l_res = xs - clipped_box[:, :, 0]
        r_res = clipped_box[:, :, 2] - xs
        t_res = ys - clipped_box[:, :, 1]
        b_res = clipped_box[:, :, 3] - ys
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
        inside_gt_box = np.min(clipped_box_reg_targets, axis=2) > 0
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            im = sample['image']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            # calculate the locations
            h, w = im.shape[1:3]
            points, num_points_each_level = self._compute_points(w, h)
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):
                object_scale_exp.append(
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (
                bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])

            l_res = xs - bboxes[:, 0]
            r_res = bboxes[:, 2] - xs
            t_res = ys - bboxes[:, 1]
            b_res = bboxes[:, 3] - ys
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
            if self.center_sampling_radius > 0:
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)
            lower_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[is_inside_box == 0] = self.INF
            points2gtarea[is_match_current_level == 0] = self.INF
            points2min_area = points2gtarea.min(axis=1)
            points2min_area_ind = points2gtarea.argmin(axis=1)
            labels = gt_class[points2min_area_ind] + 1
            labels[points2min_area == self.INF] = 0
            reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                  reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])
            ctn_targets[labels <= 0] = 0
            pos_ind = np.nonzero(labels != 0)
            reg_targets_pos = reg_targets[pos_ind[0], :]
            split_sections = []
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            labels_by_level = np.split(labels, split_sections, axis=0)
            reg_targets_by_level = np.split(reg_targets, split_sections, axis=0)
            ctn_targets_by_level = np.split(ctn_targets, split_sections, axis=0)
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))
                if self.norm_reg_targets:
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],
                            newshape=[grid_h, grid_w, 4])
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])

            sample.pop('is_crowd', None)
            sample.pop('difficult', None)
            sample.pop('gt_class', None)
            sample.pop('gt_bbox', None)
        return samples


@register_op
class Gt2GFLTarget(BaseOperator):
    """
    Generate GFocal loss targets by groud truth data
    """

    def __init__(self,
                 num_classes=80,
                 downsample_ratios=[8, 16, 32, 64, 128],
                 grid_cell_scale=4,
                 cell_offset=0):
        super(Gt2GFLTarget, self).__init__()
        self.num_classes = num_classes
        self.downsample_ratios = downsample_ratios
        self.grid_cell_scale = grid_cell_scale
        self.cell_offset = cell_offset

        self.assigner = ATSSAssigner()

    def get_grid_cells(self, featmap_size, scale, stride, offset=0):
        """
        Generate grid cells of a feature map for target assignment.
        Args:
            featmap_size: Size of a single level feature map.
            scale: Grid cell scale.
            stride: Down sample stride of the feature map.
            offset: Offset of grid cells.
        return:
            Grid_cells xyxy position. Size should be [feat_w * feat_h, 4]
        """
        cell_size = stride * scale
        h, w = featmap_size
        x_range = (np.arange(w, dtype=np.float32) + offset) * stride
        y_range = (np.arange(h, dtype=np.float32) + offset) * stride
        x, y = np.meshgrid(x_range, y_range)
        y = y.flatten()
        x = x.flatten()
        grid_cells = np.stack(
            [
                x - 0.5 * cell_size, y - 0.5 * cell_size, x + 0.5 * cell_size,
                y + 0.5 * cell_size
            ],
            axis=-1)
        return grid_cells

    def get_sample(self, assign_gt_inds, gt_bboxes):
        pos_inds = np.unique(np.nonzero(assign_gt_inds > 0)[0])
        neg_inds = np.unique(np.nonzero(assign_gt_inds == 0)[0])
        pos_assigned_gt_inds = assign_gt_inds[pos_inds] - 1

        if gt_bboxes.size == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.size == 0
            pos_gt_bboxes = np.empty_like(gt_bboxes).reshape(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.resize(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def __call__(self, samples, context=None):
        assert len(samples) > 0
        batch_size = len(samples)
        # get grid cells of image
        h, w = samples[0]['image'].shape[1:3]
        multi_level_grid_cells = []
        for stride in self.downsample_ratios:
            featmap_size = (int(math.ceil(h / stride)),
                            int(math.ceil(w / stride)))
            multi_level_grid_cells.append(
                self.get_grid_cells(featmap_size, self.grid_cell_scale, stride,
                                    self.cell_offset))
        mlvl_grid_cells_list = [
            multi_level_grid_cells for i in range(batch_size)
        ]
        # pixel cell number of multi-level feature maps
        num_level_cells = [
            grid_cells.shape[0] for grid_cells in mlvl_grid_cells_list[0]
        ]
        num_level_cells_list = [num_level_cells] * batch_size
        # concat all level cells and to a single array
        for i in range(batch_size):
            mlvl_grid_cells_list[i] = np.concatenate(mlvl_grid_cells_list[i])
        # target assign on all images
        for sample, grid_cells, num_level_cells in zip(
                samples, mlvl_grid_cells_list, num_level_cells_list):
            gt_bboxes = sample['gt_bbox']
            gt_labels = sample['gt_class'].squeeze()
            if gt_labels.size == 1:
                gt_labels = np.array([gt_labels]).astype(np.int32)
            gt_bboxes_ignore = None
            assign_gt_inds, _ = self.assigner(grid_cells, num_level_cells,
                                              gt_bboxes, gt_bboxes_ignore,
                                              gt_labels)
            pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.get_sample(
                assign_gt_inds, gt_bboxes)

            num_cells = grid_cells.shape[0]
            bbox_targets = np.zeros_like(grid_cells)
            bbox_weights = np.zeros_like(grid_cells)
            labels = np.ones([num_cells], dtype=np.int64) * self.num_classes
            label_weights = np.zeros([num_cells], dtype=np.float32)

            if len(pos_inds) > 0:
                pos_bbox_targets = pos_gt_bboxes
                bbox_targets[pos_inds, :] = pos_bbox_targets
                bbox_weights[pos_inds, :] = 1.0
                if not np.any(gt_labels):
                    labels[pos_inds] = 0
                else:
                    labels[pos_inds] = gt_labels[pos_assigned_gt_inds]

                label_weights[pos_inds] = 1.0
            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0
            sample['grid_cells'] = grid_cells
            sample['labels'] = labels
            sample['label_weights'] = label_weights
            sample['bbox_targets'] = bbox_targets
            sample['pos_num'] = max(pos_inds.size, 1)
            sample.pop('is_crowd', None)
            sample.pop('difficult', None)
            sample.pop('gt_class', None)
            sample.pop('gt_bbox', None)
            sample.pop('gt_score', None)
        return samples


@register_op
class Gt2TTFTarget(BaseOperator):
    __shared__ = ['num_classes']
    """
    Gt2TTFTarget
    Generate TTFNet targets by ground truth data
    
    Args:
        num_classes(int): the number of classes.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        alpha(float): the alpha parameter to generate gaussian target.
            0.54 by default.
    """

    def __init__(self, num_classes=80, down_ratio=4, alpha=0.54):
        super(Gt2TTFTarget, self).__init__()
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, samples, context=None):
        output_size = samples[0]['image'].shape[1]
        feat_size = output_size // self.down_ratio
        for sample in samples:
            heatmap = np.zeros(
                (self.num_classes, feat_size, feat_size), dtype='float32')
            box_target = np.ones(
                (4, feat_size, feat_size), dtype='float32') * -1
            reg_weight = np.zeros((1, feat_size, feat_size), dtype='float32')

            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']

            bbox_w = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
            bbox_h = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
            area = bbox_w * bbox_h
            boxes_areas_log = np.log(area)
            boxes_ind = np.argsort(boxes_areas_log, axis=0)[::-1]
            boxes_area_topk_log = boxes_areas_log[boxes_ind]
            gt_bbox = gt_bbox[boxes_ind]
            gt_class = gt_class[boxes_ind]

            feat_gt_bbox = gt_bbox / self.down_ratio
            feat_gt_bbox = np.clip(feat_gt_bbox, 0, feat_size - 1)
            feat_hs, feat_ws = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1],
                                feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0])

            ct_inds = np.stack(
                [(gt_bbox[:, 0] + gt_bbox[:, 2]) / 2,
                 (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2],
                axis=1) / self.down_ratio

            h_radiuses_alpha = (feat_hs / 2. * self.alpha).astype('int32')
            w_radiuses_alpha = (feat_ws / 2. * self.alpha).astype('int32')

            for k in range(len(gt_bbox)):
                cls_id = gt_class[k]
                fake_heatmap = np.zeros((feat_size, feat_size), dtype='float32')
                self.draw_truncate_gaussian(fake_heatmap, ct_inds[k],
                                            h_radiuses_alpha[k],
                                            w_radiuses_alpha[k])

                heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)
                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = gt_bbox[k][:, None]

                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = np.sum(local_heatmap)
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[0, box_target_inds] = local_heatmap / ct_div
            sample['ttf_heatmap'] = heatmap
            sample['ttf_box_target'] = box_target
            sample['ttf_reg_weight'] = reg_weight
            sample.pop('is_crowd', None)
            sample.pop('difficult', None)
            sample.pop('gt_class', None)
            sample.pop('gt_bbox', None)
            sample.pop('gt_score', None)
        return samples

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = gaussian2D((h, w), sigma_x, sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                   left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(
                masked_heatmap, masked_gaussian)
        return heatmap


@register_op
class Gt2Solov2Target(BaseOperator):
    """Assign mask target and labels in SOLOv2 network.
    The code of this function is based on:
        https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solov2_head.py#L271
    Args:
        num_grids (list): The list of feature map grids size.
        scale_ranges (list): The list of mask boundary range.
        coord_sigma (float): The coefficient of coordinate area length.
        sampling_ratio (float): The ratio of down sampling.
    """

    def __init__(self,
                 num_grids=[40, 36, 24, 16, 12],
                 scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768],
                               [384, 2048]],
                 coord_sigma=0.2,
                 sampling_ratio=4.0):
        super(Gt2Solov2Target, self).__init__()
        self.num_grids = num_grids
        self.scale_ranges = scale_ranges
        self.coord_sigma = coord_sigma
        self.sampling_ratio = sampling_ratio

    def _scale_size(self, im, scale):
        h, w = im.shape[:2]
        new_size = (int(w * float(scale) + 0.5), int(h * float(scale) + 0.5))
        resized_img = cv2.resize(
            im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return resized_img

    def __call__(self, samples, context=None):
        sample_id = 0
        max_ins_num = [0] * len(self.num_grids)
        for sample in samples:
            gt_bboxes_raw = sample['gt_bbox']
            gt_labels_raw = sample['gt_class'] + 1
            im_c, im_h, im_w = sample['image'].shape[:]
            gt_masks_raw = sample['gt_segm'].astype(np.uint8)
            mask_feat_size = [
                int(im_h / self.sampling_ratio), int(im_w / self.sampling_ratio)
            ]
            gt_areas = np.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                               (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
            ins_ind_label_list = []
            idx = 0
            for (lower_bound, upper_bound), num_grid \
                    in zip(self.scale_ranges, self.num_grids):

                hit_indices = ((gt_areas >= lower_bound) &
                               (gt_areas <= upper_bound)).nonzero()[0]
                num_ins = len(hit_indices)

                ins_label = []
                grid_order = []
                cate_label = np.zeros([num_grid, num_grid], dtype=np.int64)
                ins_ind_label = np.zeros([num_grid**2], dtype=np.bool)

                if num_ins == 0:
                    ins_label = np.zeros(
                        [1, mask_feat_size[0], mask_feat_size[1]],
                        dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        [sample_id * num_grid * num_grid + 0], dtype=np.int32)
                    idx += 1
                    continue
                gt_bboxes = gt_bboxes_raw[hit_indices]
                gt_labels = gt_labels_raw[hit_indices]
                gt_masks = gt_masks_raw[hit_indices, ...]

                half_ws = 0.5 * (
                    gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.coord_sigma
                half_hs = 0.5 * (
                    gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.coord_sigma

                for seg_mask, gt_label, half_h, half_w in zip(
                        gt_masks, gt_labels, half_hs, half_ws):
                    if seg_mask.sum() == 0:
                        continue
                    # mass center
                    upsampled_size = (mask_feat_size[0] * 4,
                                      mask_feat_size[1] * 4)
                    center_h, center_w = ndimage.measurements.center_of_mass(
                        seg_mask)
                    coord_w = int(
                        (center_w / upsampled_size[1]) // (1. / num_grid))
                    coord_h = int(
                        (center_h / upsampled_size[0]) // (1. / num_grid))

                    # left, top, right, down
                    top_box = max(0,
                                  int(((center_h - half_h) / upsampled_size[0])
                                      // (1. / num_grid)))
                    down_box = min(num_grid - 1,
                                   int(((center_h + half_h) / upsampled_size[0])
                                       // (1. / num_grid)))
                    left_box = max(0,
                                   int(((center_w - half_w) / upsampled_size[1])
                                       // (1. / num_grid)))
                    right_box = min(num_grid - 1,
                                    int(((center_w + half_w) /
                                         upsampled_size[1]) // (1. / num_grid)))

                    top = max(top_box, coord_h - 1)
                    down = min(down_box, coord_h + 1)
                    left = max(coord_w - 1, left_box)
                    right = min(right_box, coord_w + 1)

                    cate_label[top:(down + 1), left:(right + 1)] = gt_label
                    seg_mask = self._scale_size(
                        seg_mask, scale=1. / self.sampling_ratio)
                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            label = int(i * num_grid + j)
                            cur_ins_label = np.zeros(
                                [mask_feat_size[0], mask_feat_size[1]],
                                dtype=np.uint8)
                            cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[
                                1]] = seg_mask
                            ins_label.append(cur_ins_label)
                            ins_ind_label[label] = True
                            grid_order.append(sample_id * num_grid * num_grid +
                                              label)
                if ins_label == []:
                    ins_label = np.zeros(
                        [1, mask_feat_size[0], mask_feat_size[1]],
                        dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        [sample_id * num_grid * num_grid + 0], dtype=np.int32)
                else:
                    ins_label = np.stack(ins_label, axis=0)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        grid_order, dtype=np.int32)
                    assert len(grid_order) > 0
                max_ins_num[idx] = max(
                    max_ins_num[idx],
                    sample['ins_label{}'.format(idx)].shape[0])
                idx += 1
            ins_ind_labels = np.concatenate([
                ins_ind_labels_level_img
                for ins_ind_labels_level_img in ins_ind_label_list
            ])
            fg_num = np.sum(ins_ind_labels)
            sample['fg_num'] = fg_num
            sample_id += 1

            sample.pop('is_crowd')
            sample.pop('gt_class')
            sample.pop('gt_bbox')
            sample.pop('gt_poly')
            sample.pop('gt_segm')

        # padding batch
        for data in samples:
            for idx in range(len(self.num_grids)):
                gt_ins_data = np.zeros(
                    [
                        max_ins_num[idx],
                        data['ins_label{}'.format(idx)].shape[1],
                        data['ins_label{}'.format(idx)].shape[2]
                    ],
                    dtype=np.uint8)
                gt_ins_data[0:data['ins_label{}'.format(idx)].shape[
                    0], :, :] = data['ins_label{}'.format(idx)]
                gt_grid_order = np.zeros([max_ins_num[idx]], dtype=np.int32)
                gt_grid_order[0:data['grid_order{}'.format(idx)].shape[
                    0]] = data['grid_order{}'.format(idx)]
                data['ins_label{}'.format(idx)] = gt_ins_data
                data['grid_order{}'.format(idx)] = gt_grid_order

        return samples


@register_op
class Gt2SparseRCNNTarget(BaseOperator):
    '''
    Generate SparseRCNN targets by groud truth data
    '''

    def __init__(self):
        super(Gt2SparseRCNNTarget, self).__init__()

    def __call__(self, samples, context=None):
        for sample in samples:
            im = sample["image"]
            h, w = im.shape[1:3]
            img_whwh = np.array([w, h, w, h], dtype=np.int32)
            sample["img_whwh"] = img_whwh
            if "scale_factor" in sample:
                sample["scale_factor_wh"] = np.array(
                    [sample["scale_factor"][1], sample["scale_factor"][0]],
                    dtype=np.float32)
            else:
                sample["scale_factor_wh"] = np.array(
                    [1.0, 1.0], dtype=np.float32)

        return samples


@register_op
class PadMaskBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
        return_pad_mask (bool): If `return_pad_mask = True`, return
            `pad_mask` for transformer.
    """

    def __init__(self, pad_to_stride=0, return_pad_mask=False):
        super(PadMaskBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.return_pad_mask = return_pad_mask

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm
            if self.return_pad_mask:
                padding_mask = np.zeros(
                    (max_shape[1], max_shape[2]), dtype=np.float32)
                padding_mask[:im_h, :im_w] = 1.
                data['pad_mask'] = padding_mask

            if 'gt_rbox2poly' in data and data['gt_rbox2poly'] is not None:
                # ploy to rbox
                polys = data['gt_rbox2poly']
                rbox = bbox_utils.poly2rbox(polys)
                data['gt_rbox'] = rbox

        return samples


@register_op
class Gt2CenterNetTarget(BaseOperator):
    """Gt2CenterNetTarget
    Genterate CenterNet targets by ground-truth
    Args:
        down_ratio (int): The down sample ratio between output feature and 
                          input image.
        num_classes (int): The number of classes, 80 by default.
        max_objs (int): The maximum objects detected, 128 by default.
    """

    def __init__(self, down_ratio, num_classes=80, max_objs=128):
        super(Gt2CenterNetTarget, self).__init__()
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.max_objs = max_objs

    def __call__(self, sample, context=None):
        input_h, input_w = sample['image'].shape[1:]
        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio
        num_classes = self.num_classes
        c = sample['center']
        s = sample['scale']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.int32)
        cat_spec_wh = np.zeros(
            (self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros(
            (self.max_objs, num_classes * 2), dtype=np.int32)

        trans_output = get_affine_transform(c, [s, s], 0, [output_w, output_h])

        gt_det = []
        for i, (bbox, cls) in enumerate(zip(gt_bbox, gt_class)):
            cls = int(cls)
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)), 0.7)
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls], ct_int, radius)
                wh[i] = 1. * w, 1. * h
                ind[i] = ct_int[1] * output_w + ct_int[0]
                reg[i] = ct - ct_int
                reg_mask[i] = 1
                cat_spec_wh[i, cls * 2:cls * 2 + 2] = wh[i]
                cat_spec_mask[i, cls * 2:cls * 2 + 2] = 1
                gt_det.append([
                    ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2,
                    1, cls
                ])

        sample.pop('gt_bbox', None)
        sample.pop('gt_class', None)
        sample.pop('center', None)
        sample.pop('scale', None)
        sample.pop('is_crowd', None)
        sample.pop('difficult', None)
        sample['heatmap'] = hm
        sample['index_mask'] = reg_mask
        sample['index'] = ind
        sample['size'] = wh
        sample['offset'] = reg
        return sample


@register_op
class PadGT(BaseOperator):
    """
    Pad 0 to `gt_class`, `gt_bbox`, `gt_score`...
    The num_max_boxes is the largest for batch.
    Args:
        return_gt_mask (bool): If true, return `pad_gt_mask`,
                                1 means bbox, 0 means no bbox.
    """

    def __init__(self, return_gt_mask=True):
        super(PadGT, self).__init__()
        self.return_gt_mask = return_gt_mask

    def __call__(self, samples, context=None):
        num_max_boxes = max([len(s['gt_bbox']) for s in samples])
        for sample in samples:
            if self.return_gt_mask:
                sample['pad_gt_mask'] = np.zeros(
                    (num_max_boxes, 1), dtype=np.float32)
            if num_max_boxes == 0:
                continue

            num_gt = len(sample['gt_bbox'])
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt] = sample['gt_class']
                pad_gt_bbox[:num_gt] = sample['gt_bbox']
            sample['gt_class'] = pad_gt_class
            sample['gt_bbox'] = pad_gt_bbox
            # pad_gt_mask
            if 'pad_gt_mask' in sample:
                sample['pad_gt_mask'][:num_gt] = 1
            # gt_score
            if 'gt_score' in sample:
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_score[:num_gt] = sample['gt_score']
                sample['gt_score'] = pad_gt_score
            if 'is_crowd' in sample:
                pad_is_crowd = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_is_crowd[:num_gt] = sample['is_crowd']
                sample['is_crowd'] = pad_is_crowd
            if 'difficult' in sample:
                pad_diff = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_diff[:num_gt] = sample['difficult']
                sample['difficult'] = pad_diff
        return samples
