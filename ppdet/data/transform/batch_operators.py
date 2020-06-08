# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
import cv2
import numpy as np

from .operators import register_op, BaseOperator
from .op_helper import jaccard_overlap

logger = logging.getLogger(__name__)

__all__ = [
    'PadBatch', 'RandomShape', 'PadMultiScaleTest', 'Gt2YoloTarget',
    'Gt2FCOSTarget'
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

    def __init__(self,
                 pad_to_stride=0,
                 use_padded_im_info=True,
                 pad_gt=False,
                 pad_mask=False):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info
        self.pad_gt = pad_gt
        self.pad_mask = pad_mask

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]

        if self.pad_gt:
            gt_num = []
            if self.pad_mask:
                poly_num = []
                poly_part_num = []
                point_num = []
            for data in samples:
                gt_num.append(data['gt_bbox'].shape[0])
                if self.pad_mask:
                    poly_num.append(len(data['gt_poly']))
                    for poly in data['gt_poly']:
                        #p_num = 0
                        #for p in poly:
                        #    p_num += len(p)
                        #point_num.append(int(p_num / 2))
                        poly_part_num.append(int(len(poly)))
                        for p_p in poly:
                            point_num.append(int(len(p_p) / 2))
            gt_num_max = max(gt_num)
            gt_box_data = np.zeros([gt_num_max, 4])
            gt_class_data = np.zeros([gt_num_max])
            is_crowd_data = np.ones([gt_num_max])

            if self.pad_mask:
                poly_num_max = max(poly_num)
                poly_part_num_max = max(poly_part_num)
                point_num_max = max(point_num)
                gt_masks_data = -np.ones(
                    [poly_num_max, poly_part_num_max, point_num_max, 2])

            for i, data in enumerate(samples):
                gt_num = data['gt_bbox'].shape[0]
                gt_box_data[0:gt_num, :] = data['gt_bbox']
                gt_class_data[0:gt_num] = np.squeeze(data['gt_class'])
                is_crowd_data[0:gt_num] = np.squeeze(data['is_crowd'])
                if self.pad_mask:
                    for j, poly in enumerate(data['gt_poly']):
                        #if len(poly) > 1:
                        #    one_poly = []
                        #    for p in poly:
                        #        one_poly.extend(p)
                        #    poly = one_poly
                        #poly_np = np.array(poly).reshape(-1, 2)
                        #gt_masks_data[j, :poly_np.shape[0], :] = poly_np
                        for k, p_p in enumerate(poly):
                            pp_np = np.array(p_p).reshape(-1, 2)
                            gt_masks_data[j, k, :pp_np.shape[0], :] = pp_np
                    data['gt_poly'] = gt_masks_data
                data['gt_bbox'] = gt_box_data
                data['gt_class'] = gt_class_data
                data['is_crowd_data'] = is_crowd_data
        return samples


@register_op
class RandomShape(BaseOperator):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.

    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[], random_inter=False):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []

    def __call__(self, samples, context=None):
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i]['image'] = im
        return samples


@register_op
class PadMultiScaleTest(BaseOperator):
    """
    Pad the image so they can be divisible by a stride for multi-scale testing.
 
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadMultiScaleTest, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples

        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        if len(samples) != 1:
            raise ValueError("Batch size must be 1 when using multiscale test, "
                             "but now batch size is {}".format(len(samples)))
        for i in range(len(samples)):
            sample = samples[i]
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im_c, im_h, im_w = im.shape
                    max_h = int(
                        np.ceil(im_h / coarsest_stride) * coarsest_stride)
                    max_w = int(
                        np.ceil(im_w / coarsest_stride) * coarsest_stride)
                    padding_im = np.zeros(
                        (im_c, max_h, max_w), dtype=np.float32)

                    padding_im[:, :im_h, :im_w] = im
                    sample[k] = padding_im
                    info_name = 'im_info' if k == 'image' else 'im_info_' + k
                    # update im_info
                    sample[info_name][:2] = [max_h, max_w]
        if not batch_input:
            samples = samples[0]
        return samples


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
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
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
                            if iou > self.iou_thresh:
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
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            im_info = sample['im_info']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                np.floor(im_info[1] / im_info[2])
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                np.floor(im_info[0] / im_info[2])
            # calculate the locations
            h, w = sample['image'].shape[1:3]
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
        return samples
