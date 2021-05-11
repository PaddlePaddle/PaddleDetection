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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from numbers import Integral

import cv2
import copy
import numpy as np

from .operators import BaseOperator, register_op
from ppdet.modeling.bbox_utils import bbox_iou_np_expand
from ppdet.core.workspace import serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['LetterBoxResize', 'Gt2JDETargetThres', 'Gt2JDETargetMax']


@register_op
class LetterBoxResize(BaseOperator):
    def __init__(self, target_size):
        """
        Resize image to target size, convert normalized xywh to pixel xyxy
        format ([x_center, y_center, width, height] -> [x0, y0, x1, y1]).
        Args:
            target_size (int|list): image target size.
        """
        super(LetterBoxResize, self).__init__()
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, img, height, width, color=(127.5, 127.5, 127.5)):
        # letterbox: resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # [height, width]
        ratio_h = float(height) / shape[0]
        ratio_w = float(width) / shape[1]
        ratio = min(ratio_h, ratio_w)
        new_shape = (round(shape[1] * ratio),
                     round(shape[0] * ratio))  # [width, height]
        padw = (width - new_shape[0]) / 2
        padh = (height - new_shape[1]) / 2
        top, bottom = round(padh - 0.1), round(padh + 0.1)
        left, right = round(padw - 0.1), round(padw + 0.1)

        img = cv2.resize(
            img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)  # padded rectangular
        return img, ratio, padw, padh

    def apply_bbox(self, bbox0, h, w, ratio, padw, padh):
        bboxes = bbox0.copy()
        bboxes[:, 0] = ratio * w * (bbox0[:, 0] - bbox0[:, 2] / 2) + padw
        bboxes[:, 1] = ratio * h * (bbox0[:, 1] - bbox0[:, 3] / 2) + padh
        bboxes[:, 2] = ratio * w * (bbox0[:, 0] + bbox0[:, 2] / 2) + padw
        bboxes[:, 3] = ratio * h * (bbox0[:, 1] + bbox0[:, 3] / 2) + padh
        return bboxes

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        h, w = sample['im_shape']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        height, width = self.target_size
        img, ratio, padw, padh = self.apply_image(
            im, height=height, width=width)

        sample['image'] = img
        new_shape = (round(h * ratio), round(w * ratio))
        sample['im_shape'] = np.asarray(new_shape, dtype=np.float32)
        sample['scale_factor'] = np.asarray([ratio, ratio], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], h, w, ratio,
                                                padw, padh)
        return sample


@register_op
class Gt2JDETargetThres(BaseOperator):
    __shared__ = ['num_classes']
    """
    Generate JDE targets by groud truth data when training
    Args:
        anchors (list): anchors of JDE model
        anchor_masks (list): anchor_masks of JDE model
        downsample_ratios (list): downsample ratios of JDE model
        ide_thresh (float): thresh of identity, higher is groud truth 
        fg_thresh (float): thresh of foreground, higher is foreground
        bg_thresh (float): thresh of background, lower is background
        num_classes (int): number of classes
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 ide_thresh=0.5,
                 fg_thresh=0.5,
                 bg_thresh=0.4,
                 num_classes=1):
        super(Gt2JDETargetThres, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.ide_thresh = ide_thresh
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.num_classes = num_classes

    def generate_anchor(self, nGh, nGw, anchor_hw):
        nA = len(anchor_hw)
        yy, xx = np.meshgrid(np.arange(nGh), np.arange(nGw))

        mesh = np.stack([xx.T, yy.T], axis=0)  # [2, nGh, nGw]
        mesh = np.repeat(mesh[None, :], nA, axis=0)  # [nA, 2, nGh, nGw]

        anchor_offset_mesh = anchor_hw[:, :, None][:, :, :, None]
        anchor_offset_mesh = np.repeat(anchor_offset_mesh, nGh, axis=-2)
        anchor_offset_mesh = np.repeat(anchor_offset_mesh, nGw, axis=-1)

        anchor_mesh = np.concatenate(
            [mesh, anchor_offset_mesh], axis=1)  # [nA, 4, nGh, nGw]
        return anchor_mesh

    def encode_delta(self, gt_box_list, fg_anchor_list):
        px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                        fg_anchor_list[:, 2], fg_anchor_list[:,3]
        gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                        gt_box_list[:, 2], gt_box_list[:, 3]
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = np.log(gw / pw)
        dh = np.log(gh / ph)
        return np.stack([dx, dy, dw, dh], axis=1)

    def pad_box(self, sample, num_max):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = len(bbox)
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_score' in sample:
            pad_score = np.zeros((num_max, ), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        if 'difficult' in sample:
            pad_diff = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        if 'is_crowd' in sample:
            pad_crowd = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_crowd[:gt_num] = sample['is_crowd'][:gt_num, 0]
            sample['is_crowd'] = pad_crowd
        if 'gt_ide' in sample:
            pad_ide = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_ide[:gt_num] = sample['gt_ide'][:gt_num, 0]
            sample['gt_ide'] = pad_ide
        return sample

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."
        h, w = samples[0]['image'].shape[1:3]

        num_max = 0
        for sample in samples:
            num_max = max(num_max, len(sample['gt_bbox']))

        for sample in samples:
            gt_bbox = sample['gt_bbox']
            gt_ide = sample['gt_ide']
            for i, (anchor_hw, downsample_ratio
                    ) in enumerate(zip(self.anchors, self.downsample_ratios)):
                anchor_hw = np.array(
                    anchor_hw, dtype=np.float32) / downsample_ratio
                nA = len(anchor_hw)
                nGh, nGw = int(h / downsample_ratio), int(w / downsample_ratio)
                tbox = np.zeros((nA, nGh, nGw, 4), dtype=np.float32)
                tconf = np.zeros((nA, nGh, nGw), dtype=np.float32)
                tid = -np.ones((nA, nGh, nGw, 1), dtype=np.float32)

                gxy, gwh = gt_bbox[:, 0:2].copy(), gt_bbox[:, 2:4].copy()
                gxy[:, 0] = gxy[:, 0] * nGw
                gxy[:, 1] = gxy[:, 1] * nGh
                gwh[:, 0] = gwh[:, 0] * nGw
                gwh[:, 1] = gwh[:, 1] * nGh
                gxy[:, 0] = np.clip(gxy[:, 0], 0, nGw - 1)
                gxy[:, 1] = np.clip(gxy[:, 1], 0, nGh - 1)
                tboxes = np.concatenate([gxy, gwh], axis=1)

                anchor_mesh = self.generate_anchor(nGh, nGw, anchor_hw)

                anchor_list = np.transpose(anchor_mesh,
                                           (0, 2, 3, 1)).reshape(-1, 4)
                iou_pdist = bbox_iou_np_expand(
                    anchor_list, tboxes, x1y1x2y2=False)

                iou_max = np.max(iou_pdist, axis=1)
                max_gt_index = np.argmax(iou_pdist, axis=1)

                iou_map = iou_max.reshape(nA, nGh, nGw)
                gt_index_map = max_gt_index.reshape(nA, nGh, nGw)

                id_index = iou_map > self.ide_thresh
                fg_index = iou_map > self.fg_thresh
                bg_index = iou_map < self.bg_thresh
                ign_index = (iou_map < self.fg_thresh) * (
                    iou_map > self.bg_thresh)
                tconf[fg_index] = 1
                tconf[bg_index] = 0
                tconf[ign_index] = -1

                gt_index = gt_index_map[fg_index]
                gt_box_list = tboxes[gt_index]
                gt_id_list = gt_ide[gt_index_map[id_index]]

                if np.sum(fg_index) > 0:
                    tid[id_index] = gt_id_list

                    fg_anchor_list = anchor_list.reshape(nA, nGh, nGw,
                                                         4)[fg_index]
                    delta_target = self.encode_delta(gt_box_list,
                                                     fg_anchor_list)
                    tbox[fg_index] = delta_target

                sample['tbox{}'.format(i)] = tbox
                sample['tconf{}'.format(i)] = tconf
                sample['tide{}'.format(i)] = tid
            sample.pop('gt_class')
            sample = self.pad_box(sample, num_max)
        return samples


@register_op
class Gt2JDETargetMax(BaseOperator):
    __shared__ = ['num_classes']
    """
    Generate JDE targets by groud truth data when evaluating
    Args:
        anchors (list): anchors of JDE model
        anchor_masks (list): anchor_masks of JDE model
        downsample_ratios (list): downsample ratios of JDE model
        max_iou_thresh (float): iou thresh for high quality anchor
        num_classes (int): number of classes
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 max_iou_thresh=0.60,
                 num_classes=1):
        super(Gt2JDETargetMax, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.max_iou_thresh = max_iou_thresh
        self.num_classes = num_classes

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."
        h, w = samples[0]['image'].shape[1:3]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            gt_ide = sample['gt_ide']
            for i, (anchor_hw, downsample_ratio
                    ) in enumerate(zip(self.anchors, self.downsample_ratios)):
                anchor_hw = np.array(
                    anchor_hw, dtype=np.float32) / downsample_ratio
                nA = len(anchor_hw)
                nGh, nGw = int(h / downsample_ratio), int(w / downsample_ratio)
                tbox = np.zeros((nA, nGh, nGw, 4), dtype=np.float32)
                tconf = np.zeros((nA, nGh, nGw), dtype=np.float32)
                tid = -np.ones((nA, nGh, nGw, 1), dtype=np.float32)

                gxy, gwh = gt_bbox[:, 0:2].copy(), gt_bbox[:, 2:4].copy()
                gxy[:, 0] = gxy[:, 0] * nGw
                gxy[:, 1] = gxy[:, 1] * nGh
                gwh[:, 0] = gwh[:, 0] * nGw
                gwh[:, 1] = gwh[:, 1] * nGh
                gi = np.clip(gxy[:, 0], 0, nGw - 1).astype(int)
                gj = np.clip(gxy[:, 1], 0, nGh - 1).astype(int)

                # iou of targets-anchors (using wh only)
                box1 = gwh
                box2 = anchor_hw[:, None, :]
                inter_area = np.minimum(box1, box2).prod(2)
                iou = inter_area / (
                    box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

                # Select best iou_pred and anchor
                iou_best = iou.max(0)  # best anchor [0-2] for each target
                a = np.argmax(iou, axis=0)

                # Select best unique target-anchor combinations
                iou_order = np.argsort(-iou_best)  # best to worst

                # Unique anchor selection
                u = np.stack((gi, gj, a), 0)[:, iou_order]
                _, first_unique = np.unique(u, axis=1, return_index=True)
                mask = iou_order[first_unique]
                # best anchor must share significant commonality (iou) with target
                # TODO: examine arbitrary threshold
                idx = mask[iou_best[mask] > self.max_iou_thresh]

                if len(idx) > 0:
                    a_i, gj_i, gi_i = a[idx], gj[idx], gi[idx]
                    t_box = gt_bbox[idx]
                    t_id = gt_ide[idx]
                    if len(t_box.shape) == 1:
                        t_box = t_box.reshape(1, 4)

                    gxy, gwh = t_box[:, 0:2].copy(), t_box[:, 2:4].copy()
                    gxy[:, 0] = gxy[:, 0] * nGw
                    gxy[:, 1] = gxy[:, 1] * nGh
                    gwh[:, 0] = gwh[:, 0] * nGw
                    gwh[:, 1] = gwh[:, 1] * nGh

                    # XY coordinates
                    tbox[:, :, :, 0:2][a_i, gj_i, gi_i] = gxy - gxy.astype(int)
                    # Width and height in yolo method
                    tbox[:, :, :, 2:4][a_i, gj_i, gi_i] = np.log(gwh /
                                                                 anchor_hw[a_i])
                    tconf[a_i, gj_i, gi_i] = 1
                    tid[a_i, gj_i, gi_i] = t_id

                sample['tbox{}'.format(i)] = tbox
                sample['tconf{}'.format(i)] = tconf
                sample['tide{}'.format(i)] = tid
