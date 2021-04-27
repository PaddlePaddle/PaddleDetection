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

from numbers import Integral
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import os
import cv2
import numpy as np
import math
import copy

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'DecodeVideo', 'Resize_LetterBox', 'AugmentHSV', 'NormalizedBbox2PixelBbox', 
    'RandomAffine', 'BboxXYWH2XYXY',
    'Gt2JDETargetThres', 'Gt2JDETargetMax',
]


@register_op
class DecodeVideo(BaseOperator):
    def __init__(self):
        """ 
        Transform the video data to numpy format following the rgb format
        """
        super(DecodeVideo, self).__init__()

    def apply(self, sample, context=None):
        im = sample['image']
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im
        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warn(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warn(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        sample['img0_shape'] = sample['im_shape']
        return sample


@register_op
class Resize_LetterBox(BaseOperator):
    def __init__(self, target_size):
        """
        Resize image to target size, convert normalized xywh to pixel xyxy
        format ([x_center, y_center, width, height] -> [x0, y0, x1, y1]).
        Args:
            target_size (int|list): image target size.
        """
        super(Resize_LetterBox, self).__init__()
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
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio),
                     round(shape[0] * ratio))  # [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        img = cv2.resize(
            img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)  # padded rectangular
        return img, ratio, dw, dh

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
        sample['img0_shape'] = sample['im_shape']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        height, width = self.target_size
        img, ratio, padw, padh = self.apply_image(
            im, height=height, width=width)

        sample['image'] = img
        sample['im_shape'] = np.asarray(self.target_size, dtype=np.float32)
        sample['scale_factor'] = np.asarray([ratio, ratio], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], h, w, ratio,
                                                padw, padh)
        return sample


@register_op
class AugmentHSV(BaseOperator):
    def __init__(self, fraction=0.50, is_bgr=True):
        """ 
        Augment the SV channel of image data.
        Args:
            fraction (float): the fraction for augment 
            is_bgr (bool): whether the image is BGR mode
        """
        super(AugmentHSV, self).__init__()
        self.fraction = fraction
        self.is_bgr = is_bgr

    def apply(self, sample, context=None):
        img = sample['image']
        if self.is_bgr:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        a = (random.random() * 2 - 1) * self.fraction + 1
        S *= a
        if a > 1:
            np.clip(S, a_min=0, a_max=255, out=S)

        a = (random.random() * 2 - 1) * self.fraction + 1
        V *= a
        if a > 1:
            np.clip(V, a_min=0, a_max=255, out=V)

        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        if self.is_bgr:
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        else:
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)

        sample['image'] = img
        return sample


@register_op
class NormalizedBbox2PixelBbox(BaseOperator):
    """
    Transform the bounding box's coornidates which is in [0,1] to pixels.
    """

    def __init__(self):
        super(NormalizedBbox2PixelBbox, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        height, width = sample['image'].shape[:2]
        bbox[:, 0::2] = bbox[:, 0::2] * width
        bbox[:, 1::2] = bbox[:, 1::2] * height
        sample['gt_bbox'] = bbox
        return sample


@register_op
class RandomAffine(BaseOperator):
    def __init__(self,
                 degrees=(-5, 5),
                 translate=(0.10, 0.10),
                 scale=(0.50, 1.20),
                 shear=(-2, 2),
                 borderValue=(127.5, 127.5, 127.5)):
        """ 
        Transform the image data with random_affine
        Args:
            degrees (float): the fraction for augment 
            translate (bool): whether the image is BGR mode
            scale (bool): whether the image is BGR mode
            shear (bool): whether the image is BGR mode
            borderValue (bool): whether the image is BGR mode
        """
        super(RandomAffine, self).__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.borderValue = borderValue

    def apply(self, sample, context=None):
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        border = 0  # width of added border (optional)

        img = sample['image']
        height, width = img.shape[0], img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (self.degrees[1] - self.degrees[0]
                               ) + self.degrees[0]
        s = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        R[:2] = cv2.getRotationMatrix2D(
            angle=a, center=(width / 2, height / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (
            random.random() * 2 - 1
        ) * self.translate[0] * height + border  # x translation (pixels)
        T[1, 2] = (
            random.random() * 2 - 1
        ) * self.translate[1] * width + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() *
                            (self.shear[1] - self.shear[0]) + self.shear[0]) *
                           math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() *
                            (self.shear[1] - self.shear[0]) + self.shear[0]) *
                           math.pi / 180)  # y shear (deg)

        M = S @T @R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(
            img,
            M,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=self.borderValue)  # BGR order borderValue

        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            targets = sample['gt_bbox']
            n = targets.shape[0]
            points = targets.copy()
            area0 = (points[:, 2] - points[:, 0]) * (
                points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians)))**0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate(
                (x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            if sum(i) > 0:
                sample['gt_bbox'] = xy[i]
                sample['gt_class'] = sample['gt_class'][i]
                if 'difficult' in sample:
                    sample['difficult'] = sample['difficult'][i]
                if 'gt_ide' in sample:
                    sample['gt_ide'] = sample['gt_ide'][i]
                if 'is_crowd' in sample:
                    sample['is_crowd'] = sample['is_crowd'][i]
                sample['image'] = imw
                return sample
            else:
                return sample


@register_op
class BboxXYWH2XYXY(BaseOperator):
    """
    Convert bbox XYWH format to XYXY format.
    [x_center, y_center, width, height] -> [x0, y0, x1, y1]
    """

    def __init__(self):
        super(BboxXYWH2XYXY, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox0 = sample['gt_bbox']
        bbox = bbox0.copy()

        bbox[:, :2] = bbox0[:, :2] - bbox0[:, 2:4] / 2.
        bbox[:, 2:4] = bbox0[:, :2] + bbox0[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
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

    def bbox_iou(self, box1, box2, x1y1x2y2=False, eps=1e-16):
        # box1: anchor, box2: gt_bbox. N>>M
        N, M = len(box1), len(box2)
        if x1y1x2y2:
            b1_x1, b1_y1 = box1[:, 0], box1[:, 1]
            b1_x2, b1_y2 = box1[:, 2], box1[:, 3]
            b2_x1, b2_y1 = box2[:, 0], box2[:, 1]
            b2_x2, b2_y2 = box2[:, 2], box2[:, 3]
        else:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:,
                                                                          2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:,
                                                                          3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:,
                                                                          2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:,
                                                                          3] / 2

        # get the coordinates of the intersection rectangle
        inter_rect_x1 = np.zeros((N, M), dtype=np.float32)
        inter_rect_y1 = np.zeros((N, M), dtype=np.float32)
        inter_rect_x2 = np.zeros((N, M), dtype=np.float32)
        inter_rect_y2 = np.zeros((N, M), dtype=np.float32)
        for i in range(len(box2)):
            inter_rect_x1[:, i] = np.maximum(b1_x1, b2_x1[i])
            inter_rect_y1[:, i] = np.maximum(b1_y1, b2_y1[i])
            inter_rect_x2[:, i] = np.minimum(b1_x2, b2_x2[i])
            inter_rect_y2[:, i] = np.minimum(b1_y2, b2_y2[i])
        # Intersection area
        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(
            inter_rect_y2 - inter_rect_y1, 0)
        # Union Area
        b1_area = np.repeat(
            ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).reshape(-1, 1), M, axis=-1)
        b2_area = np.repeat(
            ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).reshape(1, -1), N, axis=0)

        return inter_area / (b1_area + b2_area - inter_area + eps)

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
                iou_pdist = self.bbox_iou(anchor_list, tboxes)

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
