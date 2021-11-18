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

import copy
import math
import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from collections import defaultdict

from .chip_box_utils import nms, transform_chip_boxes2image_boxes
from .chip_box_utils import find_chips_to_cover_overlaped_boxes
from .chip_box_utils import transform_chip_box
from .chip_box_utils import intersection_over_box


class AnnoCropper(object):
    def __init__(self, image_target_sizes: List[int],
                 valid_box_ratio_ranges: List[List[float]],
                 chip_target_size: int, chip_target_stride: int,
                 use_neg_chip: bool = False,
                 max_neg_num_per_im: int = 8,
                 max_per_img: int = -1,
                 nms_thresh: int = 0.5
                 ):
        """
        Generate chips by chip_target_size and chip_target_stride.
        These two parameters just like kernel_size and stride in cnn.

        Each image has its raw size. After resizing, then get its target size.
        The resizing scale = target_size / raw_size.
        So are chips of the image.
        box_ratio = box_raw_size / image_raw_size = box_target_size / image_target_size
        The 'size' above mentioned is the size of long-side of image, box or chip.

        :param image_target_sizes: [2000, 1000]
        :param valid_box_ratio_ranges:  [[-1, 0.1],[0.08, -1]]
        :param chip_target_size: 500
        :param chip_target_stride: 200
        """
        self.target_sizes = image_target_sizes
        self.valid_box_ratio_ranges = valid_box_ratio_ranges
        assert len(self.target_sizes) == len(self.valid_box_ratio_ranges)
        self.scale_num = len(self.target_sizes)
        self.chip_target_size = chip_target_size  # is target size
        self.chip_target_stride = chip_target_stride  # is target stride
        self.use_neg_chip = use_neg_chip
        self.max_neg_num_per_im = max_neg_num_per_im
        self.max_per_img = max_per_img
        self.nms_thresh = nms_thresh

    def crop_anno_records(self, records: List[dict]):
        """
        The main logic:
        # foreach record(image):
        #   foreach scale:
        #     1 generate chips by chip size and stride for each scale
        #     2 get pos chips
        #     - validate boxes: current scale; h,w >= 1
        #     - find pos chips greedily by valid gt boxes in each scale
        #     - for every valid gt box, find its corresponding pos chips in each scale
        #     3 get neg chips
        #     - If given proposals, find neg boxes in them which are not in pos chips
        #     - If got neg boxes in last step, we find neg chips and assign neg boxes to neg chips such as 2.
        # 4 sample neg chips if too much each image
        #   transform this image-scale annotations to chips(pos chips&neg chips) annotations

        :param records, standard coco_record but with extra key `proposals`(Px4), which are predicted by stage1
                        model and maybe have neg boxes in them.
        :return: new_records, list of dict like
        {
            'im_file': 'fake_image1.jpg',
            'im_id': np.array([1]),  # new _global_chip_id as im_id
            'h': h,  # chip height
            'w': w,  # chip width
            'is_crowd': is_crowd,  # Nx1 -> Mx1
            'gt_class': gt_class,  # Nx1 -> Mx1
            'gt_bbox': gt_bbox,  # Nx4 -> Mx4, 4 represents [x1,y1,x2,y2]
            'gt_poly': gt_poly,  # [None]xN -> [None]xM
            'chip': [x1, y1, x2, y2]  # added
        }

        Attention:
        ------------------------------>x
        |
        |    (x1,y1)------
        |       |        |
        |       |        |
        |       |        |
        |       |        |
        |       |        |
        |       ----------
        |                 (x2,y2)
        |
        ↓
        y

        If we use [x1, y1, x2, y2] to represent boxes or chips,
        (x1,y1) is the left-top point which is in the box,
        but (x2,y2) is the right-bottom point which is not in the box.
        So x1 in [0, w-1], x2 in [1, w], y1 in [0, h-1], y2 in [1,h].
        And you can use x2-x1 to get width, and you can use image[y1:y2, x1:x2] to get the box area.
        """

        self.chip_records = []
        self._global_chip_id = 1
        for r in records:
            self._cur_im_pos_chips = []  # element: (chip, boxes_idx), chip is [x1, y1, x2, y2], boxes_ids is List[int]
            self._cur_im_neg_chips = []  # element: (chip, neg_box_num)
            for scale_i in range(self.scale_num):
                self._get_current_scale_parameters(scale_i, r)

                # Cx4
                chips = self._create_chips(r['h'], r['w'], self._cur_scale)

                # # dict: chipid->[box_id, ...]
                pos_chip2boxes_idx = self._get_valid_boxes_and_pos_chips(r['gt_bbox'], chips)

                # dict: chipid->neg_box_num
                neg_chip2box_num = self._get_neg_boxes_and_chips(chips, list(pos_chip2boxes_idx.keys()), r.get('proposals', None))

                self._add_to_cur_im_chips(chips, pos_chip2boxes_idx, neg_chip2box_num)

            cur_image_records = self._trans_all_chips2annotations(r)
            self.chip_records.extend(cur_image_records)
        return self.chip_records

    def _add_to_cur_im_chips(self, chips, pos_chip2boxes_idx, neg_chip2box_num):
        for pos_chipid, boxes_idx in pos_chip2boxes_idx.items():
            chip = np.array(chips[pos_chipid])  # copy chips slice
            self._cur_im_pos_chips.append((chip, boxes_idx))

        if neg_chip2box_num is None:
            return

        for neg_chipid, neg_box_num in neg_chip2box_num.items():
            chip = np.array(chips[neg_chipid])
            self._cur_im_neg_chips.append((chip,  neg_box_num))

    def _trans_all_chips2annotations(self, r):
        gt_bbox = r['gt_bbox']
        im_file = r['im_file']
        is_crowd = r['is_crowd']
        gt_class = r['gt_class']
        # gt_poly = r['gt_poly']   # [None]xN
        # remaining keys: im_id, h, w
        chip_records = self._trans_pos_chips2annotations(im_file, gt_bbox, is_crowd, gt_class)

        if not self.use_neg_chip:
            return chip_records

        sampled_neg_chips = self._sample_neg_chips()
        neg_chip_records = self._trans_neg_chips2annotations(im_file, sampled_neg_chips)
        chip_records.extend(neg_chip_records)
        return chip_records

    def _trans_pos_chips2annotations(self, im_file, gt_bbox, is_crowd, gt_class):
        chip_records = []
        for chip, boxes_idx in self._cur_im_pos_chips:
            chip_bbox, final_boxes_idx = transform_chip_box(gt_bbox, boxes_idx, chip)
            x1, y1, x2, y2 = chip
            chip_h = y2 - y1
            chip_w = x2 - x1
            rec = {
                'im_file': im_file,
                'im_id': np.array([self._global_chip_id]),
                'h': chip_h,
                'w': chip_w,
                'gt_bbox': chip_bbox,
                'is_crowd': is_crowd[final_boxes_idx].copy(),
                'gt_class': gt_class[final_boxes_idx].copy(),
                # 'gt_poly': [None] * len(final_boxes_idx),
                'chip': chip
            }
            self._global_chip_id += 1
            chip_records.append(rec)
        return chip_records

    def _sample_neg_chips(self):
        pos_num = len(self._cur_im_pos_chips)
        neg_num = len(self._cur_im_neg_chips)
        sample_num = min(pos_num + 2, self.max_neg_num_per_im)
        assert sample_num >= 1
        if neg_num <= sample_num:
            return self._cur_im_neg_chips

        candidate_num = int(sample_num * 1.5)
        candidate_neg_chips = sorted(self._cur_im_neg_chips, key=lambda x: -x[1])[:candidate_num]
        random.shuffle(candidate_neg_chips)
        sampled_neg_chips = candidate_neg_chips[:sample_num]
        return sampled_neg_chips

    def _trans_neg_chips2annotations(self, im_file: str, sampled_neg_chips: List[Tuple]):
        chip_records = []
        for chip, neg_box_num in sampled_neg_chips:
            x1, y1, x2, y2 = chip
            chip_h = y2 - y1
            chip_w = x2 - x1
            rec = {
                'im_file': im_file,
                'im_id': np.array([self._global_chip_id]),
                'h': chip_h,
                'w': chip_w,
                'gt_bbox': np.zeros((0, 4), dtype=np.float32),
                'is_crowd': np.zeros((0, 1), dtype=np.int32),
                'gt_class': np.zeros((0, 1), dtype=np.int32),
                # 'gt_poly': [],
                'chip': chip
            }
            self._global_chip_id += 1
            chip_records.append(rec)
        return chip_records

    def _get_current_scale_parameters(self, scale_i, r):
        im_size = max(r['h'], r['w'])
        im_target_size = self.target_sizes[scale_i]
        self._cur_im_size, self._cur_im_target_size = im_size, im_target_size
        self._cur_scale = self._get_current_scale(im_target_size, im_size)
        self._cur_valid_ratio_range = self.valid_box_ratio_ranges[scale_i]

    def _get_current_scale(self, im_target_size, im_size):
        return im_target_size / im_size

    def _create_chips(self, h: int, w: int, scale: float):
        """
        Generate chips by chip_target_size and chip_target_stride.
        These two parameters just like kernel_size and stride in cnn.
        :return: chips, Cx4, xy in raw size dimension
        """
        chip_size = self.chip_target_size  # omit target for simplicity
        stride = self.chip_target_stride
        width = int(scale * w)
        height = int(scale * h)
        min_chip_location_diff = 20  # in target size

        assert chip_size >= stride
        chip_overlap = chip_size - stride
        if (width - chip_overlap) % stride > min_chip_location_diff:  # 不能被stride整除的部分比较大，则保留
            w_steps = max(1, int(math.ceil((width - chip_overlap) / stride)))
        else:  # 不能被stride整除的部分比较小，则丢弃
            w_steps = max(1, int(math.floor((width - chip_overlap) / stride)))
        if (height - chip_overlap) % stride > min_chip_location_diff:
            h_steps = max(1, int(math.ceil((height - chip_overlap) / stride)))
        else:
            h_steps = max(1, int(math.floor((height - chip_overlap) / stride)))

        chips = list()
        for j in range(h_steps):
            for i in range(w_steps):
                x1 = i * stride
                y1 = j * stride
                x2 = min(x1 + chip_size, width)
                y2 = min(y1 + chip_size, height)
                chips.append([x1, y1, x2, y2])

        # check  chip size
        for item in chips:
            if item[2] - item[0] > chip_size * 1.1 or item[3] - item[1] > chip_size * 1.1:
                raise ValueError(item)
        chips = np.array(chips, dtype=np.float)

        raw_size_chips = chips / scale
        return raw_size_chips

    def _get_valid_boxes_and_pos_chips(self, gt_bbox, chips):
        valid_ratio_range = self._cur_valid_ratio_range
        im_size = self._cur_im_size
        scale = self._cur_scale
        #   Nx4            N
        valid_boxes, valid_boxes_idx = self._validate_boxes(valid_ratio_range, im_size, gt_bbox, scale)
        # dict: chipid->[box_id, ...]
        pos_chip2boxes_idx = self._find_pos_chips(chips, valid_boxes, valid_boxes_idx)
        return pos_chip2boxes_idx

    def _validate_boxes(self, valid_ratio_range: List[float],
                        im_size: int,
                        gt_boxes: 'np.array of Nx4',
                        scale: float):
        """
        :return: valid_boxes: Nx4, valid_boxes_idx: N
        """
        ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
        hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
        maxs = np.maximum(ws, hs)
        box_ratio = maxs / im_size
        mins = np.minimum(ws, hs)
        target_mins = mins * scale

        low = valid_ratio_range[0] if valid_ratio_range[0] > 0 else 0
        high = valid_ratio_range[1] if valid_ratio_range[1] > 0 else np.finfo(np.float).max

        valid_boxes_idx = np.nonzero((low <= box_ratio) & (box_ratio < high) & (target_mins >= 2))[0]
        valid_boxes = gt_boxes[valid_boxes_idx]
        return valid_boxes, valid_boxes_idx

    def _find_pos_chips(self, chips: 'Cx4', valid_boxes: 'Bx4', valid_boxes_idx: 'B'):
        """
        :return: pos_chip2boxes_idx, dict: chipid->[box_id, ...]
        """
        iob = intersection_over_box(chips, valid_boxes)  # overlap, CxB

        iob_threshold_to_find_chips = 1.
        pos_chip_ids, _ = self._find_chips_to_cover_overlaped_boxes(iob, iob_threshold_to_find_chips)
        pos_chip_ids = set(pos_chip_ids)

        iob_threshold_to_assign_box = 0.5
        pos_chip2boxes_idx = self._assign_boxes_to_pos_chips(
            iob, iob_threshold_to_assign_box, pos_chip_ids, valid_boxes_idx)
        return pos_chip2boxes_idx

    def _find_chips_to_cover_overlaped_boxes(self, iob, overlap_threshold):
        return find_chips_to_cover_overlaped_boxes(iob, overlap_threshold)

    def _assign_boxes_to_pos_chips(self, iob, overlap_threshold, pos_chip_ids, valid_boxes_idx):
        chip_ids, box_ids = np.nonzero(iob >= overlap_threshold)
        pos_chip2boxes_idx = defaultdict(list)
        for chip_id, box_id in zip(chip_ids, box_ids):
            if chip_id not in pos_chip_ids:
                continue
            raw_gt_box_idx = valid_boxes_idx[box_id]
            pos_chip2boxes_idx[chip_id].append(raw_gt_box_idx)
        return pos_chip2boxes_idx

    def _get_neg_boxes_and_chips(self, chips: 'Cx4', pos_chip_ids: 'D', proposals: 'Px4'):
        """
        :param chips:
        :param pos_chip_ids:
        :param proposals:
        :return: neg_chip2box_num, None or dict: chipid->neg_box_num
        """
        if not self.use_neg_chip:
            return None

        # train proposals maybe None
        if proposals is None or len(proposals) < 1:
            return None

        valid_ratio_range = self._cur_valid_ratio_range
        im_size = self._cur_im_size
        scale = self._cur_scale

        valid_props, _ = self._validate_boxes(valid_ratio_range, im_size, proposals, scale)
        neg_boxes = self._find_neg_boxes(chips, pos_chip_ids, valid_props)
        neg_chip2box_num = self._find_neg_chips(chips, pos_chip_ids, neg_boxes)
        return neg_chip2box_num

    def _find_neg_boxes(self, chips: 'Cx4', pos_chip_ids: 'D', valid_props: 'Px4'):
        """
        :return: neg_boxes: Nx4
        """
        if len(pos_chip_ids) == 0:
            return valid_props

        pos_chips = chips[pos_chip_ids]
        iob = intersection_over_box(pos_chips, valid_props)
        overlap_per_prop = np.max(iob, axis=0)
        non_overlap_props_idx = overlap_per_prop < 0.5
        neg_boxes = valid_props[non_overlap_props_idx]
        return neg_boxes

    def _find_neg_chips(self, chips: 'Cx4', pos_chip_ids: 'D', neg_boxes: 'Nx4'):
        """
        :return: neg_chip2box_num, dict: chipid->neg_box_num
        """
        neg_chip_ids = np.setdiff1d(np.arange(len(chips)), pos_chip_ids)
        neg_chips = chips[neg_chip_ids]

        iob = intersection_over_box(neg_chips, neg_boxes)
        iob_threshold_to_find_chips = 0.7
        chosen_neg_chip_ids, chip_id2overlap_box_num = \
            self._find_chips_to_cover_overlaped_boxes(iob, iob_threshold_to_find_chips)

        neg_chipid2box_num = {}
        for cid in chosen_neg_chip_ids:
            box_num = chip_id2overlap_box_num[cid]
            raw_chip_id = neg_chip_ids[cid]
            neg_chipid2box_num[raw_chip_id] = box_num
        return neg_chipid2box_num

    def crop_infer_anno_records(self, records: List[dict]):
        """
        transform image record to chips record
        :param records:
        :return: new_records, list of dict like
        {
            'im_file': 'fake_image1.jpg',
            'im_id': np.array([1]),  # new _global_chip_id as im_id
            'h': h,  # chip height
            'w': w,  # chip width
            'chip': [x1, y1, x2, y2]  # added
            'ori_im_h': ori_im_h  # added, origin image height
            'ori_im_w': ori_im_w  # added, origin image width
            'scale_i': 0  # added,
        }
        """
        self.chip_records = []
        self._global_chip_id = 1  # im_id start from 1
        self._global_chip_id2img_id = {}

        for r in records:
            for scale_i in range(self.scale_num):
                self._get_current_scale_parameters(scale_i, r)
                # Cx4
                chips = self._create_chips(r['h'], r['w'], self._cur_scale)
                cur_img_chip_record = self._get_chips_records(r, chips, scale_i)
                self.chip_records.extend(cur_img_chip_record)

        return self.chip_records

    def _get_chips_records(self, rec, chips, scale_i):
        cur_img_chip_records = []
        ori_im_h = rec["h"]
        ori_im_w = rec["w"]
        im_file = rec["im_file"]
        ori_im_id = rec["im_id"]
        for id, chip in enumerate(chips):
            chip_rec = {}
            x1, y1, x2, y2 = chip
            chip_h = y2 - y1
            chip_w = x2 - x1
            chip_rec["im_file"] = im_file
            chip_rec["im_id"] = self._global_chip_id
            chip_rec["h"] = chip_h
            chip_rec["w"] = chip_w
            chip_rec["chip"] = chip
            chip_rec["ori_im_h"] = ori_im_h
            chip_rec["ori_im_w"] = ori_im_w
            chip_rec["scale_i"] = scale_i

            self._global_chip_id2img_id[self._global_chip_id] = int(ori_im_id)
            self._global_chip_id += 1
            cur_img_chip_records.append(chip_rec)

        return cur_img_chip_records

    def aggregate_chips_detections(self, results, records=None):
        """
        # 1. transform chip dets to image dets
        # 2. nms boxes per image;
        # 3. format output results
        :param results:
        :param roidb:
        :return:
        """
        results = deepcopy(results)
        records = records if records else self.chip_records
        img_id2bbox = self._transform_chip2image_bboxes(results, records)
        nms_img_id2bbox = self._nms_dets(img_id2bbox)
        aggregate_results = self._reformat_results(nms_img_id2bbox)
        return aggregate_results

    def _transform_chip2image_bboxes(self, results, records):
        # 1. Transform chip dets to image dets;
        # 2. Filter valid range;
        # 3. Reformat and Aggregate chip dets to Get scale_cls_dets
        img_id2bbox = defaultdict(list)
        for result in results:
            bbox_locs = result['bbox']
            bbox_nums = result['bbox_num']
            if len(bbox_locs) == 1 and bbox_locs[0][0] == -1:  # current batch has no detections
                # bbox_locs = array([[-1.]], dtype=float32); bbox_nums = [[1]]
                # MultiClassNMS output: If there is no detected boxes for all images, lod will be set to {1} and Out only contains one value which is -1.
                continue
            im_ids = result['im_id'] # replace with range(len(bbox_nums))

            last_bbox_num = 0
            for idx, im_id in enumerate(im_ids):

                cur_bbox_len = bbox_nums[idx]
                bboxes = bbox_locs[last_bbox_num: last_bbox_num + cur_bbox_len]
                last_bbox_num += cur_bbox_len
                # box: [num_id, score, xmin, ymin, xmax, ymax]
                if len(bboxes) == 0:  # current image has no detections
                    continue

                chip_rec = records[int(im_id) - 1]  # im_id starts from 1, type is np.int64
                image_size = max(chip_rec["ori_im_h"], chip_rec["ori_im_w"])

                bboxes = transform_chip_boxes2image_boxes(bboxes, chip_rec["chip"], chip_rec["ori_im_h"], chip_rec["ori_im_w"])

                scale_i = chip_rec["scale_i"]
                cur_scale = self._get_current_scale(self.target_sizes[scale_i], image_size)
                _, valid_boxes_idx = self._validate_boxes(self.valid_box_ratio_ranges[scale_i], image_size,
                                                                    bboxes[:, 2:], cur_scale)
                ori_img_id = self._global_chip_id2img_id[int(im_id)]

                img_id2bbox[ori_img_id].append(bboxes[valid_boxes_idx])

        return img_id2bbox

    def _nms_dets(self, img_id2bbox):
        # 1. NMS on each image-class
        # 2. Limit number of detections to MAX_PER_IMAGE if requested
        max_per_img = self.max_per_img
        nms_thresh = self.nms_thresh

        for img_id in img_id2bbox:
            box = img_id2bbox[img_id]  # list of np.array of shape [N, 6], 6 is [label, score, x1, y1, x2, y2]
            box = np.concatenate(box, axis=0)
            nms_dets = nms(box, nms_thresh)
            if max_per_img > 0:
                if len(nms_dets) > max_per_img:
                    keep = np.argsort(-nms_dets[:, 1])[:max_per_img]
                    nms_dets = nms_dets[keep]

            img_id2bbox[img_id] = nms_dets

        return img_id2bbox

    def _reformat_results(self, img_id2bbox):
        """reformat results"""
        im_ids = img_id2bbox.keys()
        results = []
        for img_id in im_ids:  # output by original im_id order
            if len(img_id2bbox[img_id]) == 0:
                bbox = np.array([[-1.,  0.,  0.,  0.,  0.,  0.]])  # edge case: no detections
                bbox_num = np.array([0])
            else:
                # np.array of shape [N, 6], 6 is [label, score, x1, y1, x2, y2]
                bbox = img_id2bbox[img_id]
                bbox_num = np.array([len(bbox)])
            res = dict(
                im_id=np.array([[img_id]]),
                bbox=bbox,
                bbox_num=bbox_num
            )
            results.append(res)
        return results


