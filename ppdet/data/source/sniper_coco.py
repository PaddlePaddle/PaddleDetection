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

import os
import cv2
import json
import copy
import numpy as np

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from ppdet.core.workspace import register, serializable
from ppdet.data.crop_utils.annotation_cropper import AnnoCropper
from .coco import COCODataSet
from .dataset import _make_dataset, _is_valid_file
from ppdet.utils.logger import setup_logger

logger = setup_logger('sniper_coco_dataset')


@register
@serializable
class SniperCOCODataSet(COCODataSet):
    """SniperCOCODataSet"""

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 proposals_file=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=True,
                 empty_ratio=1.,
                 is_trainset=True,
                 image_target_sizes=[2000, 1000],
                 valid_box_ratio_ranges=[[-1, 0.1],[0.08, -1]],
                 chip_target_size=500,
                 chip_target_stride=200,
                 use_neg_chip=False,
                 max_neg_num_per_im=8,
                 max_per_img=-1,
                 nms_thresh=0.5):
        super(SniperCOCODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            load_crowd=load_crowd,
            allow_empty=allow_empty,
            empty_ratio=empty_ratio
        )
        self.proposals_file = proposals_file
        self.proposals = None
        self.anno_cropper = None
        self.is_trainset = is_trainset
        self.image_target_sizes = image_target_sizes
        self.valid_box_ratio_ranges = valid_box_ratio_ranges
        self.chip_target_size = chip_target_size
        self.chip_target_stride = chip_target_stride
        self.use_neg_chip = use_neg_chip
        self.max_neg_num_per_im = max_neg_num_per_im
        self.max_per_img = max_per_img
        self.nms_thresh = nms_thresh


    def parse_dataset(self):
        if not hasattr(self, "roidbs"):
            super(SniperCOCODataSet, self).parse_dataset()
        if self.is_trainset:
            self._parse_proposals()
            self._merge_anno_proposals()
        self.ori_roidbs = copy.deepcopy(self.roidbs)
        self.init_anno_cropper()
        self.roidbs = self.generate_chips_roidbs(self.roidbs, self.is_trainset)

    def set_proposals_file(self, file_path):
        self.proposals_file = file_path

    def init_anno_cropper(self):
        logger.info("Init AnnoCropper...")
        self.anno_cropper = AnnoCropper(
            image_target_sizes=self.image_target_sizes,
            valid_box_ratio_ranges=self.valid_box_ratio_ranges,
            chip_target_size=self.chip_target_size,
            chip_target_stride=self.chip_target_stride,
            use_neg_chip=self.use_neg_chip,
            max_neg_num_per_im=self.max_neg_num_per_im,
            max_per_img=self.max_per_img,
            nms_thresh=self.nms_thresh
        )

    def generate_chips_roidbs(self, roidbs, is_trainset):
        if is_trainset:
            roidbs = self.anno_cropper.crop_anno_records(roidbs)
        else:
            roidbs = self.anno_cropper.crop_infer_anno_records(roidbs)
        return roidbs

    def _parse_proposals(self):
        if self.proposals_file:
            self.proposals = {}
            logger.info("Parse proposals file:{}".format(self.proposals_file))
            with open(self.proposals_file, 'r') as f:
                proposals = json.load(f)
            for prop in proposals:
                image_id = prop["image_id"]
                if image_id not in self.proposals:
                    self.proposals[image_id] = []
                x, y, w, h = prop["bbox"]
                self.proposals[image_id].append([x, y, x + w, y + h])

    def _merge_anno_proposals(self):
        assert self.roidbs
        if self.proposals and len(self.proposals.keys()) > 0:
            logger.info("merge proposals to annos")
            for id, record in enumerate(self.roidbs):
                image_id = int(record["im_id"])
                if image_id not in self.proposals.keys():
                    logger.info("image id :{} no proposals".format(image_id))
                record["proposals"] = np.array(self.proposals.get(image_id, []), dtype=np.float32)
                self.roidbs[id] = record

    def get_ori_roidbs(self):
        if not hasattr(self, "ori_roidbs"):
            return None
        return self.ori_roidbs

    def get_roidbs(self):
        if not hasattr(self, "roidbs"):
            self.parse_dataset()
        return self.roidbs

    def set_roidbs(self, roidbs):
        self.roidbs = roidbs

    def check_or_download_dataset(self):
        return

    def _parse(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._parse()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            im = cv2.imread(image)
            h, w, c = im.shape
            rec = {'im_id': np.array([ct]), 'im_file': image, "h": h, "w": w}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images):
        self._imid2path = {}
        self.image_dir = images
        self.roidbs = self._load_images()

