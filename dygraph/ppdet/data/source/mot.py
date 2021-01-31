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

import os
import numpy as np
from .dataset import DetDataset

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


def person_label(with_background=True):
    labels_map = {'person': 1, }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map


@register
@serializable
class MOTDataSet(DetDataset):
    """
    Load dataset with MOT format.
    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        sample_num (int): number of samples to load, -1 means all.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1):
        super(MOTDataSet, self).__init__(dataset_dir, image_dir, anno_path,
                                         data_fields, sample_num)

    def parse_dataset(self, with_background=True):
        #image_dir = os.path.join(self.dataset_dir, self.image_dir)
        anno_path = os.path.join(self.dataset_dir, self.anno_path)

        # mapping category name to class id
        # if with_background is True:
        #   background:0, first_class:1, second_class:2, ...
        # if with_background is False:
        #   first_class:0, second_class:1, ...
        records = []
        ct = 0
        cname2cid = person_label(with_background)

        with open(anno_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [
            x.replace('images', 'labels_with_ids').replace(
                '.png', '.txt').replace('.jpg', '.txt') for x in self.img_files
        ]
        self.nF = len(self.img_files)  # number of image files

        for i in range(self.nF):
            img_file = os.path.join(self.dataset_dir, self.img_files[i])
            lbl_file = os.path.join(self.dataset_dir, self.label_files[i])

            if not os.path.exists(img_file):
                logger.warn('Illegal image file: {}, and it will be ignored'.
                            format(img_file))
                continue
            if not os.path.isfile(lbl_file):
                logger.warn('Illegal label file: {}, and it will be ignored'.
                            format(lbl_file))
                continue

            # Load labels, normalized xywh to x1y1x2y2 format
            labels0 = np.loadtxt(lbl_file, dtype=np.float32).reshape(-1, 6)
            x1 = labels0[:, 2] - labels0[:, 4] / 2
            y1 = labels0[:, 3] - labels0[:, 5] / 2
            x2 = labels0[:, 2] + labels0[:, 4] / 2
            y2 = labels0[:, 3] + labels0[:, 5] / 2

            gt_norm_bbox = np.concatenate((x1, y1, x2, y2)).T
            gt_norm_bbox = np.array(gt_norm_bbox).astype('float32')
            gt_class = (labels0[:, 0] + 1).reshape(-1, 1).astype('int32')
            gt_id = np.array(labels0[:, 1]).astype('int32')
            gt_score = np.ones((len(labels0), 1)).astype('float32')

            mot_rec = {
                'im_file': img_file,
                'im_id': np.array([ct]),
            } if 'image' in self.data_fields else {}
            gt_rec = {
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_norm_bbox,
            }
            for k, v in gt_rec.items():
                if k in self.data_fields:
                    mot_rec[k] = v

            records.append(mot_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert len(records) > 0, 'not found any mot record in %s' % (
            self.anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid
