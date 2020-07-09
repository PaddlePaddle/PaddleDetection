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
import logging
logger = logging.getLogger(__name__)

from ppdet.core.workspace import register, serializable
from .dataset import DataSet


@register
@serializable
class WIDERFaceDataSet(DataSet):
    """
    Load WiderFace records with 'anno_path'

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): root directory for voc annotation data
        sample_num (int): number of samples to load, -1 means all
        with_background (bool): whether load background as a class.
            if True, total class number will be 2. default True.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 with_background=True,
                 with_lmk=False):
        super(WIDERFaceDataSet, self).__init__(
            image_dir=image_dir,
            anno_path=anno_path,
            sample_num=sample_num,
            dataset_dir=dataset_dir,
            with_background=with_background)
        self.anno_path = anno_path
        self.sample_num = sample_num
        self.with_background = with_background
        self.roidbs = None
        self.cname2cid = None
        self.with_lmk = with_lmk

    def load_roidb_and_cname2cid(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        txt_file = anno_path

        records = []
        ct = 0
        file_lists = self._load_file_list(txt_file)
        cname2cid = widerface_label(self.with_background)

        for item in file_lists:
            im_fname = item[0]
            im_id = np.array([ct])
            gt_bbox = np.zeros((len(item) - 1, 4), dtype=np.float32)
            gt_class = np.ones((len(item) - 1, 1), dtype=np.int32)
            gt_lmk_labels = np.zeros((len(item) - 1, 10), dtype=np.float32)
            lmk_ignore_flag = np.zeros((len(item) - 1, 1), dtype=np.int32)
            for index_box in range(len(item)):
                if index_box < 1:
                    continue
                gt_bbox[index_box - 1] = item[index_box][0]
                if self.with_lmk:
                    gt_lmk_labels[index_box - 1] = item[index_box][1]
                    lmk_ignore_flag[index_box - 1] = item[index_box][2]
            im_fname = os.path.join(image_dir,
                                    im_fname) if image_dir else im_fname
            widerface_rec = {
                'im_file': im_fname,
                'im_id': im_id,
                'gt_bbox': gt_bbox,
                'gt_class': gt_class,
            }
            if self.with_lmk:
                widerface_rec['gt_keypoint'] = gt_lmk_labels
                widerface_rec['keypoint_ignore'] = lmk_ignore_flag

            if len(item) != 0:
                records.append(widerface_rec)

            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert len(records) > 0, 'not found any widerface in %s' % (anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid

    def _load_file_list(self, input_txt):
        with open(input_txt, 'r') as f_dir:
            lines_input_txt = f_dir.readlines()

        file_dict = {}
        num_class = 0
        for i in range(len(lines_input_txt)):
            line_txt = lines_input_txt[i].strip('\n\t\r')
            if '.jpg' in line_txt:
                if i != 0:
                    num_class += 1
                file_dict[num_class] = []
                file_dict[num_class].append(line_txt)
            if '.jpg' not in line_txt:
                if len(line_txt) <= 6:
                    continue
                result_boxs = []
                split_str = line_txt.split(' ')
                xmin = float(split_str[0])
                ymin = float(split_str[1])
                w = float(split_str[2])
                h = float(split_str[3])
                # Filter out wrong labels
                if w < 0 or h < 0:
                    logger.warn('Illegal box with w: {}, h: {} in '
                                'img: {}, and it will be ignored'.format(
                                    w, h, file_dict[num_class][0]))
                    continue
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = xmin + w
                ymax = ymin + h
                gt_bbox = [xmin, ymin, xmax, ymax]
                result_boxs.append(gt_bbox)
                if self.with_lmk:
                    assert len(split_str) > 18, 'When `with_lmk=True`, the number' \
                            'of characters per line in the annotation file should' \
                            'exceed 18.'
                    lmk0_x = float(split_str[5])
                    lmk0_y = float(split_str[6])
                    lmk1_x = float(split_str[8])
                    lmk1_y = float(split_str[9])
                    lmk2_x = float(split_str[11])
                    lmk2_y = float(split_str[12])
                    lmk3_x = float(split_str[14])
                    lmk3_y = float(split_str[15])
                    lmk4_x = float(split_str[17])
                    lmk4_y = float(split_str[18])
                    lmk_ignore_flag = 0 if lmk0_x == -1 else 1
                    gt_lmk_label = [
                        lmk0_x, lmk0_y, lmk1_x, lmk1_y, lmk2_x, lmk2_y, lmk3_x,
                        lmk3_y, lmk4_x, lmk4_y
                    ]
                    result_boxs.append(gt_lmk_label)
                    result_boxs.append(lmk_ignore_flag)
                file_dict[num_class].append(result_boxs)

        return list(file_dict.values())


def widerface_label(with_background=True):
    labels_map = {'face': 1}
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
