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


def load(anno_path,
         sample_num=-1,
         cname2cid=None,
         with_background=True):
    """
    Load WiderFace records with 'anno_path'

    Args:
        anno_path (str): root directory for voc annotation data
        sample_num (int): number of samples to load, -1 means all
        with_background (bool): whether load background as a class.
                                 if True, total class number will
                                 be 2. default True

    Returns:
        (records, catname2clsid)
        'records' is list of dict whose structure is:
        {
            'im_file': im_fname, # image file name
            'im_id': im_id, # image id
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
        }
        'cname2id' is a dict to map category name to class id
    """

    txt_file = anno_path

    records = []
    ct = 0
    file_lists = _load_file_list(txt_file)
    cname2cid = widerface_label(with_background)

    for item in file_lists:
        im_fname = item[0]
        im_id = np.array([ct])
        gt_bbox = np.zeros((len(item) - 2, 4), dtype=np.float32)
        gt_class = np.ones((len(item) - 2, 1), dtype=np.int32)
        for index_box in range(len(item)):
            if index_box >= 2:
                temp_info_box = item[index_box].split(' ')
                xmin = float(temp_info_box[0])
                ymin = float(temp_info_box[1])
                w = float(temp_info_box[2])
                h = float(temp_info_box[3])
                # Filter out wrong labels
                if w < 0 or h < 0:
                    continue
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = xmin + w
                ymax = ymin + h
                gt_bbox[index_box - 2] = [xmin, ymin, xmax, ymax]

        widerface_rec = {
            'im_file': im_fname,
            'im_id': im_id,
            'gt_bbox': gt_bbox,
            'gt_class': gt_class,
        }
        # logger.debug
        if len(item) != 0:
            records.append(widerface_rec)

        ct += 1
        if sample_num > 0 and ct >= sample_num:
            break
    assert len(records) > 0, 'not found any widerface in %s' % (anno_path)
    logger.info('{} samples in file {}'.format(ct, anno_path))
    return records, cname2cid


def _load_file_list(input_txt):
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
            if len(line_txt) > 6:
                split_str = line_txt.split(' ')
                x1_min = float(split_str[0])
                y1_min = float(split_str[1])
                x2_max = float(split_str[2])
                y2_max = float(split_str[3])
                line_txt = str(x1_min) + ' ' + str(y1_min) + ' ' + str(
                    x2_max) + ' ' + str(y2_max)
                file_dict[num_class].append(line_txt)
            else:
                file_dict[num_class].append(line_txt)

    return list(file_dict.values())


def widerface_label(with_background=True):
    labels_map = {
            'face': 1
    }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
