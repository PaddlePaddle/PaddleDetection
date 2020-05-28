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

import os
import numpy as np

import json
import io

from ppdet.core.workspace import register, serializable

from .dataset import DataSet
import logging
logger = logging.getLogger(__name__)


@register
@serializable
class TrafficDataSet(DataSet):
    """
    Load dataset with Traffic format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether use the default mapping of
            label to integer index. Default True.
        with_background (bool): whether load background as a class,
            default True.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=True,
                 with_background=True):
        super(TrafficDataSet, self).__init__(
            image_dir=image_dir,
            anno_path=anno_path,
            sample_num=sample_num,
            dataset_dir=dataset_dir,
            with_background=with_background)

        self.roidbs = None
        # 'cname2id' is a dict to map category name to class id
        self.cname2cid = None
        self.use_default_label = use_default_label
        self.load_image_only = False

    def load_roidb_and_cname2cid(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        records = []
        ct = 0
        #im_info_dict = { 
        #       pic_id: {
        #                'im_file': img_file,
        #                'im_id': im_id,
        #                'gt_class': gt_class,
        #                'gt_score': gt_score,
        #                'gt_bbox': gt_bbox,
        #               }
        # }
        im_info_dict = {}
        cname2cid = trans_label(self.with_background)

        for file in os.listdir(anno_path):
            anno_file = os.path.join(anno_path, file)
            with open(anno_file, 'r') as fr:
                jsonfile = json.load(fr)
                if 'signs' not in jsonfile.keys():
                    self.load_image_only = True
                    groups = jsonfile['group']
                    for group in groups:
                        pic_list = group['pic_list']
                        for pic_id in pic_list:
                            img_file = os.path.join(image_dir,
                                                    str(pic_id) + '.jpg')
                            if not os.path.exists(img_file):
                                logger.warn(
                                    'Illegal image file: {}, and it will be ignored'.
                                    format(img_file))
                                continue
                            im_id = np.array([ct])
                            im_info_dict[pic_id] = {
                                'im_file': img_file,
                                'im_id': im_id,
                                'group_id': str(file)
                            }
                            ct += 1
                else:
                    anno_list = jsonfile['signs']
                    for obj in anno_list:
                        img_file = os.path.join(image_dir,
                                                str(obj['pic_id']) + '.jpg')
                        pic_id = obj['pic_id']
                        if not os.path.exists(img_file):
                            logger.warn(
                                'Illegal image file: {}, and it will be ignored'.
                                format(img_file))
                            continue

                        x1 = float(obj['x'])
                        y1 = float(obj['y'])
                        x2 = float(obj['x']) + float(obj['w'])
                        y2 = float(obj['y']) + float(obj['h'])
                        x1 = max(0, x1)
                        y1 = max(0, y1)

                        gt_bbox = [x1, y1, x2, y2]
                        gt_class = [cname2cid[obj['type']]]
                        gt_score = [1.]
                        if pic_id not in im_info_dict.keys():
                            im_id = np.array([ct])
                            im_info_dict[pic_id] = {
                                'im_file': img_file,
                                'im_id': im_id,
                                'group_id': str(file),
                                'gt_class': [gt_class],
                                'gt_score': [gt_score],
                                'gt_bbox': [gt_bbox],
                            }
                            ct += 1
                        else:
                            im_info_dict[pic_id]['gt_bbox'].append(gt_bbox)
                            im_info_dict[pic_id]['gt_class'].append(gt_class)
                            im_info_dict[pic_id]['gt_score'].append(gt_score)

        write_im_info(self.dataset_dir, im_info_dict)

        for traffic_rec in im_info_dict.values():
            if not self.load_image_only:
                traffic_rec['gt_bbox'] = np.array(traffic_rec[
                    'gt_bbox']).astype('float32')
                traffic_rec['gt_class'] = np.array(traffic_rec[
                    'gt_class']).astype('int32')
                traffic_rec['gt_score'] = np.array(traffic_rec[
                    'gt_score']).astype('float32')
                traffic_rec['is_crowd'] = np.zeros(
                    (len(traffic_rec['gt_class']), 1)).astype('int32')

            records.append(traffic_rec)

        assert len(records) > 0, 'not found any traffic record in %s' % (
            self.anno_path)
        logger.info('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid


def trans_label(with_background=True):
    labels_map = {
        '102': 1,
        '103': 2,
        '104': 3,
        '105': 4,
        '106': 5,
        '107': 6,
        '108': 7,
        '109': 8,
        '110': 9,
        '111': 10,
        '112': 11,
        '201': 12,
        '202': 13,
        '203': 14,
        '204': 15,
        '205': 16,
        '206': 17,
        '207': 18,
        '301': 19,
    }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    else:
        labels_map.update({'background': 0})
    return labels_map


def write_im_info(out_dir, im_info_dict):
    txt_file = os.path.join(out_dir, 'data_info.txt')
    with open(txt_file, 'w') as f:
        for pic_id, traffic_rec in im_info_dict.items():
            f.write("{} {} {}\n".format(traffic_rec['group_id'], pic_id,
                                        traffic_rec['im_id'][0]))
