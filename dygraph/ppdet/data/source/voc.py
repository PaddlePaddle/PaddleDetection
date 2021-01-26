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

import xml.etree.ElementTree as ET

from ppdet.core.workspace import register, serializable

from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class VOCDataSet(DetDataset):
    """
    Load dataset with PascalVOC format.

    Notes:
    `anno_path` must contains xml file and image file path for annotations.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        sample_num (int): number of samples to load, -1 means all.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 label_list=None):
        super(VOCDataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num)
        self.label_list = label_list

    def parse_dataset(self, ):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        # mapping category name to class id
        # first_class:0, second_class:1, ...
        records = []
        ct = 0
        cname2cid = {}
        if self.label_list:
            label_path = os.path.join(self.dataset_dir, self.label_list)
            if not os.path.exists(label_path):
                raise ValueError("label_list {} does not exists".format(
                    label_path))
            with open(label_path, 'r') as fr:
                label_id = 0
                for line in fr.readlines():
                    cname2cid[line.strip()] = label_id
                    label_id += 1
        else:
            cname2cid = pascalvoc_label()

        with open(anno_path, 'r') as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                img_file, xml_file = [os.path.join(image_dir, x) \
                        for x in line.strip().split()[:2]]
                if not os.path.exists(img_file):
                    logger.warn(
                        'Illegal image file: {}, and it will be ignored'.format(
                            img_file))
                    continue
                if not os.path.isfile(xml_file):
                    logger.warn('Illegal xml file: {}, and it will be ignored'.
                                format(xml_file))
                    continue
                tree = ET.parse(xml_file)
                if tree.find('id') is None:
                    im_id = np.array([ct])
                else:
                    im_id = np.array([int(tree.find('id').text)])

                objs = tree.findall('object')
                im_w = float(tree.find('size').find('width').text)
                im_h = float(tree.find('size').find('height').text)
                if im_w < 0 or im_h < 0:
                    logger.warn(
                        'Illegal width: {} or height: {} in annotation, '
                        'and {} will be ignored'.format(im_w, im_h, xml_file))
                    continue
                gt_bbox = []
                gt_class = []
                gt_score = []
                difficult = []
                for i, obj in enumerate(objs):
                    cname = obj.find('name').text
                    _difficult = int(obj.find('difficult').text)
                    x1 = float(obj.find('bndbox').find('xmin').text)
                    y1 = float(obj.find('bndbox').find('ymin').text)
                    x2 = float(obj.find('bndbox').find('xmax').text)
                    y2 = float(obj.find('bndbox').find('ymax').text)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(im_w - 1, x2)
                    y2 = min(im_h - 1, y2)
                    if x2 > x1 and y2 > y1:
                        gt_bbox.append([x1, y1, x2, y2])
                        gt_class.append([cname2cid[cname]])
                        gt_score.append([1.])
                        difficult.append([_difficult])
                    else:
                        logger.warn(
                            'Found an invalid bbox in annotations: xml_file: {}'
                            ', x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                xml_file, x1, y1, x2, y2))
                gt_bbox = np.array(gt_bbox).astype('float32')
                gt_class = np.array(gt_class).astype('int32')
                gt_score = np.array(gt_score).astype('float32')
                difficult = np.array(difficult).astype('int32')

                voc_rec = {
                    'im_file': img_file,
                    'im_id': im_id,
                    'h': im_h,
                    'w': im_w
                } if 'image' in self.data_fields else {}

                gt_rec = {
                    'gt_class': gt_class,
                    'gt_score': gt_score,
                    'gt_bbox': gt_bbox,
                    'difficult': difficult
                }
                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        voc_rec[k] = v

                if len(objs) != 0:
                    records.append(voc_rec)

                ct += 1
                if self.sample_num > 0 and ct >= self.sample_num:
                    break
        assert len(records) > 0, 'not found any voc record in %s' % (
            self.anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid

    def get_label_list(self):
        return os.path.join(self.dataset_dir, self.label_list)


def pascalvoc_label():
    labels_map = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }
    return labels_map
