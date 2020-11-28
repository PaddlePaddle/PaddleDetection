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
from collections import OrderedDict

from ppdet.core.workspace import register, serializable
from ppdet.evaluation.map_utils import DetectionMAP

from .dataset import DataSet
import logging
logger = logging.getLogger(__name__)


@register
@serializable
class VOCDataSet(DataSet):
    """
    Load dataset with PascalVOC format.

    Notes:
    `anno_path` must contains xml file and image file path for annotations.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether use the default mapping of
            label to integer index. Default True.
        with_background (bool): whether load background as a class,
            default True.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=False,
                 with_background=True,
                 label_list='label_list.txt'):
        super(VOCDataSet, self).__init__(
            image_dir=image_dir,
            anno_path=anno_path,
            sample_num=sample_num,
            dataset_dir=dataset_dir,
            with_background=with_background)
        # roidbs is list of dict whose structure is:
        # {
        #     'im_file': im_fname, # image file name
        #     'im_id': im_id, # image id
        #     'h': im_h, # height of image
        #     'w': im_w, # width
        #     'is_crowd': is_crowd,
        #     'gt_class': gt_class,
        #     'gt_score': gt_score,
        #     'gt_bbox': gt_bbox,
        #     'difficult': difficult
        # }
        self.roidbs = None
        # 'cname2id' is a dict to map category name to class id
        self.cname2cid = OrderedDict()
        self.use_default_label = use_default_label
        self.label_list = label_list
        self.id_indexs = {}

    def load_roidb_and_cname2cid(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        # mapping category name to class id
        # if with_background is True:
        #   background:0, first_class:1, second_class:2, ...
        # if with_background is False:
        #   first_class:0, second_class:1, ...
        records = []
        ct = 0
        if not self.use_default_label:
            label_path = os.path.join(self.dataset_dir, self.label_list)
            if not os.path.exists(label_path):
                raise ValueError("label_list {} does not exists".format(
                    label_path))
            with open(label_path, 'r') as fr:
                label_id = int(self.with_background)
                for line in fr.readlines():
                    self.cname2cid[line.strip()] = label_id
                    label_id += 1
        else:
            self.cname2cid = pascalvoc_label(self.with_background)

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
                is_crowd = []
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
                        gt_class.append([self.cname2cid[cname]])
                        gt_score.append([1.])
                        is_crowd.append([0])
                        difficult.append([_difficult])
                    else:
                        logger.warn(
                            'Found an invalid bbox in annotations: xml_file: {}'
                            ', x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                xml_file, x1, y1, x2, y2))
                gt_bbox = np.array(gt_bbox).astype('float32')
                gt_class = np.array(gt_class).astype('int32')
                gt_score = np.array(gt_score).astype('float32')
                is_crowd = np.array(is_crowd).astype('int32')
                difficult = np.array(difficult).astype('int32')
                voc_rec = {
                    'im_file': img_file,
                    'im_id': im_id,
                    'h': im_h,
                    'w': im_w,
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_score': gt_score,
                    'gt_bbox': gt_bbox,
                    'difficult': difficult
                }
                if len(objs) != 0:
                    records.append(voc_rec)
                    self.id_indexs[im_id[0]] = ct
                    ct += 1
                if self.sample_num > 0 and ct >= self.sample_num:
                    break
        assert len(records) > 0, 'not found any voc record in %s' % (
            self.anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        self.roidbs = records

    def get_gt_with_imid(self, im_ids):
        gt_boxes = []
        gt_labels = []
        difficults = []
        for im_id in im_ids:
            gt_info = self.roidbs[self.id_indexs[im_id]]
            gt_boxes.append(gt_info['gt_bbox'])
            gt_labels.append(gt_info['gt_class'])
            difficults.append(gt_info['difficult'])
        gt_boxes = np.asarray(gt_boxes)
        gt_labels = np.asarray(gt_labels)
        difficults = np.asarray(difficults)
        return gt_boxes, gt_labels, difficults

    def evaluate(self,
                 results=None,
                 jsonfile=None,
                 style='11point',
                 classwise=False,
                 is_bbox_normalized=False,
                 num_classes=20,
                 max_dets=(100, 300, 1000)):
        assert jsonfile == None, 'Currently does not support json file evaluation.'
        assert style in ['11point', 'integral'
                         ], 'evaluate style can only be `11point` or `integral`'

        assert 'bbox' in results[0]
        logger.info("Start evaluate...")
        overlap_thresh = 0.5

        detection_map = DetectionMAP(
            class_num=num_classes,
            overlap_thresh=overlap_thresh,
            map_type=style,
            classwise=classwise,
            is_bbox_normalized=is_bbox_normalized,
            cname2cid=self.cname2cid)

        for t in results:
            bboxes = t['bbox'][0]
            bbox_lengths = t['bbox'][1][0]
            im_ids = np.array(t['im_id'][0]).flatten()
            if bboxes.shape == (1, 1) or bboxes is None:
                continue

            gt_boxes, gt_labels, difficults = self.get_gt_with_imid(im_ids)

            if gt_boxes.shape[0] == 0:
                # gt_bbox, gt_class, difficult read as zero padded Tensor
                bbox_idx = 0
                for i in range(len(gt_boxes)):
                    gt_box = gt_boxes[i]
                    gt_label = gt_labels[i]
                    difficult = None if difficults is None \
                                    else difficults[i]
                    bbox_num = bbox_lengths[i]
                    bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
                    gt_box, gt_label, difficult = self._prune_zero_padding(
                        gt_box, gt_label, difficult)
                    detection_map.update(bbox, gt_box, gt_label, difficult)
                    bbox_idx += bbox_num
            else:
                # gt_box, gt_label, difficult read as LoDTensor
                #gt_box_lengths = gt_boxes.shape[0]
                bbox_idx = 0
                gt_box_idx = 0
                for i in range(len(bbox_lengths)):
                    bbox_num = bbox_lengths[i]
                    bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
                    gt_box = gt_boxes[i]
                    gt_label = gt_labels[i]
                    difficult = None if difficults is None else \
                                difficults[i]
                    detection_map.update(bbox, gt_box, gt_label, difficult)
                    bbox_idx += bbox_num

        logger.info("Accumulating evaluatation results...")
        detection_map.accumulate()
        map_stat = 100. * detection_map.get_map()
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(overlap_thresh, style,
                                                       map_stat))
        return map_stat

    def _prune_zero_padding(self, gt_box, gt_label, difficult=None):
        valid_cnt = 0
        for i in range(len(gt_box)):
            if gt_box[i, 0] == 0 and gt_box[i, 1] == 0 and \
                    gt_box[i, 2] == 0 and gt_box[i, 3] == 0:
                break
            valid_cnt += 1
        return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
                if difficult is not None else None)


def pascalvoc_label(with_background=True):
    labels_map = OrderedDict({
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    })
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "voc2012 categories.".format(anno_file))
        return vocall_category_info(with_background)
    else:
        logger.info("Load categories from {}".format(anno_file))
        return get_category_info_from_anno(anno_file, with_background)


def get_category_info_from_anno(anno_file, with_background=True):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        anno_file (str): annotation file path
        with_background (bool, default True):
            whether load background as class 0.
    """
    cats = []
    with open(anno_file) as f:
        for line in f.readlines():
            cats.append(line.strip())

    if cats[0] != 'background' and with_background:
        cats.insert(0, 'background')
    if cats[0] == 'background' and not with_background:
        cats = cats[1:]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def vocall_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup voc dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = pascalvoc_label(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
