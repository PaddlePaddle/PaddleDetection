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


def get_roidb(anno_path,
              sample_num=-1,
              cname2cid=None,
              with_background=True):
    """
    Load VOC records with annotations in xml directory 'anno_path'

    Notes:
    ${anno_path}/ImageSets/Main/train.txt must contains xml file names for annotations
    ${anno_path}/Annotations/xxx.xml must contain annotation info for one record

    Args:
        anno_path (str): root directory for voc annotation data
        sample_num (int): number of samples to load, -1 means all
        cname2cid (dict): the label name to id dictionary
        with_background (bool): whether load background as a class.
                                if True, total class number will
                                be 81. default True

    Returns:
        (records, catname2clsid)
        'records' is list of dict whose structure is:
        {
            'im_file': im_fname, # image file name
            'im_id': im_id, # image id
            'h': im_h, # height of image
            'w': im_w, # width
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
        }
        'cname2id' is a dict to map category name to class id
    """

    txt_file = anno_path
    part = txt_file.split('ImageSets')
    xml_path = os.path.join(part[0], 'Annotations')
    assert os.path.isfile(txt_file) and \
        os.path.isdir(xml_path), 'invalid xml path'

    records = []
    ct = 0
    existence = False if cname2cid is None else True
    if cname2cid is None:
        cname2cid = {}

    # mapping category name to class id
    # background:0, first_class:1, second_class:2, ...
    with open(txt_file, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            fname = line.strip() + '.xml'
            xml_file = os.path.join(xml_path, fname)
            if not os.path.isfile(xml_file):
                continue
            tree = ET.parse(xml_file)
            im_fname = tree.find('filename').text
            if tree.find('id') is None:
                im_id = np.array([ct])
            else:
                im_id = np.array([int(tree.find('id').text)])

            objs = tree.findall('object')
            im_w = float(tree.find('size').find('width').text)
            im_h = float(tree.find('size').find('height').text)
            gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
            gt_class = np.zeros((len(objs), 1), dtype=np.int32)
            gt_score = np.ones((len(objs), 1), dtype=np.float32)
            is_crowd = np.zeros((len(objs), 1), dtype=np.int32)
            difficult = np.zeros((len(objs), 1), dtype=np.int32)
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                if not existence and cname not in cname2cid:
                    # the background's id is 0, so need to add 1.
                    cname2cid[cname] = len(cname2cid) + int(with_background)
                elif existence and cname not in cname2cid:
                    raise KeyError(
                        'Not found cname[%s] in cname2cid when map it to cid.' %
                        (cname))
                gt_class[i][0] = cname2cid[cname]
                _difficult = int(obj.find('difficult').text)
                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                gt_bbox[i] = [x1, y1, x2, y2]
                is_crowd[i][0] = 0
                difficult[i][0] = _difficult
            voc_rec = {
                'im_file': im_fname,
                'im_id': im_id,
                'h': im_h,
                'w': im_w,
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
                'gt_poly': [],
                'difficult': difficult
            }
            if len(objs) != 0:
                records.append(voc_rec)

            ct += 1
            if sample_num > 0 and ct >= sample_num:
                break
    assert len(records) > 0, 'not found any voc record in %s' % (anno_path)
    return [records, cname2cid]


def load(anno_path,
         sample_num=-1,
         use_default_label=True,
         with_background=True):
    """
    Load VOC records with annotations in
    xml directory 'anno_path'

    Notes:
    ${anno_path}/ImageSets/Main/train.txt must contains xml file names for annotations
    ${anno_path}/Annotations/xxx.xml must contain annotation info for one record

    Args:
        @anno_path (str): root directory for voc annotation data
        @sample_num (int): number of samples to load, -1 means all
        @use_default_label (bool): whether use the default mapping of label to id
        @with_background (bool): whether load background as a class.
                                 if True, total class number will
                                 be 81. default True

    Returns:
        (records, catname2clsid)
        'records' is list of dict whose structure is:
        {
            'im_file': im_fname, # image file name
            'im_id': im_id, # image id
            'h': im_h, # height of image
            'w': im_w, # width
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
        }
        'cname2id' is a dict to map category name to class id
    """

    txt_file = anno_path
    part = txt_file.split('ImageSets')
    xml_path = os.path.join(part[0], 'Annotations')
    assert os.path.isfile(txt_file) and \
        os.path.isdir(xml_path), 'invalid xml path'

    # mapping category name to class id
    # if with_background is True:
    #   background:0, first_class:1, second_class:2, ...
    # if with_background is False:
    #   first_class:0, second_class:1, ...
    records = []
    ct = 0
    cname2cid = {}
    if not use_default_label:
        label_path = os.path.join(part[0], 'ImageSets/Main/label_list.txt')
        with open(label_path, 'r') as fr:
            label_id = int(with_background)
            for line in fr.readlines():
                cname2cid[line.strip()] = label_id
                label_id += 1
    else:
        cname2cid = pascalvoc_label(with_background)

    with open(txt_file, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            fname = line.strip() + '.xml'
            xml_file = os.path.join(xml_path, fname)
            if not os.path.isfile(xml_file):
                continue
            tree = ET.parse(xml_file)
            im_fname = tree.find('filename').text
            if tree.find('id') is None:
                im_id = np.array([ct])
            else:
                im_id = np.array([int(tree.find('id').text)])

            objs = tree.findall('object')
            im_w = float(tree.find('size').find('width').text)
            im_h = float(tree.find('size').find('height').text)
            gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
            gt_class = np.zeros((len(objs), 1), dtype=np.int32)
            gt_score = np.ones((len(objs), 1), dtype=np.float32)
            is_crowd = np.zeros((len(objs), 1), dtype=np.int32)
            difficult = np.zeros((len(objs), 1), dtype=np.int32)
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                gt_class[i][0] = cname2cid[cname]
                _difficult = int(obj.find('difficult').text)
                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                gt_bbox[i] = [x1, y1, x2, y2]
                is_crowd[i][0] = 0
                difficult[i][0] = _difficult
            voc_rec = {
                'im_file': im_fname,
                'im_id': im_id,
                'h': im_h,
                'w': im_w,
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
                'gt_poly': [],
                'difficult': difficult
            }
            if len(objs) != 0:
                records.append(voc_rec)

            ct += 1
            if sample_num > 0 and ct >= sample_num:
                break
    assert len(records) > 0, 'not found any voc record in %s' % (anno_path)
    return [records, cname2cid]


def pascalvoc_label(with_background=True):
    labels_map = {
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
    }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
