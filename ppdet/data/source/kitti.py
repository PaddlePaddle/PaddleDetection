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

from ppdet.core.workspace import register, serializable
from .ape_proto.frame_pb2 import Frame

from .dataset import DataSet
import logging
logger = logging.getLogger(__name__)

obstacle_categories = {
        3: 'CAR',
        4: 'VAN',
        5: 'TRUNK',
        7: 'CYCLIST',
        10: 'PEDESTRIAN',
        13: 'DNOT_CARE',
        }

@register
@serializable
class KITTIDataSet(DataSet):
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
                 split_file=None,
                 label_file=None,
                 sample_num=-1,
                 use_default_label=True,
                 with_background=True,
                 label_list='label_list.txt'):
        super(KITTIDataSet, self).__init__(
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
        self.split_file = split_file
        self.roidbs = None
        # 'cname2id' is a dict to map category name to class id
        self.cname2cid = None
        self.use_default_label = use_default_label
        self.label_list = label_list

    def load_roidb_and_cname2cid(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)
        split_file = os.path.join(self.dataset_dir, self.split_file)

        # mapping category name to class id
        # if with_background is True:
        #   background:0, first_class:1, second_class:2, ...
        # if with_background is False:
        #   first_class:0, second_class:1, ...
        records = []
        ct = 0
        cname2cid = {}
        if not self.use_default_label:
            label_path = os.path.join(self.dataset_dir, self.label_list)
            if not os.path.exists(label_path):
                raise ValueError("label_list {} does not exists".format(
                    label_path))
            with open(label_path, 'r') as fr:
                label_id = int(self.with_background)
                for line in fr.readlines():
                    cname2cid[line.strip()] = label_id
                    label_id += 1
        else:
            cname2cid = kitti_label(self.with_background)

        with open(split_file, 'r') as fr:
            while True:
                sid = fr.readline().strip()
                if not sid:
                    break
                img_file = os.path.join(image_dir, "{}.png".format(sid))
                pb_file = os.path.join(anno_path, "{}.pb".format(sid))
                if not os.path.exists(img_file):
                    logger.warn(
                        'Illegal image file: {}, and it will be ignored'.format(
                            img_file))
                    continue
                if not os.path.isfile(pb_file):
                    logger.warn('Illegal pb file: {}, and it will be ignored'.
                                format(pb_file))
                    continue

                frame = Frame()
                frame.ParseFromString(open(pb_file).read())
                im_w = int(frame.width)
                im_h = int(frame.height)
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
                for i, obs in enumerate(frame.obstacle):
                    cname = obstacle_categories[int(obs.type)].lower()
		    x1 = float(obs.box2d.xmin)
		    y1 = float(obs.box2d.ymin)
		    x2 = float(obs.box2d.xmax)
		    y2 = float(obs.box2d.ymax)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(im_w - 1, x2)
                    y2 = min(im_h - 1, y2)
                    if x2 > x1 and y2 > y1:
                        gt_bbox.append([x1, y1, x2, y2])
                        gt_class.append([cname2cid[cname]])
                        gt_score.append([1.])
                        is_crowd.append([0])
                        difficult.append([0])
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
                kitti_rec = {
                    'im_file': img_file,
                    'im_id': int(sid),
                    'h': im_h,
                    'w': im_w,
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_score': gt_score,
                    'gt_bbox': gt_bbox,
                    'difficult': difficult
                }

                if len(frame.obstacle) != 0:
                    records.append(kitti_rec)

                ct += 1
                if self.sample_num > 0 and ct >= self.sample_num:
                    break
        assert len(records) > 0, 'not found any kitti record in %s' % (
            self.anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid


def kitti_label(with_background=True):
    labels_map = {
        'car': 1,
        'van': 2,
        'trunk': 3,
        'cyclist': 4,
        'pedestrian': 5,
        'dnot_care': 6,
    }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
