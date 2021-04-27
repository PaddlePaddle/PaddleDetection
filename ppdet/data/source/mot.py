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
import numpy as np
from collections import OrderedDict

from .dataset import DetDataset
from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class MOTDataSet(DetDataset):
    """
    Load dataset with MOT format.
    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str|list): mot annotation file path, muiti-source mot dataset.
        sample_num (int): number of samples to load, -1 means all.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=[],
                 data_fields=['image'],
                 sample_num=-1,
                 label_list=None):
        super(MOTDataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num)
        self.anno_path = anno_path
        self.label_list = label_list

        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        if isinstance(self.anno_path, str):
            self.anno_path = [self.anno_path]
        for ds in self.anno_path:
            with open(os.path.join(dataset_dir, 'image_lists', ds),
                      'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    os.path.join(dataset_dir, x.strip())
                    for x in self.img_files[ds]
                ]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace(
                    '.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]
            ]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = int(max_index + 1)

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)

        logger.info('=' * 80)
        logger.info('MOT dataset summary: ')
        logger.info(self.tid_num)
        logger.info('total identities: {}'.format(self.nID))
        logger.info('start index: {}'.format(self.tid_start_index))
        logger.info('=' * 80)

    def get_anno(self):
        if self.anno_path == []:
            return
        # only used to get categories and metric
        return os.path.join(self.dataset_dir, self.anno_path[0])

    def parse_dataset(self):
        # mapping category name to class id
        #   first_class:0, second_class:1, ...
        records = []
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
            cname2cid = mot_label()

        for files_index in range(self.nF):
            for i, c in enumerate(self.cds):
                if files_index >= c:
                    ds = list(self.label_files.keys())[i]
                    start_index = c
            img_file = self.img_files[ds][files_index - start_index]
            lbl_file = self.label_files[ds][files_index - start_index]

            if not os.path.exists(img_file):
                logger.warn('Illegal image file: {}, and it will be ignored'.
                            format(img_file))
                continue
            if not os.path.isfile(lbl_file):
                logger.warn('Illegal label file: {}, and it will be ignored'.
                            format(lbl_file))
                continue

            labels0 = np.loadtxt(lbl_file, dtype=np.float32).reshape(-1, 6)
            # each row in labels0 (N, 6) is [gt_class, gt_identity, cx, cy, w, h]

            cx, cy = labels0[:, 2], labels0[:, 3]
            w, h = labels0[:, 4], labels0[:, 5]
            gt_bbox = np.stack((cx, cy, w, h)).T.astype('float32')
            gt_class = labels0[:, 0:1].astype('int32')
            gt_score = np.ones((len(labels0), 1)).astype('float32')
            gt_ide = labels0[:, 1:2].astype('int32')  # gt_identity

            mot_rec = {
                'im_file': img_file,
                'im_id': files_index,
            } if 'image' in self.data_fields else {}

            gt_rec = {
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
                'gt_ide': gt_ide,
            }

            for k, v in gt_rec.items():
                if k in self.data_fields:
                    mot_rec[k] = v

            records.append(mot_rec)
            if self.sample_num > 0 and files_index >= self.sample_num:
                break
        assert len(records) > 0, 'not found any mot record in %s' % (
            self.anno_path)
        logger.info('{} samples in file {}'.format(self.nF, self.anno_path))
        self.roidbs, self.cname2cid = records, cname2cid


def mot_label():
    labels_map = {'person': 0}
    return labels_map


def _is_valid_video(f, extensions=('.mp4', '.avi', '.mov', '.rmvb', 'flv')):
    return f.lower().endswith(extensions)


@register
@serializable
class MOTVideoDataset(DetDataset):
    """
    Load MOT dataset with MOT format from video for inference.
    Args:
        video_file (str): name of the video file
        dataset_dir (str): root directory for dataset.
    """

    def __init__(self, video_file='', dataset_dir=None, **kwargs):
        super(MOTVideoDataset, self).__init__(dataset_dir=dataset_dir)
        self.video_file = video_file
        self.dataset_dir = dataset_dir
        self.roidbs = None

    def parse_dataset(self, ):
        if not self.roidbs:
            self.roidbs = self._load_video_images()

    def _load_video_images(self):
        video_path = self.video_file
        self.cap = cv2.VideoCapture(video_path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info('Lenth of the video: {:d} frames'.format(self.vn))
        res = True
        ct = 0
        records = []
        while res:
            res, img = self.cap.read()
            image = np.ascontiguousarray(img, dtype=np.float32)
            rec = {
                'im_id': np.array([ct]),
                'image': image,
                'img0': image,
            }
            ct += 1
            records.append(rec)
        records = records[:-1]
        assert len(records) > 0, "No image file found"
        return records

    def set_video(self, video_file):
        self.video_file = video_file
        assert os.path.isfile(self.video_file) and _is_valid_video(self.video_file), \
                "wrong or unsupported file format: {}".format(self.video_file)
        self.roidbs = self._load_video_images()
