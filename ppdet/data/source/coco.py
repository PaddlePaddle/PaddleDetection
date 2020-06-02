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

from .dataset import DataSet
from ppdet.core.workspace import register, serializable

import logging
logger = logging.getLogger(__name__)


@register
@serializable
class COCODataSet(DataSet):
    """
    Load COCO records with annotations in json file 'anno_path'

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): json file path.
        sample_num (int): number of samples to load, -1 means all.
        with_background (bool): whether load background as a class.
            if True, total class number will be 81. default True.
    """

    def __init__(self,
                 image_dir=None,
                 anno_path=None,
                 dataset_dir=None,
                 sample_num=-1,
                 with_background=True,
                 load_semantic=False):
        super(COCODataSet, self).__init__(
            image_dir=image_dir,
            anno_path=anno_path,
            dataset_dir=dataset_dir,
            sample_num=sample_num,
            with_background=with_background)
        self.anno_path = anno_path
        self.sample_num = sample_num
        self.with_background = with_background
        # `roidbs` is list of dict whose structure is:
        # {
        #     'im_file': im_fname, # image file name
        #     'im_id': img_id, # image id
        #     'h': im_h, # height of image
        #     'w': im_w, # width
        #     'is_crowd': is_crowd,
        #     'gt_score': gt_score,
        #     'gt_class': gt_class,
        #     'gt_bbox': gt_bbox,
        #     'gt_poly': gt_poly,
        # }
        self.roidbs = None
        # a dict used to map category name to class id
        self.cname2cid = None
        self.load_image_only = False
        self.load_semantic = load_semantic

    def load_roidb_and_cname2cid(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        records = []
        ct = 0

        # when with_background = True, mapping category to classid, like:
        #   background:0, first_class:1, second_class:2, ...
        catid2clsid = dict({
            catid: i + int(self.with_background)
            for i, catid in enumerate(cat_ids)
        })
        cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warn('Annotation file: {} does not contains ground truth '
                        'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs(img_id)[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            if not os.path.exists(im_path):
                logger.warn('Illegal image file: {}, and it will be '
                            'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warn('Illegal width: {} or height: {} in annotation, '
                            'and im_id: {} will be ignored'.format(im_w, im_h,
                                                                   img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            }

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                for inst in instances:
                    x, y, box_w, box_h = inst['bbox']
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(im_w - 1, x1 + max(0, box_w - 1))
                    y2 = min(im_h - 1, y1 + max(0, box_h - 1))
                    if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                        inst['clean_bbox'] = [x1, y1, x2, y2]
                        bboxes.append(inst)
                    else:
                        logger.warn(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))
                num_bbox = len(bboxes)

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_score = np.ones((num_bbox, 1), dtype=np.float32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                difficult = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    if 'segmentation' in box:
                        gt_poly[i] = box['segmentation']

                coco_rec.update({
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_score': gt_score,
                    'gt_poly': gt_poly,
                })

                if self.load_semantic:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))
            records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert len(records) > 0, 'not found any coco record in %s' % (anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid
