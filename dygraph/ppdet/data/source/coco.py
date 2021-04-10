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
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class COCODataSet(DetDataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1):
        super(COCODataSet, self).__init__(dataset_dir, image_dir, anno_path,
                                          data_fields, sample_num)
        self.load_image_only = False
        self.load_semantic = False

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        ct = 0

        catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                for inst in instances:
                    # check gt bbox
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue
                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0:
                    continue

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                difficult = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format 
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0], ]
                    elif 'segmentation' in box and box['segmentation']:
                        gt_poly[i] = box['segmentation']
                        has_segmentation = True

                if has_segmentation and not any(gt_poly):
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }
                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
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
