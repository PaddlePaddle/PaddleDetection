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
import copy
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
import numpy as np
from ppdet.core.workspace import register, serializable
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['COCODataSet', 'SlicedCOCODataSet', 'SemiCOCODataSet', 'ZeroshotCOCODataSet']


@register
@serializable
class COCODataSet(DetDataset):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth. 
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total 
            record's, if empty_ratio is out of [0. ,1.), do not sample the 
            records and use all the empty entries. 1. as default
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1):
        super(COCODataSet, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            data_fields,
            sample_num,
            repeat=repeat)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

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
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
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
            is_empty = False
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
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
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
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format 
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(
                                box['segmentation'],
                                dtype=object).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                    if 'track_id' in box:
                        gt_track_id[i][0] = box['track_id']
                        has_track_id = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }
                if has_track_id:
                    gt_rec.update({'gt_track_id': gt_track_id})

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
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records


@register
@serializable
class SlicedCOCODataSet(COCODataSet):
    """Sliced COCODataSet"""

    def __init__(
            self,
            dataset_dir=None,
            image_dir=None,
            anno_path=None,
            data_fields=['image'],
            sample_num=-1,
            load_crowd=False,
            allow_empty=False,
            empty_ratio=1.,
            repeat=1,
            sliced_size=[640, 640],
            overlap_ratio=[0.25, 0.25], ):
        super(SlicedCOCODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            load_crowd=load_crowd,
            allow_empty=allow_empty,
            empty_ratio=empty_ratio,
            repeat=repeat, )
        self.sliced_size = sliced_size
        self.overlap_ratio = overlap_ratio

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
        empty_records = []
        ct = 0
        ct_sub = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))
        try:
            import sahi
            from sahi.slicing import slice_image
        except Exception as e:
            logger.error(
                'sahi not found, plaese install sahi. '
                'for example: `pip install sahi`, see https://github.com/obss/sahi.'
            )
            raise e

        sub_img_ids = 0
        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            slice_image_result = sahi.slicing.slice_image(
                image=im_path,
                slice_height=self.sliced_size[0],
                slice_width=self.sliced_size[1],
                overlap_height_ratio=self.overlap_ratio[0],
                overlap_width_ratio=self.overlap_ratio[1])

            sub_img_num = len(slice_image_result)
            for _ind in range(sub_img_num):
                im = slice_image_result.images[_ind]
                coco_rec = {
                    'image': im,
                    'im_id': np.array([sub_img_ids + _ind]),
                    'h': im.shape[0],
                    'w': im.shape[1],
                    'ori_im_id': np.array([img_id]),
                    'st_pix': np.array(
                        slice_image_result.starting_pixels[_ind],
                        dtype=np.float32),
                    'is_last': 1 if _ind == sub_img_num - 1 else 0,
                } if 'image' in self.data_fields else {}
                records.append(coco_rec)
            ct_sub += sub_img_num
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('{} samples and slice to {} sub_samples in file {}'.format(
            ct, ct_sub, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records


@register
@serializable
class SemiCOCODataSet(COCODataSet):
    """Semi-COCODataSet used for supervised and unsupervised dataSet"""

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1,
                 supervised=True):
        super(SemiCOCODataSet, self).__init__(
            dataset_dir, image_dir, anno_path, data_fields, sample_num,
            load_crowd, allow_empty, empty_ratio, repeat)
        self.supervised = supervised
        self.length = -1  # defalut -1 means all

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
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset or self.supervised == False:
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
            is_empty = False
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
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
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
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format 
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(box['segmentation']
                                        ).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
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
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records

        if self.supervised:
            logger.info(f'Use {len(self.roidbs)} sup_samples data as LABELED')
        else:
            if self.length > 0:  # unsup length will be decide by sup length
                all_roidbs = self.roidbs.copy()
                selected_idxs = [
                    np.random.choice(len(all_roidbs))
                    for _ in range(self.length)
                ]
                self.roidbs = [all_roidbs[i] for i in selected_idxs]
            logger.info(
                f'Use {len(self.roidbs)} unsup_samples data as UNLABELED')

    def __getitem__(self, idx):
        n = len(self.roidbs)
        if self.repeat > 1:
            idx %= n
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(4)
            ]
        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
        else:
            roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1

        return self.transform(roidb)


class ZeroshotCOCODataSet(COCODataSet):
    """Zeroshot COCODataSet used for OV-DETR"""
    SEEN_CLASSES = (
        "toilet",
        "bicycle",
        "apple",
        "train",
        "laptop",
        "carrot",
        "motorcycle",
        "oven",
        "chair",
        "mouse",
        "boat",
        "kite",
        "sheep",
        "horse",
        "sandwich",
        "clock",
        "tv",
        "backpack",
        "toaster",
        "bowl",
        "microwave",
        "bench",
        "book",
        "orange",
        "bird",
        "pizza",
        "fork",
        "frisbee",
        "bear",
        "vase",
        "toothbrush",
        "spoon",
        "giraffe",
        "handbag",
        "broccoli",
        "refrigerator",
        "remote",
        "surfboard",
        "car",
        "bed",
        "banana",
        "donut",
        "skis",
        "person",
        "truck",
        "bottle",
        "suitcase",
        "zebra",
    )
    UNSEEN_CLASSES = (
        "umbrella",
        "cow",
        "cup",
        "bus",
        "keyboard",
        "skateboard",
        "dog",
        "couch",
        "tie",
        "snowboard",
        "sink",
        "elephant",
        "cake",
        "scissors",
        "airplane",
        "cat",
        "knife",
    )

    def __init__(
            self,
            dataset_dir,
            image_dir,
            anno_path,
            dataset_fields=['image'],
            sample_num=-1,
            load_crowd=False,
            allow_empty=False,
            empty_ratio=1.,
            repeat=-1,
            label_map=False,
    ):
        super(CocoDetection, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            dataset_fields,
            sample_num,
            repeat=repeat,
        )
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self._transforms = transforms
        self.cat_ids = self.coco.getCatIds(self.SEEN_CLASSES + self.UNSEEN_CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_ids_unseen = self.coco.getCatIds(self.UNSEEN_CLASSES)
        self.prepare = ConvertCocoPolysToMask(
            return_masks, self.cat2label, label_map, self.cat_ids_unseen
        )

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        # print('getitem_target',target)
        image_id = self.ids[idx]
        # print('cat_ids', self.cat_ids)
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        # print('target', target['labels'])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if len(target["labels"]) == 0:
            return self[(idx + 1) % len(self)]
        else:
            return img, target
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = paddle.to_tensor(mask, dtype='uint8')
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = paddle.stack(masks, dim=0)
    else:
        masks = paddle.zeros((0, height, width), dtype='uint8')
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, cat2label=None, label_map=False, cat_ids_unseen=None):
        self.return_masks = return_masks
        self.cat2label = cat2label
        self.label_map = label_map
        self.cat_ids_unseen = cat_ids_unseen

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = paddle.to_tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        # boxes = paddle.to_tensor(boxes, dtype='float32').reshape(-1, 4)
        boxes = paddle.reshape(paddle.to_tensor(boxes, dtype='float32'), [-1, 4])
        # print('ori_boxes', boxes)
        boxes[:, 2:] += boxes[:, :2]
        # boxes[:, 0::2].clamp_(min=0, max=w)
        # boxes[:, 1::2].clamp_(min=0, max=h)
        boxes[:, 0::2] = paddle.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = paddle.clip(boxes[:, 1::2], 0, h)
        # print('boxes', boxes)

        # for obj in anno :
        #     print('obj["category_id"]', obj["category_id"])
        # print('self.cat2label', self.cat2label)
        if self.label_map:
            classes = [
                self.cat2label[obj["category_id"]]
                if obj["category_id"] >= 0
                else obj["category_id"]
                for obj in anno
            ]
        else:
            classes = [obj["category_id"] for obj in anno]
        classes = paddle.to_tensor(classes, dtype='int64')
        # print('classes', classes)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = paddle.to_tensor(keypoints, dtype='float32')
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        # print('out_boxes', boxes)
        # exit()
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        # exit()
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = paddle.to_tensor([obj["area"] for obj in anno])
        iscrowd = paddle.to_tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = paddle.to_tensor([int(h), int(w)])
        target["size"] = paddle.to_tensor([int(h), int(w)])

        return image, target


# 单元测试
if __name__ == '__main__':
    set_seed(7)

    #字典转结构体
    class DictToStruct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    #COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    params = {"dataset_dir": 'dataset/coco',
              "image_dir": 'val2017',
              "anno_path": 'annotations/instances_val2017_all_2.json',
              }
    args = DictToStruct(**params)
    trans = COCODataSet(args)