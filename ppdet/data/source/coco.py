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
import os
import sys
import json
import itertools

from .dataset import DataSet
from ppdet.core.workspace import register, serializable
from ppdet.evaluation.map_utils import draw_pr_curve
from ppdet.evaluation.result_out import proposal2out, bbox2out, mask2out, segm2out

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

        self.anno_file = os.path.join(self.dataset_dir, self.anno_path)
        assert self.anno_file.endswith('.json'), \
            'invalid coco annotation file: ' + self.anno_file

        self.coco = COCO(self.anno_file)
        self.cat_ids = self.coco.getCatIds()

    def load_roidb_and_cname2cid(self):
        image_dir = os.path.join(self.dataset_dir, self.image_dir)
        img_ids = self.coco.getImgIds()
        records = []
        ct = 0

        # when with_background = True, mapping category to classid, like:
        #   background:0, first_class:1, second_class:2, ...
        catid2clsid = dict({
            catid: i + int(self.with_background)
            for i, catid in enumerate(self.cat_ids)
        })
        cname2cid = dict({
            self.coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })

        if 'annotations' not in self.coco.dataset:
            self.load_image_only = True
            logger.warn('Annotation file: {} does not contains ground truth '
                        'and load image information only.'.format(
                            self.anno_file))

        for img_id in img_ids:
            img_anno = self.coco.loadImgs(img_id)[0]
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
                ins_anno_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                instances = self.coco.loadAnns(ins_anno_ids)

                bboxes = []
                for inst in instances:
                    x, y, box_w, box_h = inst['bbox']
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(im_w - 1, x1 + max(0, box_w - 1))
                    y2 = min(im_h - 1, y1 + max(0, box_h - 1))
                    if x2 >= x1 and y2 >= y1:
                        inst['clean_bbox'] = [x1, y1, x2, y2]
                        bboxes.append(inst)
                    else:
                        logger.warn(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, x1, y1, x2, y2))
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
        logger.debug('{} samples in file {}'.format(ct, self.anno_file))
        self.roidbs, self.cname2cid = records, cname2cid

    def evaluate(self,
                 results=None,
                 jsonfile=None,
                 style=['bbox'],
                 classwise=False,
                 is_bbox_normalized=False,
                 num_classes=None,
                 max_dets=(100, 300, 1000)):
        """
        Evaluation in COCO dataset.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            style: COCOeval style, can be `bbox` , `segm` and `proposal`.
            max_dets: COCO evaluation maxDets.

        """
        assert self.coco != None and self.anno_file != None
        assert results != None or jsonfile != None
        style = style if isinstance(style, list) else [style]
        allowed_styles = ['bbox', 'segm', 'proposal']
        for style in style:
            if style not in allowed_styles:
                raise KeyError('evaluate style {} is not supported'.format(
                    style))
        logger.info("Start evaluate...")
        if jsonfile == None:
            jsonfile = []
            style = []
            jsonfile = self.save_eval_json(
                results, is_bbox_normalized=is_bbox_normalized)
            for v in allowed_styles:
                for f in jsonfile:
                    if str(v + '.json') in f:
                        style.append(v)
                        break
        jsonfile = jsonfile if isinstance(jsonfile, list) else [jsonfile]
        assert jsonfile != [], 'No correct result in network output'
        assert len(jsonfile) == len(style)
        map_stats = []
        for i, f in enumerate(jsonfile):
            coco_dt = self.coco.loadRes(f)
            if style[i] == 'proposal':
                coco_eval = COCOeval(self.coco, coco_dt, 'bbox')
                coco_eval.params.useCats = 0
                coco_eval.params.maxDets = list(max_dets)
            else:
                coco_eval = COCOeval(self.coco, coco_dt, style[i])
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            map_stats.append(coco_eval.stats)
            if classwise:
                # Compute per-category AP and PR curve
                # from https://github.com/facebookresearch/detectron2/
                try:
                    from terminaltables import AsciiTable
                except Exception as e:
                    logger.warn(
                        'terminaltables not found, plaese install terminaltables. for example: `pip install terminaltables`.'
                    )
                    raise e
                precisions = coco_eval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]
                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (str(nm["name"]), '{:0.3f}'.format(float(ap))))
                    pr_array = precisions[0, :, idx, 0, 2]
                    recall_array = np.arange(0.0, 1.01, 0.01)
                    draw_pr_curve(
                        pr_array,
                        recall_array,
                        out_dir=style[i] + '_pr_curve',
                        file_name='{}_precision_recall_curve.jpg'.format(nm[
                            "name"]))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                logger.info('Per-category AP \n' + table.table)
                logger.info("per-category PR curve has output to {} folder.".
                            format(style[i] + '_pr_curve'))
        # flush coco evaluation result
        sys.stdout.flush()
        return map_stats[0][0]

    def save_eval_json(self, results, is_bbox_normalized=False, out_dir=None):
        json_file_list = []
        if 'bbox' in results[0]:
            outfile = 'bbox.json'
            if out_dir:
                outfile = os.path.join(out_dir, 'bbox.json')
            clsid2catid = dict({
                i + int(self.with_background): catid
                for i, catid in enumerate(self.cat_ids)
            })

            xywh_results = bbox2out(
                results, clsid2catid, is_bbox_normalized=is_bbox_normalized)

            if len(xywh_results) == 0:
                logger.warning("The number of valid bbox detected is zero.\n \
                    Please use reasonable model and check input data.\n \
                    stop eval!")
                return [0.0]
            with open(outfile, 'w') as f:
                json.dump(xywh_results, f)
            json_file_list.append(outfile)

        if 'mask' in results[0]:
            outfile = 'segm.json'
            if out_dir:
                outfile = os.path.join(out_dir, 'mask.json')
            clsid2catid = {i + 1: v for i, v in enumerate(self.cat_ids)}
            segm_results = []
            for t in results:
                im_ids = np.array(t['im_id'][0])
                bboxes = t['bbox'][0]
                lengths = t['bbox'][1][0]
                masks = t['mask']
                if bboxes.shape == (1, 1) or bboxes is None:
                    continue
                if len(bboxes.tolist()) == 0:
                    continue
                s = 0
                for i in range(len(lengths)):
                    num = lengths[i]
                    im_id = int(im_ids[i][0])
                    clsid_scores = bboxes[s:s + num][:, 0:2]
                    mask = masks[s:s + num]
                    s += num
                    for j in range(num):
                        clsid, score = clsid_scores[j].tolist()
                        catid = int(clsid2catid[clsid])
                        segm = mask[j]
                        segm['counts'] = segm['counts'].decode('utf8')
                        coco_res = {
                            'image_id': im_id,
                            'category_id': int(catid),
                            'segmentation': segm,
                            'score': score
                        }
                        segm_results.append(coco_res)

            if len(segm_results) == 0:
                logger.warning("The number of valid mask detected is zero.\n \
                    Please use reasonable model and check input data.")
                return

            with open(outfile, 'w') as f:
                json.dump(segm_results, f)
            json_file_list.append(outfile)

        elif 'segm' in results[0]:
            outfile = 'segm.json'
            if out_dir:
                outfile = os.path.join(out_dir, output)
            clsid2catid = {i: v for i, v in enumerate(self.cat_ids)}
            segm_results = []
            for t in results:
                im_id = int(t['im_id'][0][0])
                segs = t['segm']
                for mask in segs:
                    catid = int(clsid2catid[mask[0]])
                    masks = mask[1]
                    mask_score = masks[1]
                    segm = masks[0]
                    segm['counts'] = segm['counts'].decode('utf8')
                    coco_res = {
                        'image_id': im_id,
                        'category_id': catid,
                        'segmentation': segm,
                        'score': mask_score
                    }
                    segm_results.append(coco_res)

            if len(segm_results) == 0:
                logger.warning("The number of valid mask detected is zero.\n \
                    Please use reasonable model and check input data.")
                return

            with open(outfile, 'w') as f:
                json.dump(segm_results, f)
            json_file_list.append(outfile)

        if 'proposal' in results[0]:
            outfile = 'proposal.json'
            if out_dir:
                outfile = os.path.join(out_dir, 'proposal.json')
            xywh_results = proposal2out(results)
            assert len(
                xywh_results
            ) > 0, "The number of valid proposal detected is zero.\n \
                Please use reasonable model and check input data."

            with open(outfile, 'w') as f:
                json.dump(xywh_results, f)
            json_file_list.append(outfile)
        return json_file_list


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "coco17 categories.".format(anno_file))
        return coco17_category_info(with_background)
    else:
        logger.info("Load categories from {}".format(anno_file))
        from pycocotools.coco import COCO
        coco = COCO(anno_file)
        cats = coco.loadCats(coco.getCatIds())
        clsid2catid = {
            i + int(with_background): cat['id']
            for i, cat in enumerate(cats)
        }
        catid2name = {cat['id']: cat['name'] for cat in cats}
        if with_background:
            clsid2catid.update({0: -1})
            catid2name.update({-1: 'background'})
        return clsid2catid, catid2name


def coco17_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of COCO2017 dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    catid2name = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }

    if not with_background:
        clsid2catid = {k - 1: v for k, v in clsid2catid.items()}
        catid2name.pop(0)
    else:
        clsid2catid.update({0: 0})

    return clsid2catid, catid2name
