# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""

Authors: yanglijuan04@baidu.com
Date:     2021/7/30 下午2:54
"""
import os
import copy
import json
import numpy as np
from PIL import Image

import paddle

from ppdet.core.workspace import create
from ppdet.engine.trainer import Trainer
from ppdet.data.source.category import get_categories
from ppdet.metrics import Metric, COCOMetric, VOCMetric, get_infer_results, SNIPERCOCOMetric
from ppdet.utils.logger import setup_logger
from ppdet.utils.visualizer import visualize_results, save_result

logger = setup_logger('ppdet.engine')


__all__ = ['SNIPERTrainer']


def _get_categories(metric_type, anno_file=None, arch=None):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        metric_type (str): metric type, currently support 'coco', 'voc', 'oid'
            and 'widerface'.
        anno_file (str): annotation file path
    """

    if metric_type.lower() == 'snipercoco':
        if anno_file and os.path.isfile(anno_file):
            # lazy import pycocotools here
            from pycocotools.coco import COCO

            coco = COCO(anno_file)
            cats = coco.loadCats(coco.getCatIds())

            clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
            catid2name = {cat['id']: cat['name'] for cat in cats}
            return clsid2catid, catid2name
    else:
        return get_categories(metric_type, anno_file, arch)


class SNIPERTrainer(Trainer):
    """SNIPERTrainer"""

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg['classwise'] if 'classwise' in self.cfg else False
        if 'COCO' in self.cfg.metric:  # add sniper
            # TODO: bias should be unified
            bias = self.cfg['bias'] if 'bias' in self.cfg else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} \
                if self.mode == 'eval' else None

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            anno_file = self.dataset.get_anno()

            if self.mode == 'train' and validate:
                if not hasattr(self, '_eval_loader'):  # sniper add
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    self._eval_loader = create('EvalReader')(
                        self._eval_dataset,
                        self.cfg.worker_num,
                        batch_sampler=self._eval_batch_sampler)
                    anno_file = self._eval_dataset.get_anno()

            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            if self.cfg.metric == "COCO":
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only
                    )
                ]
            elif self.cfg.metric == "SNIPERCOCO": # sniper
                self._metrics = [
                    SNIPERCOCOMetric(
                        anno_file=anno_file,
                        dataset=self._eval_dataset if hasattr(self, '_eval_dataset') else self.dataset,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only
                    )
                ]
        elif self.cfg.metric == 'VOC':
            self._metrics = [
                VOCMetric(
                    label_list=self.dataset.get_label_list(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type,
                    classwise=classwise)
            ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = []

    def train(self, validate=False,):
        super(SNIPERTrainer, self).train(validate=validate)
        if self.cfg.save_proposals:
            results = []
            dataset = self._create_new_dataset()
            self.loader.dataset = dataset
            with paddle.no_grad():
                self.model.eval()
                for step_id, data in enumerate(self.loader):
                    outs = self.model(data)
                    for key in ['im_shape', 'scale_factor', 'im_id']:
                        outs[key] = data[key]
                    for key, value in outs.items():
                        if hasattr(value, 'numpy'):
                            outs[key] = value.numpy()

                    results.append(outs)

            results = dataset.anno_cropper.aggregate_chips_detections(results)
            # sniper
            proposals = []
            clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    bbox_num = outs['bbox_num']
                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                        if 'bbox' in batch_res else None
                    if bbox_res:
                        proposals += bbox_res
            logger.info("save proposals in {}".format(self.cfg.proposals_path))
            with open(self.cfg.proposals_path, 'w') as f:
                json.dump(proposals, f)

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_txt=False):
        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = _get_categories(  # sniper
            self.cfg.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        results = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()

            results.append(outs)
        # sniper
        results = self.dataset.anno_cropper.aggregate_chips_detections(results)
        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)

            bbox_num = outs['bbox_num']

            start = 0
            for i, im_id in enumerate(outs['im_id']):
                im_id = int(im_id)  # np.array to int
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                self.status['original_image'] = np.array(image.copy())

                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                    if 'bbox' in batch_res else None
                mask_res = batch_res['mask'][start:end] \
                    if 'mask' in batch_res else None
                segm_res = batch_res['segm'][start:end] \
                    if 'segm' in batch_res else None
                keypoint_res = batch_res['keypoint'][start:end] \
                    if 'keypoint' in batch_res else None
                image = visualize_results(
                    image, bbox_res, mask_res, segm_res, keypoint_res,
                    int(im_id), catid2name, draw_threshold)
                self.status['result_image'] = np.array(image.copy())
                if self._compose_callback:
                    self._compose_callback.on_step_end(self.status)
                # save image with detection
                save_name = self._get_save_image_name(output_dir, image_path)
                logger.info("Detection bbox results save in {}".format(
                    save_name))
                image.save(save_name, quality=95)

                if save_txt:
                    results = {}
                    results["im_id"] = im_id
                    if bbox_res:
                        results["bbox_res"] = bbox_res

                    save_path = os.path.splitext(save_name)[0] + '.txt'
                    save_result(save_path, results, catid2name, draw_threshold)
                start = end


    def _create_new_dataset(self):
        dataset = copy.deepcopy(self.dataset)
        # init anno_cropper
        dataset.init_anno_cropper()
        # generate infer roidbs
        ori_roidbs = dataset.get_ori_roidbs()
        roidbs = dataset.anno_cropper.crop_infer_anno_records(ori_roidbs)
        # set new roidbs
        dataset.set_roidbs(roidbs)

        return dataset
