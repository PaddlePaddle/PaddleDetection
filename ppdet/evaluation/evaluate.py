# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time
from ppdet.utils.post_process import mstest_box_post_process, mstest_mask_post_process, corner_post_process
from .result_out import mask_encode

import paddle.fluid as fluid

import logging
logger = logging.getLogger(__name__)


class Evaluation(object):
    """
    Evaluation of coco, voc, etc.

    Args:
        cfg (dict): cfg of model.
        is_bbox_normalized(bool): 
    """

    def __init__(self,
                 cfg,
                 is_bbox_normalized=False,
                 eval_dir=None,
                 classwise=False):
        self.dataset = cfg['EvalReader']['dataset']
        self.cfg = cfg
        self.is_bbox_normalized = is_bbox_normalized
        self.eval_dir = eval_dir
        self.num_classes = cfg.num_classes

    def eval_run(self,
                 exe,
                 compile_program,
                 loader,
                 keys,
                 values,
                 sub_prog=None,
                 sub_keys=None,
                 sub_values=None,
                 resolution=None):
        """
        Run evaluation program, return program outputs.
        """
        iter_id = 0
        results = []

        images_num = 0
        start_time = time.time()
        has_bbox = 'bbox' in keys

        try:
            loader.start()
            while True:
                outs = exe.run(compile_program,
                               fetch_list=values,
                               return_numpy=False)
                res = {
                    k: (np.array(v), v.recursive_sequence_lengths())
                    for k, v in zip(keys, outs)
                }
                multi_scale_test = getattr(self.cfg, 'MultiScaleTEST', None)
                mask_multi_scale_test = multi_scale_test and 'Mask' in self.cfg.architecture

                if multi_scale_test:
                    post_res = mstest_box_post_process(res, multi_scale_test,
                                                       self.cfg.num_classes)
                    res.update(post_res)
                if mask_multi_scale_test:
                    place = fluid.CUDAPlace(
                        0) if self.cfg.use_gpu else fluid.CPUPlace()
                    sub_feed = self._get_sub_feed(res, place)
                    sub_prog_outs = exe.run(sub_prog,
                                            feed=sub_feed,
                                            fetch_list=sub_values,
                                            return_numpy=False)
                    sub_prog_res = {
                        k: (np.array(v), v.recursive_sequence_lengths())
                        for k, v in zip(sub_keys, sub_prog_outs)
                    }
                    post_res = mstest_mask_post_process(sub_prog_res, self.cfg)
                    res.update(post_res)
                if multi_scale_test:
                    res = self._clean_res(
                        res, ['im_info', 'bbox', 'im_id', 'im_shape', 'mask'])
                if 'mask' in res:
                    res['mask'] = mask_encode(res, resolution)
                post_config = getattr(self.cfg, 'PostProcess', None)
                if 'Corner' in self.cfg.architecture and post_config is not None:
                    corner_post_process(res, post_config, self.cfg.num_classes)
                if 'TTFNet' in self.cfg.architecture:
                    res['bbox'][1].append([len(res['bbox'][0])])
                if 'segm' in res:
                    res['segm'] = self._get_masks(res)
                results.append(res)
                if iter_id % 100 == 0:
                    logger.info('Test iter {}'.format(iter_id))
                iter_id += 1
                if 'bbox' not in res or len(res['bbox'][1]) == 0:
                    has_bbox = False
                images_num += len(res['bbox'][1][0]) if has_bbox else 1
        except (StopIteration, fluid.core.EOFException):
            loader.reset()
        logger.info('Test finish iter {}'.format(iter_id))

        end_time = time.time()
        fps = images_num / (end_time - start_time)
        if has_bbox:
            logger.info('Total number of images: {}, inference time: {} fps.'.
                        format(images_num, fps))
        else:
            logger.info('Total iteration: {}, inference time: {} batch/s.'.
                        format(images_num, fps))

        self.results = results

    def _get_sub_feed(self, input, place):
        def _length2lod(length_lod):
            offset_lod = [0]
            for i in length_lod:
                offset_lod.append(offset_lod[-1] + i)
            return [offset_lod]

        new_dict = {}
        res_feed = {}
        key_name = ['bbox', 'im_info', 'im_id', 'im_shape', 'bbox_flip']
        for k in key_name:
            if k in input.keys():
                new_dict[k] = input[k]
        for k in input.keys():
            if 'image' in k:
                new_dict[k] = input[k]
        for k, v in new_dict.items():
            data_t = fluid.LoDTensor()
            data_t.set(v[0], place)
            if 'bbox' in k:
                lod = _length2lod(v[1][0])
                data_t.set_lod(lod)
            res_feed[k] = data_t
        return res_feed

    def _clean_res(self, result, keep_name_list):
        clean_result = {}
        for k in result.keys():
            if k in keep_name_list:
                clean_result[k] = result[k]
        result.clear()
        return clean_result

    def _get_masks(self, result):
        import pycocotools.mask as mask_util
        if result is None:
            return {}
        seg_pred = result['segm'][0].astype(np.uint8)
        cate_label = result['cate_label'][0].astype(np.int)
        cate_score = result['cate_score'][0].astype(np.float)
        num_ins = seg_pred.shape[0]
        masks = []
        for idx in range(num_ins - 1):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(
                    cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks.append([cate_label[idx], rst])
        return masks

    def eval_results(self, classwise=False):
        assert self.results is not None, 'No results found, please execute `eval_run` to generate results.'
        map_stats = self.dataset.evaluate(
            results=self.results,
            classwise=classwise,
            is_bbox_normalized=self.is_bbox_normalized,
            num_classes=self.num_classes)
        return map_stats

    def json_eval_results(self, classwise=False):
        """
        cocoapi eval with already exists proposal.json, bbox.json or mask.json
        """
        assert self.cfg.metric == 'COCO'
        allowed_json_file = ['bbox.json', 'mask.json', 'proposal.json']
        allowed_styles = ['bbox', 'segm', 'proposal']
        json_directory = self.eval_dir
        json_file_list = []
        styles = []
        for i, v in enumerate(allowed_json_file):
            if json_directory:
                assert os.path.exists(
                    json_directory
                ), "The json directory:{} does not exist".format(json_directory)
                v_json = os.path.join(str(json_directory), v)
            else:
                v_json = v
            if os.path.exists(v_json):
                json_file_list.append(v_json)
                styles.append(allowed_styles[i])
        assert json_file_list != [], "Not found any json file."
        self.dataset.evaluate(
            jsonfile=json_file_list, style=styles, classwise=classwise)

    def save_eval_json(self, out_dir=None):
        assert self.cfg.metric == 'COCO'
        self.dataset.save_eval_json(
            self.results,
            is_bbox_normalized=self.is_bbox_normalized,
            out_dir=self.eval_dir)
