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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os

import paddle.fluid as fluid

from ppdet.utils.voc_eval import bbox_eval as voc_bbox_eval

__all__ = ['parse_fetches', 'eval_run', 'eval_results', 'json_eval_results']

logger = logging.getLogger(__name__)


def parse_fetches(fetches, prog=None, extra_keys=None):
    """
    Parse fetch variable infos from model fetches,
    values for fetch_list and keys for stat
    """
    keys, values = [], []
    cls = []
    for k, v in fetches.items():
        if hasattr(v, 'name'):
            keys.append(k)
            v.persistable = True
            values.append(v.name)
        else:
            cls.append(v)

    if prog is not None and extra_keys is not None:
        for k in extra_keys:
            try:
                v = fluid.framework._get_var(k, prog)
                keys.append(k)
                values.append(v.name)
            except Exception:
                pass

    return keys, values, cls


def eval_run(exe, compile_program, pyreader, keys, values, cls):
    """
    Run evaluation program, return program outputs.
    """
    iter_id = 0
    results = []
    if len(cls) != 0:
        values = []
        for i in range(len(cls)):
            _, accum_map = cls[i].get_map_var()
            cls[i].reset(exe)
            values.append(accum_map)

    try:
        pyreader.start()
        while True:
            outs = exe.run(compile_program,
                           fetch_list=values,
                           return_numpy=False)
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            results.append(res)
            if iter_id % 100 == 0:
                logger.info('Test iter {}'.format(iter_id))
            iter_id += 1
    except (StopIteration, fluid.core.EOFException):
        pyreader.reset()
    logger.info('Test finish iter {}'.format(iter_id))

    return results


def eval_results(results,
                 feed,
                 metric,
                 num_classes,
                 resolution=None,
                 is_bbox_normalized=False,
                 output_directory=None,
                 map_type='11point'):
    """Evaluation for evaluation program results"""
    box_ap_stats = []
    if metric == 'COCO':
        from ppdet.utils.coco_eval import proposal_eval, bbox_eval, mask_eval
        anno_file = getattr(feed.dataset, 'annotation', None)
        with_background = getattr(feed, 'with_background', True)
        if 'proposal' in results[0]:
            output = 'proposal.json'
            if output_directory:
                output = os.path.join(output_directory, 'proposal.json')
            proposal_eval(results, anno_file, output)
        if 'bbox' in results[0]:
            output = 'bbox.json'
            if output_directory:
                output = os.path.join(output_directory, 'bbox.json')

            box_ap_stats = bbox_eval(results, anno_file, output,
                                     with_background,
                                     is_bbox_normalized=is_bbox_normalized)

        if 'mask' in results[0]:
            output = 'mask.json'
            if output_directory:
                output = os.path.join(output_directory, 'mask.json')
            mask_eval(results, anno_file, output, resolution)
    else:
        if 'accum_map' in results[-1]:
            res = np.mean(results[-1]['accum_map'][0])
            logger.info('mAP: {:.2f}'.format(res * 100.))
            box_ap_stats.append(res * 100.)
        elif 'bbox' in results[0]:
            box_ap = voc_bbox_eval(
                results, num_classes, 
                is_bbox_normalized=is_bbox_normalized,
                map_type=map_type)
            box_ap_stats.append(box_ap)
    return box_ap_stats


def json_eval_results(feed, metric, json_directory=None):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    from ppdet.utils.coco_eval import cocoapi_eval
    anno_file = getattr(feed.dataset, 'annotation', None)
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logger.info("{} not exists!".format(v_json))
