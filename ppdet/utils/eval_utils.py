from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
from ppdet.py_op.post_process import get_det_res, get_seg_res
import logging
logger = logging.getLogger(__name__)


def json_eval_results(metric, json_directory=None, dataset=None):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    from ppdet.utils.coco_eval import cocoapi_eval
    anno_file = dataset.get_anno()
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


def get_infer_results(outs_res, eval_type, catid, im_info):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score. 
    """
    if outs_res is None or len(outs_res) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )
    infer_res = {}

    if 'bbox' in eval_type:
        box_res = []
        for i, outs in enumerate(outs_res):
            im_ids = im_info[i][2]
            box_res += get_det_res(outs['bbox'], outs['bbox_num'], im_ids,
                                   catid)
        infer_res['bbox'] = box_res

    if 'mask' in eval_type:
        seg_res = []
        # mask post process
        for i, outs in enumerate(outs_res):
            im_shape = im_info[i][0]
            scale_factor = im_info[i][1]
            im_ids = im_info[i][2]
            mask = outs['mask']
            seg_res += get_seg_res(mask, outs['bbox_num'], im_ids, catid)
        infer_res['mask'] = seg_res

    return infer_res


def eval_results(res, metric, anno_file):
    """
    Evalute the inference result
    """
    eval_res = []
    if metric == 'COCO':
        from ppdet.utils.coco_eval import cocoapi_eval

        if 'bbox' in res:
            with open("bbox.json", 'w') as f:
                json.dump(res['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            bbox_stats = cocoapi_eval('bbox.json', 'bbox', anno_file=anno_file)
            eval_res.append(bbox_stats)
            sys.stdout.flush()
        if 'mask' in res:
            with open("mask.json", 'w') as f:
                json.dump(res['mask'], f)
                logger.info('The mask result is saved to mask.json.')

            seg_stats = cocoapi_eval('mask.json', 'segm', anno_file=anno_file)
            eval_res.append(seg_stats)
            sys.stdout.flush()
    else:
        raise NotImplemented("Only COCO metric is supported now.")

    return eval_res
