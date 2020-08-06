from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


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


def coco_eval_results(outs_res=None, include_mask=False, dataset=None):
    print("start evaluate bbox using coco api")
    import io
    import six
    import json
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ppdet.py_op.post_process import get_det_res, get_seg_res
    anno_file = os.path.join(dataset.dataset_dir, dataset.anno_path)
    cocoGt = COCO(anno_file)
    catid = {
        i + dataset.with_background: v
        for i, v in enumerate(cocoGt.getCatIds())
    }

    if outs_res is not None and len(outs_res) > 0:
        det_res = []
        for outs in outs_res:
            det_res += get_det_res(outs['bbox'], outs['bbox_num'],
                                   outs['im_id'], catid)

        with io.open("bbox.json", 'w') as outfile:
            encode_func = unicode if six.PY2 else str
            outfile.write(encode_func(json.dumps(det_res)))

        cocoDt = cocoGt.loadRes("bbox.json")
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    if outs_res is not None and len(outs_res) > 0 and include_mask:
        seg_res = []
        for outs in outs_res:
            seg_res += get_seg_res(outs['mask'], outs['bbox_num'],
                                   outs['im_id'], catid)

        with io.open("mask.json", 'w') as outfile:
            encode_func = unicode if six.PY2 else str
            outfile.write(encode_func(json.dumps(seg_res)))

        cocoSg = cocoGt.loadRes("mask.json")
        cocoEval = COCOeval(cocoGt, cocoSg, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
