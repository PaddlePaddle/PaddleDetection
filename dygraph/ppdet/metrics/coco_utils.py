from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from ppdet.py_op.post_process import get_det_res, get_seg_res

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


def get_infer_results(outs, catid):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score. 
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']
    im_shape = outs['im_shape']
    scale_factor = outs['scale_factor']

    infer_res = {}
    if 'bbox' in outs:
        infer_res['bbox'] = get_det_res(outs['bbox'], outs['bbox_num'], im_id,
                                        catid)

    if 'mask' in outs:
        # mask post process
        infer_res['mask'] = get_seg_res(outs['mask'], outs['bbox_num'], im_id,
                                        catid)

    return infer_res


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        jsonfile: Evaluation json file, eg: bbox.json, mask.json.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt != None or anno_file != None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
