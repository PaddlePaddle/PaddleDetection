import six
import os
import numpy as np
from .bbox import delta2bbox, clip_bbox, expand_bbox, nms
import cv2


def get_det_res(bboxes, scores, labels, bbox_nums, image_id,
                label_to_cat_id_map):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            box = bboxes[k]
            score = float(scores[k])
            label = int(labels[k])
            if label < 0: continue
            k = k + 1
            xmin, ymin, xmax, ymax = box.tolist()
            category_id = label_to_cat_id_map[label]
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_seg_res(masks, scores, labels, mask_nums, image_id,
                label_to_cat_id_map):
    import pycocotools.mask as mask_util
    seg_res = []
    k = 0
    for i in range(len(mask_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = mask_nums[i]
        for j in range(det_nums):
            mask = masks[k]
            score = float(scores[k])
            label = int(labels[k])
            k = k + 1
            cat_id = label_to_cat_id_map[label]
            rle = mask_util.encode(
                np.array(
                    mask[:, :, None], order="F", dtype="uint8"))[0]
            if six.PY3:
                if 'counts' in rle:
                    rle['counts'] = rle['counts'].decode("utf8")
            sg_res = {
                'image_id': cur_image_id,
                'category_id': cat_id,
                'segmentation': rle,
                'score': score
            }
            seg_res.append(sg_res)
    return seg_res


def get_solov2_segm_res(results, image_id, num_id_to_cat_id_map):
    import pycocotools.mask as mask_util
    segm_res = []
    # for each batch
    segms = results['segm'].astype(np.uint8)
    clsid_labels = results['cate_label']
    clsid_scores = results['cate_score']
    lengths = segms.shape[0]
    im_id = int(image_id[0][0])
    if lengths == 0 or segms is None:
        return None
    # for each sample
    for i in range(lengths - 1):
        clsid = int(clsid_labels[i]) + 1
        catid = num_id_to_cat_id_map[clsid]
        score = float(clsid_scores[i])
        mask = segms[i]
        segm = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        segm['counts'] = segm['counts'].decode('utf8')
        coco_res = {
            'image_id': im_id,
            'category_id': catid,
            'segmentation': segm,
            'score': score
        }
        segm_res.append(coco_res)
    return segm_res
