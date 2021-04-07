#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import six
import os
import numpy as np
import cv2


def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            w = xmax - xmin + bias
            h = ymax - ymin + bias
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_seg_res(masks, bboxes, mask_nums, image_id, label_to_cat_id_map):
    import pycocotools.mask as mask_util
    seg_res = []
    k = 0
    for i in range(len(mask_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = mask_nums[i]
        for j in range(det_nums):
            mask = masks[k].astype(np.uint8)
            score = float(bboxes[k][1])
            label = int(bboxes[k][0])
            k = k + 1
            if label == -1:
                continue
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
        clsid = int(clsid_labels[i])
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
