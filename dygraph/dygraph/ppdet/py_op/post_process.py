import six
import os
import numpy as np
from numba import jit
from .bbox import delta2bbox, clip_bbox, expand_bbox, nms
import pycocotools.mask as mask_util
import cv2


def bbox_post_process(bboxes,
                      bbox_prob,
                      bbox_deltas,
                      im_shape,
                      scale_factor,
                      keep_top_k=100,
                      score_thresh=0.05,
                      nms_thresh=0.5,
                      class_nums=81,
                      bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                      with_background=True):
    bbox, bbox_num = bboxes
    new_bbox = [[] for _ in range(len(bbox_num))]
    new_bbox_num = []
    st_num = 0
    end_num = 0
    for i in range(len(bbox_num)):
        box_num = bbox_num[i]
        end_num += box_num

        boxes = bbox[st_num:end_num, :]  # bbox 
        boxes = boxes / scale_factor[i]  # scale
        bbox_delta = bbox_deltas[st_num:end_num, :, :]  # bbox delta 
        bbox_delta = np.reshape(bbox_delta, (box_num, -1))
        # step1: decode 
        boxes = delta2bbox(bbox_delta, boxes, bbox_reg_weights)

        # step2: clip 
        boxes = clip_bbox(boxes, im_shape[i][:2] / scale_factor[i])
        # step3: nms 
        cls_boxes = [[] for _ in range(class_nums)]
        scores_n = bbox_prob[st_num:end_num, :]
        for j in range(with_background, class_nums):
            inds = np.where(scores_n[:, j] > score_thresh)[0]
            scores_j = scores_n[inds, j]
            rois_j = boxes[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                np.float32, copy=False)
            keep = nms(dets_j, nms_thresh)
            nms_dets = dets_j[keep, :]
            #add labels
            label = np.array([j for _ in range(len(keep))])
            nms_dets = np.hstack((label[:, np.newaxis], nms_dets)).astype(
                np.float32, copy=False)
            cls_boxes[j] = nms_dets

        st_num += box_num

        # Limit to max_per_image detections **over all classes**
        image_scores = np.hstack(
            [cls_boxes[j][:, 1] for j in range(with_background, class_nums)])
        if len(image_scores) > keep_top_k:
            image_thresh = np.sort(image_scores)[-keep_top_k]
            for j in range(with_background, class_nums):
                keep = np.where(cls_boxes[j][:, 1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
        new_bbox_n = np.vstack(
            [cls_boxes[j] for j in range(with_background, class_nums)])
        new_bbox[i] = new_bbox_n
        new_bbox_num.append(len(new_bbox_n))
    new_bbox = np.vstack([new_bbox[k] for k in range(len(bbox_num))])
    new_bbox_num = np.array(new_bbox_num).astype('int32')
    return new_bbox, new_bbox_num


@jit
def mask_post_process(det_res,
                      im_shape,
                      scale_factor,
                      resolution=14,
                      binary_thresh=0.5):
    bbox = det_res['bbox']
    bbox_num = det_res['bbox_num']
    masks = det_res['mask']
    if masks.shape[0] == 0:
        return masks
    M = resolution
    scale = (M + 2.0) / M
    boxes = bbox[:, 2:]
    labels = bbox[:, 0]
    segms_results = [[] for _ in range(len(bbox_num))]
    sum = 0
    st_num = 0
    end_num = 0
    for i in range(len(bbox_num)):
        length = bbox_num[i]
        end_num += length
        cls_segms = []
        boxes_n = boxes[st_num:end_num]
        labels_n = labels[st_num:end_num]
        masks_n = masks[st_num:end_num]

        im_h = int(round(im_shape[i][0] / scale_factor[i, 0]))
        im_w = int(round(im_shape[i][1] / scale_factor[i, 0]))
        boxes_n = expand_bbox(boxes_n, scale)
        boxes_n = boxes_n.astype(np.int32)
        padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)
        for j in range(len(boxes_n)):
            class_id = int(labels_n[j])
            padded_mask[1:-1, 1:-1] = masks_n[j, class_id, :, :]

            ref_box = boxes_n[j, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > binary_thresh, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)
            im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[
                1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]
            sum += im_mask.sum()
            rle = mask_util.encode(
                np.array(
                    im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms.append(rle)
        segms_results[i] = np.array(cls_segms)[:, np.newaxis]
        st_num += length
    segms_results = np.vstack([segms_results[k] for k in range(len(bbox_num))])
    bboxes = np.hstack([segms_results, bbox])
    return bboxes[:, :3]


@jit
def get_det_res(bboxes, bbox_nums, image_id, num_id_to_cat_id_map):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            category_id = num_id_to_cat_id_map[num_id]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


@jit
def get_seg_res(masks, mask_nums, image_id, num_id_to_cat_id_map):
    seg_res = []
    k = 0
    for i in range(len(mask_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = mask_nums[i]
        for j in range(det_nums):
            dt = masks[k]
            k = k + 1
            sg, num_id, score = dt.tolist()
            cat_id = num_id_to_cat_id_map[num_id]
            if six.PY3:
                if 'counts' in sg:
                    sg['counts'] = sg['counts'].decode("utf8")
            sg_res = {
                'image_id': cur_image_id,
                'category_id': cat_id,
                'segmentation': sg,
                'score': score
            }
            seg_res.append(sg_res)
    return seg_res
