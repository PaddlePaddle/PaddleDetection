import six
import os
import numpy as np
from numba import jit
from .bbox import delta2bbox, clip_bbox, expand_bbox, nms


def bbox_post_process(bboxes,
                      bbox_probs,
                      bbox_deltas,
                      im_info,
                      keep_top_k=100,
                      score_thresh=0.05,
                      nms_thresh=0.5,
                      class_nums=81,
                      bbox_reg_weights=[0.1, 0.1, 0.2, 0.2]):
    bbox_nums = [0, bboxes.shape[0]]
    bboxes_v = np.array(bboxes)
    bbox_probs_v = np.array(bbox_probs)
    bbox_deltas_v = np.array(bbox_deltas)
    variance_v = np.array(bbox_reg_weights)
    new_bboxes = [[] for _ in range(len(bbox_nums) - 1)]
    new_bbox_nums = [0]
    for i in range(len(bbox_nums) - 1):
        start = bbox_nums[i]
        end = bbox_nums[i + 1]
        if start == end:
            continue

        bbox_deltas_n = bbox_deltas_v[start:end, :]  # box delta 
        rois_n = bboxes_v[start:end, :]  # box 
        rois_n = rois_n / im_info[i][2]  # scale 
        rois_n = delta2bbox(bbox_deltas_n, rois_n, variance_v)
        rois_n = clip_bbox(rois_n, im_info[i][:2] / im_info[i][2])
        cls_boxes = [[] for _ in range(class_nums)]
        scores_n = bbox_probs_v[start:end, :]
        for j in range(1, class_nums):
            inds = np.where(scores_n[:, j] > score_thresh)[0]
            scores_j = scores_n[inds, j]
            rois_j = rois_n[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                np.float32, copy=False)
            keep = nms(dets_j, nms_thresh)
            nms_dets = dets_j[keep, :]
            #add labels
            label = np.array([j for _ in range(len(keep))])
            nms_dets = np.hstack((label[:, np.newaxis], nms_dets)).astype(
                np.float32, copy=False)
            cls_boxes[j] = nms_dets

        # Limit to max_per_image detections **over all classes**
        image_scores = np.hstack(
            [cls_boxes[j][:, 1] for j in range(1, class_nums)])
        if len(image_scores) > keep_top_k:
            image_thresh = np.sort(image_scores)[-keep_top_k]
            for j in range(1, class_nums):
                keep = np.where(cls_boxes[j][:, 1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
        new_bboxes_n = np.vstack([cls_boxes[j] for j in range(1, class_nums)])
        new_bboxes[i] = new_bboxes_n
        new_bbox_nums.append(len(new_bboxes_n) + new_bbox_nums[-1])
        labels = new_bboxes_n[:, 0]
        scores = new_bboxes_n[:, 1]
        boxes = new_bboxes_n[:, 2:]
    new_bboxes = np.vstack([new_bboxes[k] for k in range(len(bbox_nums) - 1)])
    new_bbox_nums = np.array(new_bbox_nums)
    return new_bbox_nums, new_bboxes


@jit
def mask_post_process(bbox_nums, bboxes, masks, im_info):
    bboxes = np.array(bboxes)
    M = cfg.resolution
    scale = (M + 2.0) / M
    masks_v = np.array(masks)
    boxes = bboxes[:, 2:]
    labels = bboxes[:, 0]
    segms_results = [[] for _ in range(len(bbox_nums) - 1)]
    sum = 0
    for i in range(len(bbox_nums) - 1):
        bboxes_n = bboxes[bbox_nums[i]:bbox_nums[i + 1]]
        cls_segms = []
        masks_n = masks_v[bbox_nums[i]:bbox_nums[i + 1]]
        boxes_n = boxes[bbox_nums[i]:bbox_nums[i + 1]]
        labels_n = labels[bbox_nums[i]:bbox_nums[i + 1]]
        im_h = int(round(im_info[i][0] / im_info[i][2]))
        im_w = int(round(im_info[i][1] / im_info[i][2]))
        boxes_n = expand_boxes(boxes_n, scale)
        boxes_n = boxes_n.astype(np.int32)
        padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)
        for j in range(len(bboxes_n)):
            class_id = int(labels_n[j])
            padded_mask[1:-1, 1:-1] = masks_n[j, class_id, :, :]

            ref_box = boxes_n[j, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.mrcnn_thresh_binarize, dtype=np.uint8)
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
    segms_results = np.vstack([segms_results[k] for k in range(len(lod) - 1)])
    bboxes = np.hstack([segms_results, bboxes])
    return bboxes[:, :3]


@jit
def get_det_res(bbox_nums,
                bbox,
                image_id,
                image_shape,
                num_id_to_cat_id_map,
                batch_size=1):
    det_res = []
    bbox_v = np.array(bbox)
    if bbox_v.shape == (
            1,
            1, ):
        return dts_res
    assert (len(bbox_nums) == batch_size + 1), \
      "Error bbox_nums Tensor offset dimension. bbox_nums({}) vs. batch_size({})"\
                    .format(len(bbox_nums), batch_size)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = bbox_nums[i + 1] - bbox_nums[i]
        image_id = int(image_id[i][0])
        image_width = int(image_shape[i][1])  #int(data[i][-1][1])
        image_height = int(image_shape[i][2])  #int(data[i][-1][2])
        for j in range(dt_num_this_img):
            dt = bbox_v[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            category_id = num_id_to_cat_id_map[num_id]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


@jit
def get_seg_res(mask_nums, mask, image_id, num_id_to_cat_id_map, batch_size=1):
    seg_res = []
    mask_v = np.array(mask)
    k = 0
    for i in range(batch_size):
        image_id = int(image_id[i][0])
        dt_num_this_img = mask_nums[i + 1] - mask_nums[i]
        for j in range(dt_num_this_img):
            dt = mask_v[k]
            k = k + 1
            sg, num_id, score = dt.tolist()
            cat_id = num_id_to_cat_id_map[num_id]
            if six.PY3:
                if 'counts' in sg:
                    sg['counts'] = sg['counts'].decode("utf8")
            sg_res = {
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': sg,
                'score': score
            }
            seg_res.append(sg_res)
    return seg_res
