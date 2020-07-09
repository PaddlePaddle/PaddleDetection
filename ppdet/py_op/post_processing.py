import six
import os
import numpy as np
from numba import jit
from .bbox import nms


@jit
def box_decoder(deltas, boxes, weights, bbox_clip=4.13):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] * wx
    dy = deltas[:, 1::4] * wy
    dw = deltas[:, 2::4] * ww
    dh = deltas[:, 3::4] * wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, bbox_clip)
    dh = np.minimum(dh, bbox_clip)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


@jit
def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


#@jit 
def get_nmsed_box(rpn_rois,
                  confs,
                  locs,
                  class_nums,
                  im_info,
                  bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                  score_thresh=0.05,
                  nms_thresh=0.5,
                  detections_per_im=100):
    box_nums = [0, rpn_rois.shape[0]]
    variance_v = np.array(bbox_reg_weights)
    rpn_rois_v = np.array(rpn_rois)
    confs_v = np.array(confs)
    locs_v = np.array(locs)

    im_results = [[] for _ in range(len(box_nums) - 1)]
    new_box_nums = [0]
    for i in range(len(box_nums) - 1):
        start = box_nums[i]
        end = box_nums[i + 1]
        if start == end:
            continue

        locs_n = locs_v[start:end, :]  # box delta 
        rois_n = rpn_rois_v[start:end, :]  # box 
        rois_n = rois_n / im_info[i][2]  # scale 
        rois_n = box_decoder(locs_n, rois_n, variance_v)
        rois_n = clip_tiled_boxes(rois_n, im_info[i][:2] / im_info[i][2])
        cls_boxes = [[] for _ in range(class_nums)]
        scores_n = confs_v[start:end, :]
        for j in range(1, class_nums):
            inds = np.where(scores_n[:, j] > TEST.score_thresh)[0]
            scores_j = scores_n[inds, j]
            rois_j = rois_n[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                np.float32, copy=False)
            keep = nms(dets_j, TEST.nms_thresh)
            nms_dets = dets_j[keep, :]
            #add labels
            label = np.array([j for _ in range(len(keep))])
            nms_dets = np.hstack((label[:, np.newaxis], nms_dets)).astype(
                np.float32, copy=False)
            cls_boxes[j] = nms_dets

        # Limit to max_per_image detections **over all classes**
        image_scores = np.hstack(
            [cls_boxes[j][:, 1] for j in range(1, class_nums)])
        if len(image_scores) > detections_per_im:
            image_thresh = np.sort(image_scores)[-detections_per_im]
            for j in range(1, class_nums):
                keep = np.where(cls_boxes[j][:, 1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
        im_results_n = np.vstack([cls_boxes[j] for j in range(1, class_nums)])
        im_results[i] = im_results_n
        new_box_nums.append(len(im_results_n) + new_box_nums[-1])
        labels = im_results_n[:, 0]
        scores = im_results_n[:, 1]
        boxes = im_results_n[:, 2:]
    im_results = np.vstack([im_results[k] for k in range(len(box_nums) - 1)])
    return new_box_nums, im_results


@jit
def get_dt_res(batch_size, box_nums, nmsed_out, data, num_id_to_cat_id_map):
    dts_res = []
    nmsed_out_v = np.array(nmsed_out)
    if nmsed_out_v.shape == (
            1,
            1, ):
        return dts_res
    assert (len(box_nums) == batch_size + 1), \
      "Error Tensor offset dimension. Box Nums({}) vs. batch_size({})"\
                    .format(len(box_nums), batch_size)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = box_nums[i + 1] - box_nums[i]
        image_id = int(data[i][-1])
        image_width = int(data[i][1][1])
        image_height = int(data[i][1][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
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
            dts_res.append(dt_res)
    return dts_res


@jit
def get_segms_res(batch_size, box_nums, segms_out, data, num_id_to_cat_id_map):
    segms_res = []
    segms_out_v = np.array(segms_out)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = box_nums[i + 1] - box_nums[i]
        image_id = int(data[i][-1])
        for j in range(dt_num_this_img):
            dt = segms_out_v[k]
            k = k + 1
            segm, num_id, score = dt.tolist()
            cat_id = num_id_to_cat_id_map[num_id]
            if six.PY3:
                if 'counts' in segm:
                    segm['counts'] = segm['counts'].decode("utf8")
            segm_res = {
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': segm,
                'score': score
            }
            segms_res.append(segm_res)
    return segms_res
