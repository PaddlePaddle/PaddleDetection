import numpy as np
from numba import jit


@jit
def bbox_overlaps(roi_boxes, gt_boxes):
    w1 = np.maximum(roi_boxes[:, 2] - roi_boxes[:, 0] + 1, 0)
    h1 = np.maximum(roi_boxes[:, 3] - roi_boxes[:, 1] + 1, 0)
    w2 = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0] + 1, 0)
    h2 = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1] + 1, 0)
    area1 = w1 * h1
    area2 = w2 * h2

    overlaps = np.zeros((roi_boxes.shape[0], gt_boxes.shape[0]))
    for ind1 in range(roi_boxes.shape[0]):
        for ind2 in range(gt_boxes.shape[0]):
            inter_x1 = np.maximum(roi_boxes[ind1, 0], gt_boxes[ind2, 0])
            inter_y1 = np.maximum(roi_boxes[ind1, 1], gt_boxes[ind2, 1])
            inter_x2 = np.minimum(roi_boxes[ind1, 2], gt_boxes[ind2, 2])
            inter_y2 = np.minimum(roi_boxes[ind1, 3], gt_boxes[ind2, 3])
            inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0)
            inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0)
            inter_area = inter_w * inter_h
            iou = inter_area * 1.0 / (area1[ind1] + area2[ind2] - inter_area)
            overlaps[ind1, ind2] = iou
    return overlaps


@jit
def box_to_delta(ex_boxes, gt_boxes, weights):
    ex_w = ex_boxes[:, 2] - ex_boxes[:, 0] + 1
    ex_h = ex_boxes[:, 3] - ex_boxes[:, 1] + 1
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_w
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    targets = np.vstack([dx, dy, dw, dh]).transpose()
    return targets


@jit
def compute_targets(roi_boxes, gt_boxes, labels, bbox_reg_weights):
    assert roi_boxes.shape[0] == gt_boxes.shape[0]
    assert roi_boxes.shape[1] == 4
    assert gt_boxes.shape[1] == 4

    targets = np.zeros(roi_boxes.shape)
    bbox_reg_weights = np.asarray(bbox_reg_weights)
    targets = box_to_delta(
        ex_boxes=roi_boxes, gt_boxes=gt_boxes, weights=bbox_reg_weights)

    return np.hstack([labels[:, np.newaxis], targets]).astype(
        np.float32, copy=False)


@jit
def expand_bbox_targets(bbox_targets_input,
                        class_nums=81,
                        is_cls_agnostic=False):
    class_labels = bbox_targets_input[:, 0]
    fg_inds = np.where(class_labels > 0)[0]
    if not is_cls_agnostic:
        class_nums = 2
    bbox_targets = np.zeros((class_labels.shape[0], 4 * class_nums))
    bbox_inside_weights = np.zeros(bbox_targets.shape)
    for ind in fg_inds:
        class_label = int(class_labels[ind]) if not is_cls_agnostic else 1
        start_ind = class_label * 4
        end_ind = class_label * 4 + 4
        bbox_targets[ind, start_ind:end_ind] = bbox_targets_input[ind, 1:]
        bbox_inside_weights[ind, start_ind:end_ind] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


@jit
def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return np.where(suppressed == 0)[0]


@jit
def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


@jit
def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


@jit
def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


@jit
def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1)))
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')
