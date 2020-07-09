import six
import math
import numpy as np
from numba import jit
from .bbox import *
from .mask import *


@jit
def generate_rpn_anchor_target(anchor_box,
                               gt_boxes,
                               is_crowd,
                               im_info,
                               rpn_straddle_thresh,
                               rpn_batch_size_per_im,
                               rpn_positive_overlap,
                               rpn_negative_overlap,
                               rpn_fg_fraction,
                               use_random=True):
    anchor_num = anchor_box.shape[0]
    batch_size = gt_boxes.shape[0]

    for i in range(batch_size):
        im_height = im_info[i][0]
        im_width = im_info[i][1]
        im_scale = im_info[i][2]
        if rpn_straddle_thresh >= 0:
            # Only keep anchors inside the image by a margin of straddle_thresh
            inds_inside = np.where(
                (anchor_box[:, 0] >= -rpn_straddle_thresh
                 ) & (anchor_box[:, 1] >= -rpn_straddle_thresh) & (
                     anchor_box[:, 2] < im_width + rpn_straddle_thresh) & (
                         anchor_box[:, 3] < im_height + rpn_straddle_thresh))[0]
            # keep only inside anchors
            inside_anchors = anchor_box[inds_inside, :]
        else:
            inds_inside = np.arange(anchor_box.shape[0])
            inside_anchors = anchor_box
        gt_boxes_slice = gt_boxes[i] * im_scale
        is_crowd_slice = is_crowd[i]

        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_boxes_slice = gt_boxes_slice[not_crowd_inds]
        iou = bbox_overlaps(inside_anchors, gt_boxes_slice)

        loc_inds, score_inds, labels, gt_inds, bbox_inside_weight = _sample_anchor(
            iou, rpn_batch_size_per_im, rpn_positive_overlap,
            rpn_negative_overlap, rpn_fg_fraction, use_random)
        # unmap to all anchor 
        loc_inds = inds_inside[loc_inds]
        score_inds = inds_inside[score_inds]
        sampled_anchor = anchor_box[loc_inds]
        sampled_gt = gt_boxes_slice[gt_inds]
        box_deltas = bbox2delta(sampled_anchor, sampled_gt, [1., 1., 1., 1.])

        if i == 0:
            loc_indexes = loc_inds
            score_indexes = score_inds
            tgt_labels = labels
            tgt_bboxes = box_deltas
            bbox_inside_weights = bbox_inside_weight
        else:
            loc_indexes = np.concatenate(
                [loc_indexes, loc_inds + i * anchor_num])
            score_indexes = np.concatenate(
                [score_indexes, score_inds + i * anchor_num])
            tgt_labels = np.concatenate([tgt_labels, labels])
            tgt_bboxes = np.vstack([tgt_bboxes, box_deltas])
            bbox_inside_weights = np.vstack([bbox_inside_weights, \
                                             bbox_inside_weight])
    tgt_labels = tgt_labels.astype('float32')
    tgt_bboxes = tgt_bboxes.astype('float32')
    return loc_indexes, score_indexes, tgt_labels, tgt_bboxes, bbox_inside_weights


@jit
def _sample_anchor(anchor_by_gt_overlap,
                   rpn_batch_size_per_im,
                   rpn_positive_overlap,
                   rpn_negative_overlap,
                   rpn_fg_fraction,
                   use_random=True):

    anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
    anchor_to_gt_max = anchor_by_gt_overlap[np.arange(
        anchor_by_gt_overlap.shape[0]), anchor_to_gt_argmax]

    gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
    gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(
        anchor_by_gt_overlap.shape[1])]
    anchors_with_max_overlap = np.where(
        anchor_by_gt_overlap == gt_to_anchor_max)[0]

    labels = np.ones((anchor_by_gt_overlap.shape[0], ), dtype=np.int32) * -1
    labels[anchors_with_max_overlap] = 1
    labels[anchor_to_gt_max >= rpn_positive_overlap] = 1

    num_fg = int(rpn_fg_fraction * rpn_batch_size_per_im)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg and use_random:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    else:
        disable_inds = fg_inds[num_fg:]

    labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    num_bg = rpn_batch_size_per_im - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < rpn_negative_overlap)[0]
    if len(bg_inds) > num_bg and use_random:
        enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
    else:
        enable_inds = bg_inds[:num_bg]

    fg_fake_inds = np.array([], np.int32)
    fg_value = np.array([fg_inds[0]], np.int32)
    fake_num = 0
    for bg_id in enable_inds:
        if bg_id in fg_inds:
            fake_num += 1
            fg_fake_inds = np.hstack([fg_fake_inds, fg_value])
    labels[enable_inds] = 0

    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]

    loc_index = np.hstack([fg_fake_inds, fg_inds])
    score_index = np.hstack([fg_inds, bg_inds])
    labels = labels[score_index]

    gt_inds = anchor_to_gt_argmax[loc_index]

    bbox_inside_weight = np.zeros((len(loc_index), 4), dtype=np.float32)
    bbox_inside_weight[fake_num:, :] = 1
    return loc_index, score_index, labels, gt_inds, bbox_inside_weight


@jit
def generate_proposal_target(rpn_rois,
                             rpn_rois_nums,
                             gt_classes,
                             is_crowd,
                             gt_boxes,
                             im_info,
                             batch_size_per_im,
                             fg_fraction,
                             fg_thresh,
                             bg_thresh_hi,
                             bg_thresh_lo,
                             bbox_reg_weights,
                             class_nums=81,
                             use_random=True,
                             is_cls_agnostic=False,
                             is_cascade_rcnn=False):

    rois = []
    labels_int32 = []
    bbox_targets = []
    bbox_inside_weights = []
    bbox_outside_weights = []
    rois_nums = []
    batch_size = gt_boxes.shape[0]
    # TODO: modify here
    # rpn_rois = rpn_rois.reshape(batch_size, -1, 4)
    st_num = 0
    print("debug: ", rpn_rois_nums)
    for im_i in range(len(rpn_rois_nums)):
        rpn_rois_num = rpn_rois_nums[im_i]
        frcn_blobs = _sample_rois(
            rpn_rois[st_num:rpn_rois_num], gt_classes[im_i], is_crowd[im_i],
            gt_boxes[im_i], im_info[im_i], batch_size_per_im, fg_fraction,
            fg_thresh, bg_thresh_hi, bg_thresh_lo, bbox_reg_weights, class_nums,
            use_random, is_cls_agnostic, is_cascade_rcnn)
        st_num = rpn_rois_num

        rois.append(frcn_blobs['rois'])
        labels_int32.append(frcn_blobs['labels_int32'])
        bbox_targets.append(frcn_blobs['bbox_targets'])
        bbox_inside_weights.append(frcn_blobs['bbox_inside_weights'])
        bbox_outside_weights.append(frcn_blobs['bbox_outside_weights'])
        rois_nums.append(frcn_blobs['rois'].shape[0])

    rois = np.concatenate(rois, axis=0).astype(np.float32)
    bbox_labels = np.concatenate(
        labels_int32, axis=0).astype(np.int32).reshape(-1, 1)
    bbox_gts = np.concatenate(bbox_targets, axis=0).astype(np.float32)
    bbox_inside_weights = np.concatenate(
        bbox_inside_weights, axis=0).astype(np.float32)
    bbox_outside_weights = np.concatenate(
        bbox_outside_weights, axis=0).astype(np.float32)
    rois_nums = np.asarray(rois_nums, np.int32)

    return rois, bbox_labels, bbox_gts, bbox_inside_weights, bbox_outside_weights, rois_nums


@jit
def _sample_rois(rpn_rois,
                 gt_classes,
                 is_crowd,
                 gt_boxes,
                 im_info,
                 batch_size_per_im,
                 fg_fraction,
                 fg_thresh,
                 bg_thresh_hi,
                 bg_thresh_lo,
                 bbox_reg_weights,
                 class_nums,
                 use_random=True,
                 is_cls_agnostic=False,
                 is_cascade_rcnn=False):
    rois_per_image = int(batch_size_per_im)
    fg_rois_per_im = int(np.round(fg_fraction * rois_per_image))

    # Roidb
    im_scale = im_info[2]
    inv_im_scale = 1. / im_scale
    rpn_rois = rpn_rois * inv_im_scale
    if is_cascade_rcnn:
        rpn_rois = rpn_rois[gt_boxes.shape[0]:, :]
    boxes = np.vstack([gt_boxes, rpn_rois])
    gt_overlaps = np.zeros((boxes.shape[0], class_nums))
    box_to_gt_ind_map = np.zeros((boxes.shape[0]), dtype=np.int32)
    if len(gt_boxes) > 0:
        proposal_to_gt_overlaps = bbox_overlaps(boxes, gt_boxes)
        overlaps_argmax = proposal_to_gt_overlaps.argmax(axis=1)
        overlaps_max = proposal_to_gt_overlaps.max(axis=1)
        # Boxes which with non-zero overlap with gt boxes
        overlapped_boxes_ind = np.where(overlaps_max > 0)[0].astype('int32')
        overlapped_boxes_gt_classes = gt_classes[overlaps_argmax[
            overlapped_boxes_ind]].astype('int32')
        gt_overlaps[overlapped_boxes_ind,
                    overlapped_boxes_gt_classes] = overlaps_max[
                        overlapped_boxes_ind]
        box_to_gt_ind_map[overlapped_boxes_ind] = overlaps_argmax[
            overlapped_boxes_ind]

    crowd_ind = np.where(is_crowd)[0]
    gt_overlaps[crowd_ind] = -1

    max_overlaps = gt_overlaps.max(axis=1)
    max_classes = gt_overlaps.argmax(axis=1)

    # Cascade RCNN Decode Filter
    if is_cascade_rcnn:
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws > 0) & (hs > 0))[0]
        boxes = boxes[keep]
        fg_inds = np.where(max_overlaps >= fg_thresh)[0]
        bg_inds = np.where((max_overlaps < bg_thresh_hi) & (max_overlaps >=
                                                            bg_thresh_lo))[0]
        fg_rois_per_this_image = fg_inds.shape[0]
        bg_rois_per_this_image = bg_inds.shape[0]
    else:
        # Foreground
        fg_inds = np.where(max_overlaps >= fg_thresh)[0]
        fg_rois_per_this_image = np.minimum(fg_rois_per_im, fg_inds.shape[0])
        # Sample foreground if there are too many
        if (fg_inds.shape[0] > fg_rois_per_this_image) and use_random:
            fg_inds = np.random.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)
        fg_inds = fg_inds[:fg_rois_per_this_image]
        # Background
        bg_inds = np.where((max_overlaps < bg_thresh_hi) & (max_overlaps >=
                                                            bg_thresh_lo))[0]
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                            bg_inds.shape[0])
        # Sample background if there are too many
        if (bg_inds.shape[0] > bg_rois_per_this_image) and use_random:
            bg_inds = np.random.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)
        bg_inds = bg_inds[:bg_rois_per_this_image]

    keep_inds = np.append(fg_inds, bg_inds)
    sampled_labels = max_classes[keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0
    sampled_boxes = boxes[keep_inds]
    sampled_gts = gt_boxes[box_to_gt_ind_map[keep_inds]]
    sampled_gts[fg_rois_per_this_image:, :] = gt_boxes[0]
    bbox_label_targets = compute_bbox_targets(sampled_boxes, sampled_gts,
                                              sampled_labels, bbox_reg_weights)
    bbox_targets, bbox_inside_weights = expand_bbox_targets(
        bbox_label_targets, class_nums, is_cls_agnostic)
    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)

    # Scale rois
    sampled_rois = sampled_boxes * im_scale

    # Faster RCNN blobs
    frcn_blobs = dict(
        rois=sampled_rois,
        labels_int32=sampled_labels,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights)
    return frcn_blobs


@jit
def generate_mask_target(im_info, gt_classes, is_crowd, gt_segms, rois,
                         rois_nums, labels_int32, num_classes, resolution):
    mask_rois = []
    rois_has_mask_int32 = []
    mask_int32 = []
    st_num = 0
    for i in range(len(rois_nums)):
        rois_num = rois_nums[i]
        mask_blob = _sample_mask(
            rois[st_num:rois_num], labels_int32[st_num:rois_num], gt_segms[i],
            im_info[i], gt_classes[i], is_crowd[i], num_classes, resolution)

        st_num = rois_num
        mask_rois.append(mask_blob['mask_rois'])
        rois_has_mask_int32.append(mask_blob['roi_has_mask_int32'])
        mask_int32.append(mask_blob['mask_int32'])
    mask_rois = np.concatenate(mask_rois, axis=0).astype(np.float32)
    rois_has_mask_int32 = np.concatenate(
        rois_has_mask_int32, axis=0).astype(np.int32)
    mask_int32 = np.concatenate(mask_int32, axis=0).astype(np.int32)

    return mask_rois, rois_has_mask_int32, mask_int32


@jit
def _sample_mask(
        rois,
        label_int32,
        gt_polys,
        im_info,
        gt_classes,
        is_crowd,
        num_classes,
        resolution, ):

    # remove padding 
    new_gt_polys = []
    for i in range(gt_polys.shape[0]):
        gt_segs = []
        for j in range(gt_polys[i].shape[0]):
            new_poly = []
            polys = gt_polys[i][j]
            for ii in range(polys.shape[0]):
                x, y = polys[ii]
                if (x == -1 and y == -1):
                    continue
                elif (x >= 0 and y >= 0):
                    new_poly.append([x, y])  # array, one poly 
            if len(new_poly) > 0:
                gt_segs.append(new_poly)
        new_gt_polys.append(gt_segs)

    im_scale = im_info[2]
    sample_boxes = rois / im_scale

    polys_gt_inds = np.where((gt_classes > 0) & (is_crowd == 0))[0]

    polys_gt = [new_gt_polys[i] for i in polys_gt_inds]
    boxes_from_polys = polys_to_boxes(polys_gt)
    fg_inds = np.where(label_int32 > 0)[0]
    roi_has_mask = fg_inds.copy()

    if fg_inds.shape[0] > 0:
        mask_class_labels = label_int32[fg_inds]
        masks = np.zeros((fg_inds.shape[0], resolution**2), dtype=np.int32)
        rois_fg = sample_boxes[fg_inds]

        overlaps_bbfg_bbpolys = bbox_overlaps_mask(rois_fg, boxes_from_polys)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            poly_gt = polys_gt[fg_polys_ind]
            roi_fg = rois_fg[i]

            mask = polys_to_mask_wrt_box(poly_gt, roi_fg, resolution)
            mask = np.array(mask > 0, dtype=np.int32)
            masks[i, :] = np.reshape(mask, resolution**2)
    else:
        bg_inds = np.where(label_int32 == 0)[0]
        rois_fg = sample_boxes[bg_inds[0]].reshape((1, -1))
        masks = -np.ones((1, resolution**2), dtype=np.int32)
        mask_class_labels = np.zeros((1, ))
        roi_has_mask = np.append(roi_has_mask, 0)

    masks = expand_mask_targets(masks, mask_class_labels, resolution,
                                num_classes)

    rois_fg *= im_scale
    mask_blob = dict()
    mask_blob['mask_rois'] = rois_fg
    mask_blob['roi_has_mask_int32'] = roi_has_mask
    mask_blob['mask_int32'] = masks

    return mask_blob
