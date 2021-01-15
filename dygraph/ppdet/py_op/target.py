import six
import math
import numpy as np
from numba import jit
from .bbox import *
from .mask import *


@jit
def generate_rpn_anchor_target(anchors,
                               gt_boxes,
                               is_crowd,
                               im_info,
                               rpn_straddle_thresh,
                               rpn_batch_size_per_im,
                               rpn_positive_overlap,
                               rpn_negative_overlap,
                               rpn_fg_fraction,
                               use_random=True,
                               anchor_reg_weights=[1., 1., 1., 1.]):
    anchor_num = anchors.shape[0]
    batch_size = gt_boxes.shape[0]

    loc_indexes = []
    cls_indexes = []
    tgt_labels = []
    tgt_deltas = []
    anchor_inside_weights = []

    for i in range(batch_size):

        # TODO: move anchor filter into anchor generator 
        im_height = im_info[i][0]
        im_width = im_info[i][1]
        im_scale = im_info[i][2]
        if rpn_straddle_thresh >= 0:
            anchor_inds = np.where((anchors[:, 0] >= -rpn_straddle_thresh) & (
                anchors[:, 1] >= -rpn_straddle_thresh) & (
                    anchors[:, 2] < im_width + rpn_straddle_thresh) & (
                        anchors[:, 3] < im_height + rpn_straddle_thresh))[0]
            anchor = anchors[anchor_inds, :]
        else:
            anchor_inds = np.arange(anchors.shape[0])
            anchor = anchors

        gt_bbox = gt_boxes[i] * im_scale
        is_crowd_slice = is_crowd[i]
        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_bbox = gt_bbox[not_crowd_inds]

        # Step1: match anchor and gt_bbox
        anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels = label_anchor(anchor,
                                                                       gt_bbox)

        # Step2: sample anchor 
        fg_inds, bg_inds, fg_fake_inds, fake_num = sample_anchor(
            anchor_gt_bbox_iou, labels, rpn_positive_overlap,
            rpn_negative_overlap, rpn_batch_size_per_im, rpn_fg_fraction,
            use_random)

        # Step3: make output  
        loc_inds = np.hstack([fg_fake_inds, fg_inds])
        cls_inds = np.hstack([fg_inds, bg_inds])

        sampled_labels = labels[cls_inds]

        sampled_anchors = anchor[loc_inds]
        sampled_gt_boxes = gt_bbox[anchor_gt_bbox_inds[loc_inds]]
        sampled_deltas = bbox2delta(sampled_anchors, sampled_gt_boxes,
                                    anchor_reg_weights)

        anchor_inside_weight = np.zeros((len(loc_inds), 4), dtype=np.float32)
        anchor_inside_weight[fake_num:, :] = 1

        loc_indexes.append(anchor_inds[loc_inds] + i * anchor_num)
        cls_indexes.append(anchor_inds[cls_inds] + i * anchor_num)
        tgt_labels.append(sampled_labels)
        tgt_deltas.append(sampled_deltas)
        anchor_inside_weights.append(anchor_inside_weight)

    loc_indexes = np.concatenate(loc_indexes)
    cls_indexes = np.concatenate(cls_indexes)
    tgt_labels = np.concatenate(tgt_labels).astype('float32')
    tgt_deltas = np.vstack(tgt_deltas).astype('float32')
    anchor_inside_weights = np.vstack(anchor_inside_weights)

    return loc_indexes, cls_indexes, tgt_labels, tgt_deltas, anchor_inside_weights


@jit
def label_anchor(anchors, gt_boxes):
    iou = bbox_overlaps(anchors, gt_boxes)
    # every gt's anchor's index
    gt_bbox_anchor_inds = iou.argmax(axis=0)
    gt_bbox_anchor_iou = iou[gt_bbox_anchor_inds, np.arange(iou.shape[1])]
    gt_bbox_anchor_iou_inds = np.where(iou == gt_bbox_anchor_iou)[0]

    # every anchor's gt bbox's index 
    anchor_gt_bbox_inds = iou.argmax(axis=1)
    anchor_gt_bbox_iou = iou[np.arange(iou.shape[0]), anchor_gt_bbox_inds]

    labels = np.ones((iou.shape[0], ), dtype=np.int32) * -1
    labels[gt_bbox_anchor_iou_inds] = 1

    return anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels


@jit
def sample_anchor(anchor_gt_bbox_iou,
                  labels,
                  rpn_positive_overlap,
                  rpn_negative_overlap,
                  rpn_batch_size_per_im,
                  rpn_fg_fraction,
                  use_random=True):

    labels[anchor_gt_bbox_iou >= rpn_positive_overlap] = 1
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
    bg_inds = np.where(anchor_gt_bbox_iou < rpn_negative_overlap)[0]
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

    return fg_inds, bg_inds, fg_fake_inds, fake_num


@jit
def filter_roi(rois, max_overlap):
    ws = rois[:, 2] - rois[:, 0] + 1
    hs = rois[:, 3] - rois[:, 1] + 1
    keep = np.where((ws > 0) & (hs > 0) & (max_overlap < 1))[0]
    if len(keep) > 0:
        return rois[keep, :]
    return np.zeros((1, 4)).astype('float32')


@jit
def generate_proposal_target(rpn_rois,
                             rpn_rois_num,
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
                             is_cascade_rcnn=False,
                             max_overlaps=None):

    rois = []
    tgt_labels = []
    tgt_deltas = []
    rois_inside_weights = []
    rois_outside_weights = []
    sampled_max_overlaps = []
    new_rois_num = []
    st_num = 0
    end_num = 0
    for im_i in range(len(rpn_rois_num)):
        length = rpn_rois_num[im_i]
        end_num += length
        rpn_roi = rpn_rois[st_num:end_num]
        max_overlap = max_overlaps[st_num:end_num] if is_cascade_rcnn else None
        im_scale = im_info[im_i][2]
        rpn_roi = rpn_roi / im_scale
        gt_bbox = gt_boxes[im_i]

        if is_cascade_rcnn:
            rpn_roi = filter_roi(rpn_roi, max_overlap)
        bbox = np.vstack([gt_bbox, rpn_roi]).astype('float32')

        # Step1: label bbox 
        roi_gt_bbox_inds, labels, max_overlap = label_bbox(
            bbox, gt_bbox, gt_classes[im_i], is_crowd[im_i])

        # Step2: sample bbox 
        fg_inds, bg_inds, fg_nums = sample_bbox(
            max_overlap, batch_size_per_im, fg_fraction, fg_thresh,
            bg_thresh_hi, bg_thresh_lo, bbox_reg_weights, class_nums,
            use_random, is_cls_agnostic, is_cascade_rcnn)

        # Step3: make output 
        sampled_inds = np.append(fg_inds, bg_inds)

        sampled_labels = labels[sampled_inds]
        sampled_labels[fg_nums:] = 0

        sampled_boxes = bbox[sampled_inds]
        sampled_max_overlap = max_overlap[sampled_inds]
        sampled_gt_boxes = gt_bbox[roi_gt_bbox_inds[sampled_inds]]
        sampled_gt_boxes[fg_nums:, :] = 0
        sampled_deltas = compute_bbox_targets(sampled_boxes, sampled_gt_boxes,
                                              sampled_labels, bbox_reg_weights)
        sampled_deltas[fg_nums:, :] = 0
        sampled_deltas, bbox_inside_weights = expand_bbox_targets(
            sampled_deltas, class_nums, is_cls_agnostic)
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)

        roi = sampled_boxes * im_scale
        st_num += length

        rois.append(roi)
        new_rois_num.append(roi.shape[0])
        tgt_labels.append(sampled_labels)
        tgt_deltas.append(sampled_deltas)
        rois_inside_weights.append(bbox_inside_weights)
        rois_outside_weights.append(bbox_outside_weights)
        sampled_max_overlaps.append(sampled_max_overlap)

    rois = np.concatenate(rois, axis=0).astype(np.float32)
    tgt_labels = np.concatenate(
        tgt_labels, axis=0).astype(np.int32).reshape(-1, 1)
    tgt_deltas = np.concatenate(tgt_deltas, axis=0).astype(np.float32)
    rois_inside_weights = np.concatenate(
        rois_inside_weights, axis=0).astype(np.float32)
    rois_outside_weights = np.concatenate(
        rois_outside_weights, axis=0).astype(np.float32)
    sampled_max_overlaps = np.concatenate(
        sampled_max_overlaps, axis=0).astype(np.float32)
    new_rois_num = np.asarray(new_rois_num, np.int32)
    return rois, tgt_labels, tgt_deltas, rois_inside_weights, rois_outside_weights, new_rois_num, sampled_max_overlaps


@jit
def label_bbox(boxes, gt_boxes, gt_classes, is_crowd, class_nums=81):

    iou = bbox_overlaps(boxes, gt_boxes)

    # every roi's gt box's index  
    roi_gt_bbox_inds = np.zeros((boxes.shape[0]), dtype=np.int32)
    roi_gt_bbox_iou = np.zeros((boxes.shape[0], class_nums), dtype=np.float32)

    iou_argmax = iou.argmax(axis=1)
    iou_max = iou.max(axis=1)
    overlapped_boxes_ind = np.where(iou_max > 0)[0].astype('int32')
    roi_gt_bbox_inds[overlapped_boxes_ind] = iou_argmax[overlapped_boxes_ind]
    overlapped_boxes_gt_classes = gt_classes[iou_argmax[
        overlapped_boxes_ind]].astype('int32')
    roi_gt_bbox_iou[overlapped_boxes_ind,
                    overlapped_boxes_gt_classes] = iou_max[overlapped_boxes_ind]

    crowd_ind = np.where(is_crowd)[0]
    roi_gt_bbox_iou[crowd_ind] = -1

    max_overlap = roi_gt_bbox_iou.max(axis=1)
    labels = roi_gt_bbox_iou.argmax(axis=1)

    return roi_gt_bbox_inds, labels, max_overlap


@jit
def sample_bbox(max_overlap,
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

    if is_cascade_rcnn:
        fg_inds = np.where(max_overlap >= fg_thresh)[0]
        bg_inds = np.where((max_overlap < bg_thresh_hi) & (max_overlap >=
                                                           bg_thresh_lo))[0]
        fg_nums = fg_inds.shape[0]
        bg_nums = bg_inds.shape[0]
    else:
        # sampe fg 
        fg_inds = np.where(max_overlap >= fg_thresh)[0]
        fg_nums = np.minimum(fg_rois_per_im, fg_inds.shape[0])
        if (fg_inds.shape[0] > fg_nums) and use_random:
            fg_inds = np.random.choice(fg_inds, size=fg_nums, replace=False)
        fg_inds = fg_inds[:fg_nums]

        # sample bg 
        bg_inds = np.where((max_overlap < bg_thresh_hi) & (max_overlap >=
                                                           bg_thresh_lo))[0]
        bg_nums = rois_per_image - fg_nums
        bg_nums = np.minimum(bg_nums, bg_inds.shape[0])
        if (bg_inds.shape[0] > bg_nums) and use_random:
            bg_inds = np.random.choice(bg_inds, size=bg_nums, replace=False)
        bg_inds = bg_inds[:bg_nums]

    return fg_inds, bg_inds, fg_nums


@jit
def generate_mask_target(im_info, gt_classes, is_crowd, gt_segms, rois,
                         rois_num, labels_int32, num_classes, resolution):
    mask_rois = []
    mask_rois_num = []
    rois_has_mask_int32 = []
    mask_int32 = []
    st_num = 0
    end_num = 0
    for k in range(len(rois_num)):
        length = rois_num[k]
        end_num += length

        # remove padding
        gt_polys = gt_segms[k]
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
                    elif (x >= 0 or y >= 0):
                        new_poly.append([x, y])  # array, one poly 
                if len(new_poly) > 0:
                    gt_segs.append(new_poly)
            new_gt_polys.append(gt_segs)
        im_scale = im_info[k][2]
        boxes = rois[st_num:end_num] / im_scale

        bbox_fg, bbox_has_mask, masks = sample_mask(
            boxes, new_gt_polys, labels_int32[st_num:end_num], gt_classes[k],
            is_crowd[k], num_classes, resolution)

        st_num += length

        mask_rois.append(bbox_fg * im_scale)
        mask_rois_num.append(len(bbox_fg))
        rois_has_mask_int32.append(bbox_has_mask)
        mask_int32.append(masks)

    mask_rois = np.concatenate(mask_rois, axis=0).astype(np.float32)
    mask_rois_num = np.array(mask_rois_num).astype(np.int32)
    rois_has_mask_int32 = np.concatenate(
        rois_has_mask_int32, axis=0).astype(np.int32)
    mask_int32 = np.concatenate(mask_int32, axis=0).astype(np.int32)

    return mask_rois, mask_rois_num, rois_has_mask_int32, mask_int32


@jit
def sample_mask(boxes, gt_polys, label_int32, gt_classes, is_crowd, num_classes,
                resolution):

    gt_polys_inds = np.where((gt_classes > 0) & (is_crowd == 0))[0]
    _gt_polys = [gt_polys[i] for i in gt_polys_inds]
    boxes_from_polys = polys_to_boxes(_gt_polys)

    fg_inds = np.where(label_int32 > 0)[0]
    bbox_has_mask = fg_inds.copy()

    if fg_inds.shape[0] > 0:
        labels_fg = label_int32[fg_inds]
        masks_fg = np.zeros((fg_inds.shape[0], resolution**2), dtype=np.int32)
        bbox_fg = boxes[fg_inds]

        iou = bbox_overlaps_mask(bbox_fg, boxes_from_polys)
        fg_polys_inds = np.argmax(iou, axis=1)

        for i in range(bbox_fg.shape[0]):
            poly_gt = _gt_polys[fg_polys_inds[i]]
            roi_fg = bbox_fg[i]

            mask = polys_to_mask_wrt_box(poly_gt, roi_fg, resolution)
            mask = np.array(mask > 0, dtype=np.int32)
            masks_fg[i, :] = np.reshape(mask, resolution**2)
    else:
        bg_inds = np.where(label_int32 == 0)[0]
        bbox_fg = boxes[bg_inds[0]].reshape((1, -1))
        masks_fg = -np.ones((1, resolution**2), dtype=np.int32)
        labels_fg = np.zeros((1, ))
        bbox_has_mask = np.append(bbox_has_mask, 0)
    masks = expand_mask_targets(masks_fg, labels_fg, resolution, num_classes)
    return bbox_fg, bbox_has_mask, masks


@jit
def libra_sample_pos(max_overlaps, max_classes, pos_inds, num_expected):
    if len(pos_inds) <= num_expected:
        return pos_inds
    else:
        unique_gt_inds = np.unique(max_classes[pos_inds])
        num_gts = len(unique_gt_inds)
        num_per_gt = int(round(num_expected / float(num_gts)) + 1)

        sampled_inds = []
        for i in unique_gt_inds:
            inds = np.nonzero(max_classes == i)[0]
            before_len = len(inds)
            inds = list(set(inds) & set(pos_inds))
            after_len = len(inds)
            if len(inds) > num_per_gt:
                inds = np.random.choice(
                    inds, size=num_per_gt, replace=False)
            sampled_inds.extend(list(inds))  # combine as a new sampler
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(
                list(set(pos_inds) - set(sampled_inds)))
            assert len(sampled_inds) + len(extra_inds) == len(pos_inds), \
                "sum of sampled_inds({}) and extra_inds({}) length must be equal with pos_inds({})!".format(
                    len(sampled_inds), len(extra_inds), len(pos_inds))
            if len(extra_inds) > num_extra:
                extra_inds = np.random.choice(
                    extra_inds, size=num_extra, replace=False)
            sampled_inds.extend(extra_inds.tolist())
        elif len(sampled_inds) > num_expected:
            sampled_inds = np.random.choice(
                sampled_inds, size=num_expected, replace=False)
        return sampled_inds


def libra_sample_via_interval(max_overlaps, full_set, num_expected, floor_thr,
                              num_bins, bg_thresh_hi):
    max_iou = max_overlaps.max()
    iou_interval = (max_iou - floor_thr) / num_bins
    per_num_expected = int(num_expected / num_bins)

    sampled_inds = []
    for i in range(num_bins):
        start_iou = floor_thr + i * iou_interval
        end_iou = floor_thr + (i + 1) * iou_interval

        tmp_set = set(
            np.where(
                np.logical_and(max_overlaps >= start_iou, max_overlaps <
                               end_iou))[0])
        tmp_inds = list(tmp_set & full_set)

        if len(tmp_inds) > per_num_expected:
            tmp_sampled_set = np.random.choice(
                tmp_inds, size=per_num_expected, replace=False)
        else:
            tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
        sampled_inds.append(tmp_sampled_set)

    sampled_inds = np.concatenate(sampled_inds)
    if len(sampled_inds) < num_expected:
        num_extra = num_expected - len(sampled_inds)
        extra_inds = np.array(list(full_set - set(sampled_inds)))
        assert len(sampled_inds) + len(extra_inds) == len(full_set), \
            "sum of sampled_inds({}) and extra_inds({}) length must be equal with full_set({})!".format(
                len(sampled_inds), len(extra_inds), len(full_set))

        if len(extra_inds) > num_extra:
            extra_inds = np.random.choice(
                extra_inds, num_extra, replace=False)
        sampled_inds = np.concatenate([sampled_inds, extra_inds])

    return sampled_inds


def libra_sample_neg(max_overlaps,
                     max_classes,
                     neg_inds,
                     num_expected,
                     floor_thr=-1,
                     floor_fraction=0,
                     num_bins=3,
                     bg_thresh_hi=0.5):
    if len(neg_inds) <= num_expected:
        return neg_inds
    else:
        # balance sampling for negative samples
        neg_set = set(neg_inds)
        if floor_thr > 0:
            floor_set = set(
                np.where(
                    np.logical_and(max_overlaps >= 0, max_overlaps <
                                   floor_thr))[0])
            iou_sampling_set = set(
                np.where(max_overlaps >= floor_thr)[0])
        elif floor_thr == 0:
            floor_set = set(np.where(max_overlaps == 0)[0])
            iou_sampling_set = set(
                np.where(max_overlaps > floor_thr)[0])
        else:
            floor_set = set()
            iou_sampling_set = set(
                np.where(max_overlaps > floor_thr)[0])
            floor_thr = 0

        floor_neg_inds = list(floor_set & neg_set)
        iou_sampling_neg_inds = list(iou_sampling_set & neg_set)

        num_expected_iou_sampling = int(num_expected *
                                        (1 - floor_fraction))
        if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
            if num_bins >= 2:
                iou_sampled_inds = libra_sample_via_interval(
                    max_overlaps,
                    set(iou_sampling_neg_inds),
                    num_expected_iou_sampling, floor_thr, num_bins,
                    bg_thresh_hi)
            else:
                iou_sampled_inds = np.random.choice(
                    iou_sampling_neg_inds,
                    size=num_expected_iou_sampling,
                    replace=False)
        else:
            iou_sampled_inds = np.array(
                iou_sampling_neg_inds, dtype=np.int)
        num_expected_floor = num_expected - len(iou_sampled_inds)
        if len(floor_neg_inds) > num_expected_floor:
            sampled_floor_inds = np.random.choice(
                floor_neg_inds, size=num_expected_floor, replace=False)
        else:
            sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
        sampled_inds = np.concatenate(
            (sampled_floor_inds, iou_sampled_inds))
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(neg_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = np.random.choice(
                    extra_inds, size=num_extra, replace=False)
            sampled_inds = np.concatenate((sampled_inds, extra_inds))
        return sampled_inds


@jit
def libra_generate_proposal_target(rpn_rois,
                                   rpn_rois_num,
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
                                   is_cascade_rcnn=False,
                                   max_overlaps=None,
                                   num_bins=3):
    rois = []
    tgt_labels = []
    tgt_deltas = []
    rois_inside_weights = []
    rois_outside_weights = []
    sampled_max_overlaps = []
    new_rois_num = []
    st_num = 0
    end_num = 0
    for im_i in range(len(rpn_rois_num)):
        length = rpn_rois_num[im_i]
        end_num += length
        rpn_roi = rpn_rois[st_num:end_num]
        max_overlap = max_overlaps[st_num:end_num] if is_cascade_rcnn else None

        im_scale = im_info[im_i][2]
        rpn_roi = rpn_roi / im_scale
        gt_bbox = gt_boxes[im_i]
        if is_cascade_rcnn:
            rpn_roi = filter_roi(rpn_roi, max_overlap)
        bbox = np.vstack([gt_bbox, rpn_roi]).astype('float32')
        gt_overlaps = np.zeros((bbox.shape[0], class_nums))

        box_to_gt_ind_map = np.zeros((bbox.shape[0]), dtype=np.int32)
        if len(gt_bbox) > 0:
            proposal_to_gt_overlaps = bbox_overlaps(bbox, gt_bbox)
            overlaps_argmax = proposal_to_gt_overlaps.argmax(axis=1)
            overlaps_max = proposal_to_gt_overlaps.max(axis=1)
            # Boxes which with non-zero overlap with gt boxes
            overlapped_boxes_ind = np.where(overlaps_max > 0)[0]
            gt_classes = gt_classes.reshape(-1, 1)
            overlapped_boxes_gt_classes = gt_classes[overlaps_argmax[
                overlapped_boxes_ind]]

            for idx in range(len(overlapped_boxes_ind)):
                gt_overlaps[overlapped_boxes_ind[
                                idx], overlapped_boxes_gt_classes[idx]] = overlaps_max[
                    overlapped_boxes_ind[idx]]
                box_to_gt_ind_map[overlapped_boxes_ind[
                    idx]] = overlaps_argmax[overlapped_boxes_ind[idx]]

        crowd_ind = np.where(is_crowd)[0]
        gt_overlaps[crowd_ind] = -1

        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)

        # Step2: sample bbox
        rois_per_image = int(batch_size_per_im)
        fg_rois_per_im = int(np.round(fg_fraction * rois_per_image))
        bg_rois_per_im = rois_per_image - fg_rois_per_im

        if is_cascade_rcnn:
            fg_inds = np.where(max_overlaps >= fg_thresh)[0]
            bg_inds = np.where((max_overlaps < bg_thresh_hi) & (max_overlaps >=
                                                                bg_thresh_lo))[0]
            fg_nums = fg_inds.shape[0]
            bg_nums = bg_inds.shape[0]
        else:
            # sampe fg
            fg_inds = np.where(max_overlaps >= fg_thresh)[0]
            fg_nums = np.minimum(fg_rois_per_im, fg_inds.shape[0])
            if (fg_inds.shape[0] > fg_nums) and use_random:
                # fg_inds = np.random.choice(fg_inds, size=fg_nums, replace=False)
                fg_inds = libra_sample_pos(max_overlaps, max_classes,
                                           fg_inds, fg_rois_per_im)
            fg_inds = fg_inds[:fg_nums]

            # sample bg
            bg_inds = np.where((max_overlaps < bg_thresh_hi) & (max_overlaps >=
                                                                bg_thresh_lo))[0]
            bg_nums = rois_per_image - fg_nums
            bg_nums = np.minimum(bg_nums, bg_inds.shape[0])
            if (bg_inds.shape[0] > bg_nums) and use_random:
                # bg_inds = np.random.choice(bg_inds, size=bg_nums, replace=False)
                # libra neg sample
                bg_inds = libra_sample_neg(
                    max_overlaps,
                    max_classes,
                    bg_inds,
                    bg_rois_per_im,
                    num_bins=num_bins,
                    bg_thresh_hi=bg_thresh_hi)
            bg_inds = bg_inds[:bg_nums]

        # Step3: make output
        sampled_inds = np.append(fg_inds, bg_inds)

        sampled_labels = max_classes[sampled_inds]
        sampled_labels[fg_nums:] = 0

        sampled_boxes = bbox[sampled_inds]
        sampled_max_overlap = max_overlaps[sampled_inds]
        sampled_gt_boxes = gt_bbox[box_to_gt_ind_map[sampled_inds]]
        sampled_gt_boxes[fg_nums:, :] = 0
        sampled_deltas = compute_bbox_targets(sampled_boxes, sampled_gt_boxes,
                                              sampled_labels, bbox_reg_weights)
        sampled_deltas[fg_nums:, :] = 0
        sampled_deltas, bbox_inside_weights = expand_bbox_targets(
            sampled_deltas, class_nums, is_cls_agnostic)
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)

        roi = sampled_boxes * im_scale

        st_num += length

        rois.append(roi)
        new_rois_num.append(roi.shape[0])
        tgt_labels.append(sampled_labels)
        tgt_deltas.append(sampled_deltas)
        rois_inside_weights.append(bbox_inside_weights)
        rois_outside_weights.append(bbox_outside_weights)
        sampled_max_overlaps.append(sampled_max_overlap)

    rois = np.concatenate(rois, axis=0).astype(np.float32)
    tgt_labels = np.concatenate(
        tgt_labels, axis=0).astype(np.int32).reshape(-1, 1)
    tgt_deltas = np.concatenate(tgt_deltas, axis=0).astype(np.float32)
    rois_inside_weights = np.concatenate(
        rois_inside_weights, axis=0).astype(np.float32)
    rois_outside_weights = np.concatenate(
        rois_outside_weights, axis=0).astype(np.float32)
    sampled_max_overlaps = np.concatenate(
        sampled_max_overlaps, axis=0).astype(np.float32)
    new_rois_num = np.asarray(new_rois_num, np.int32)
    return rois, tgt_labels, tgt_deltas, rois_inside_weights, rois_outside_weights, new_rois_num, sampled_max_overlaps