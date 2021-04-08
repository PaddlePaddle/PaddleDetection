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
import math
import numpy as np
import paddle
from ..bbox_utils import bbox2delta, bbox_overlaps
import copy


def rpn_anchor_target(anchors,
                      gt_boxes,
                      is_crowd,
                      rpn_batch_size_per_im,
                      rpn_positive_overlap,
                      rpn_negative_overlap,
                      rpn_fg_fraction,
                      use_random=True,
                      batch_size=1,
                      ignore_thresh=-1,
                      weights=[1., 1., 1., 1.]):
    tgt_labels = []
    tgt_bboxes = []
    tgt_deltas = []
    for i in range(batch_size):
        gt_bbox = gt_boxes[i]
        is_crowd_i = is_crowd[i]
        # Step1: match anchor and gt_bbox
        matches, match_labels = label_box(
            anchors, gt_bbox, is_crowd_i, rpn_positive_overlap,
            rpn_negative_overlap, True, ignore_thresh)
        # Step2: sample anchor 
        fg_inds, bg_inds = subsample_labels(match_labels, rpn_batch_size_per_im,
                                            rpn_fg_fraction, 0, use_random)
        # Fill with the ignore label (-1), then set positive and negative labels
        labels = paddle.full(match_labels.shape, -1, dtype='int32')
        if bg_inds.shape[0] > 0:
            labels = paddle.scatter(labels, bg_inds, paddle.zeros_like(bg_inds))
        if fg_inds.shape[0] > 0:
            labels = paddle.scatter(labels, fg_inds, paddle.ones_like(fg_inds))
        # Step3: make output  
        if gt_bbox.shape[0] == 0:
            matched_gt_boxes = paddle.zeros([0, 4])
            tgt_delta = paddle.zeros([0, 4])
        else:
            matched_gt_boxes = paddle.gather(gt_bbox, matches)
            tgt_delta = bbox2delta(anchors, matched_gt_boxes, weights)
            matched_gt_boxes.stop_gradient = True
            tgt_delta.stop_gradient = True
        labels.stop_gradient = True
        tgt_labels.append(labels)
        tgt_bboxes.append(matched_gt_boxes)
        tgt_deltas.append(tgt_delta)

    return tgt_labels, tgt_bboxes, tgt_deltas


def label_box(anchors, gt_boxes, is_crowd, positive_overlap, negative_overlap,
              allow_low_quality, ignore_thresh):
    iou = bbox_overlaps(gt_boxes, anchors)
    N_gt = gt_boxes.shape[0]
    N_gt_crowd = paddle.nonzero(is_crowd).shape[0]
    if iou.numel() == 0 or N_gt_crowd == N_gt:
        # No truth, assign everything to background
        default_matches = paddle.full((iou.shape[1], ), 0, dtype='int64')
        default_match_labels = paddle.full((iou.shape[1], ), 0, dtype='int32')
        return default_matches, default_match_labels
    # if ignore_thresh > 0, remove anchor if it is closed to 
    # one of the crowded ground-truth
    if N_gt_crowd > 0:
        N_a = anchors.shape[0]
        ones = paddle.ones([N_a])
        mask = is_crowd * ones

        if ignore_thresh > 0:
            crowd_iou = iou * mask
            valid = (paddle.sum((crowd_iou > ignore_thresh).cast('int32'),
                                axis=0) > 0).cast('float32')
            iou = iou * (1 - valid) - valid

        # ignore the iou between anchor and crowded ground-truth
        iou = iou * (1 - mask) - mask

    matched_vals, matches = paddle.topk(iou, k=1, axis=0)
    match_labels = paddle.full(matches.shape, -1, dtype='int32')
    # set ignored anchor with iou = -1
    neg_cond = paddle.logical_and(matched_vals > -1,
                                  matched_vals < negative_overlap)
    match_labels = paddle.where(neg_cond,
                                paddle.zeros_like(match_labels), match_labels)
    match_labels = paddle.where(matched_vals >= positive_overlap,
                                paddle.ones_like(match_labels), match_labels)
    if allow_low_quality:
        highest_quality_foreach_gt = iou.max(axis=1, keepdim=True)
        pred_inds_with_highest_quality = paddle.logical_and(
            iou > 0, iou == highest_quality_foreach_gt).cast('int32').sum(
                0, keepdim=True)
        match_labels = paddle.where(pred_inds_with_highest_quality > 0,
                                    paddle.ones_like(match_labels),
                                    match_labels)

    matches = matches.flatten()
    match_labels = match_labels.flatten()

    return matches, match_labels


def subsample_labels(labels,
                     num_samples,
                     fg_fraction,
                     bg_label=0,
                     use_random=True):
    positive = paddle.nonzero(
        paddle.logical_and(labels != -1, labels != bg_label))
    negative = paddle.nonzero(labels == bg_label)

    fg_num = int(num_samples * fg_fraction)
    fg_num = min(positive.numel(), fg_num)
    bg_num = num_samples - fg_num
    bg_num = min(negative.numel(), bg_num)
    if fg_num == 0 and bg_num == 0:
        fg_inds = paddle.zeros([0], dtype='int32')
        bg_inds = paddle.zeros([0], dtype='int32')
        return fg_inds, bg_inds

    # randomly select positive and negative examples

    negative = negative.cast('int32').flatten()
    bg_perm = paddle.randperm(negative.numel(), dtype='int32')
    bg_perm = paddle.slice(bg_perm, axes=[0], starts=[0], ends=[bg_num])
    if use_random:
        bg_inds = paddle.gather(negative, bg_perm)
    else:
        bg_inds = paddle.slice(negative, axes=[0], starts=[0], ends=[bg_num])
    if fg_num == 0:
        fg_inds = paddle.zeros([0], dtype='int32')
        return fg_inds, bg_inds

    positive = positive.cast('int32').flatten()
    fg_perm = paddle.randperm(positive.numel(), dtype='int32')
    fg_perm = paddle.slice(fg_perm, axes=[0], starts=[0], ends=[fg_num])
    if use_random:
        fg_inds = paddle.gather(positive, fg_perm)
    else:
        fg_inds = paddle.slice(positive, axes=[0], starts=[0], ends=[fg_num])

    return fg_inds, bg_inds


def generate_proposal_target(rpn_rois,
                             gt_classes,
                             gt_boxes,
                             is_crowd,
                             batch_size_per_im,
                             fg_fraction,
                             fg_thresh,
                             bg_thresh,
                             num_classes,
                             ignore_thresh=-1.,
                             use_random=True,
                             is_cascade=False,
                             cascade_iou=0.5):

    rois_with_gt = []
    tgt_labels = []
    tgt_bboxes = []
    tgt_gt_inds = []
    new_rois_num = []

    # In cascade rcnn, the threshold for foreground and background
    # is used from cascade_iou
    fg_thresh = cascade_iou if is_cascade else fg_thresh
    bg_thresh = cascade_iou if is_cascade else bg_thresh
    for i, rpn_roi in enumerate(rpn_rois):
        gt_bbox = gt_boxes[i]
        is_crowd_i = is_crowd[i]
        gt_class = paddle.squeeze(gt_classes[i], axis=-1)

        # Concat RoIs and gt boxes except cascade rcnn or none gt
        if not is_cascade and gt_bbox.shape[0] > 0:
            bbox = paddle.concat([rpn_roi, gt_bbox])
        else:
            bbox = rpn_roi

        # Step1: label bbox
        matches, match_labels = label_box(bbox, gt_bbox, is_crowd_i, fg_thresh,
                                          bg_thresh, False, ignore_thresh)
        # Step2: sample bbox 
        sampled_inds, sampled_gt_classes = sample_bbox(
            matches, match_labels, gt_class, batch_size_per_im, fg_fraction,
            num_classes, use_random, is_cascade)

        # Step3: make output 
        rois_per_image = bbox if is_cascade else paddle.gather(bbox,
                                                               sampled_inds)
        sampled_gt_ind = matches if is_cascade else paddle.gather(matches,
                                                                  sampled_inds)
        if gt_bbox.shape[0] > 0:
            sampled_bbox = paddle.gather(gt_bbox, sampled_gt_ind)
        else:
            sampled_bbox = paddle.zeros([0, 4], dtype='float32')

        rois_per_image.stop_gradient = True
        sampled_gt_ind.stop_gradient = True
        sampled_bbox.stop_gradient = True
        tgt_labels.append(sampled_gt_classes)
        tgt_bboxes.append(sampled_bbox)
        rois_with_gt.append(rois_per_image)
        tgt_gt_inds.append(sampled_gt_ind)
        new_rois_num.append(paddle.shape(sampled_inds)[0])
    new_rois_num = paddle.concat(new_rois_num)
    return rois_with_gt, tgt_labels, tgt_bboxes, tgt_gt_inds, new_rois_num


def sample_bbox(matches,
                match_labels,
                gt_classes,
                batch_size_per_im,
                fg_fraction,
                num_classes,
                use_random=True,
                is_cascade=False):

    N_gt = gt_classes.shape[0]
    if N_gt == 0:
        # No truth, assign everything to background
        gt_classes = paddle.ones(matches.shape, dtype='int32') * num_classes
        #return matches, match_labels + num_classes
    else:
        gt_classes = paddle.gather(gt_classes, matches)
        gt_classes = paddle.where(match_labels == 0,
                                  paddle.ones_like(gt_classes) * num_classes,
                                  gt_classes)
        gt_classes = paddle.where(match_labels == -1,
                                  paddle.ones_like(gt_classes) * -1, gt_classes)
    if is_cascade:
        index = paddle.arange(matches.shape[0])
        return index, gt_classes
    rois_per_image = int(batch_size_per_im)

    fg_inds, bg_inds = subsample_labels(gt_classes, rois_per_image, fg_fraction,
                                        num_classes, use_random)
    if fg_inds.shape[0] == 0 and bg_inds.shape[0] == 0:
        # fake output labeled with -1 when all boxes are neither
        # foreground nor background
        sampled_inds = paddle.zeros([1], dtype='int32')
    else:
        sampled_inds = paddle.concat([fg_inds, bg_inds])
    sampled_gt_classes = paddle.gather(gt_classes, sampled_inds)
    return sampled_inds, sampled_gt_classes


def polygons_to_mask(polygons, height, width):
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)
    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    import pycocotools.mask as mask_util
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool)


def rasterize_polygons_within_box(poly, box, resolution):
    w, h = box[2] - box[0], box[3] - box[1]
    polygons = [np.asarray(p, dtype=np.float64) for p in poly]
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    ratio_h = resolution / max(h, 0.1)
    ratio_w = resolution / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_mask(polygons, resolution, resolution)
    mask = paddle.to_tensor(mask, dtype='int32')
    return mask


def generate_mask_target(gt_segms, rois, labels_int32, sampled_gt_inds,
                         num_classes, resolution):
    mask_rois = []
    mask_rois_num = []
    tgt_masks = []
    tgt_classes = []
    mask_index = []
    tgt_weights = []
    for k in range(len(rois)):
        labels_per_im = labels_int32[k]
        # select rois labeled with foreground
        fg_inds = paddle.nonzero(
            paddle.logical_and(labels_per_im != -1, labels_per_im !=
                               num_classes))
        has_fg = True
        # generate fake roi if foreground is empty
        if fg_inds.numel() == 0:
            has_fg = False
            fg_inds = paddle.ones([1], dtype='int32')
        inds_per_im = sampled_gt_inds[k]
        inds_per_im = paddle.gather(inds_per_im, fg_inds)

        rois_per_im = rois[k]
        fg_rois = paddle.gather(rois_per_im, fg_inds)
        # Copy the foreground roi to cpu
        # to generate mask target with ground-truth
        boxes = fg_rois.numpy()
        gt_segms_per_im = gt_segms[k]
        new_segm = []
        inds_per_im = inds_per_im.numpy()
        for i in inds_per_im:
            new_segm.append(gt_segms_per_im[i])
        fg_inds_new = fg_inds.reshape([-1]).numpy()
        results = []
        for j in fg_inds_new:
            results.append(
                rasterize_polygons_within_box(new_segm[j], boxes[j],
                                              resolution))

        fg_classes = paddle.gather(labels_per_im, fg_inds)
        weight = paddle.ones([fg_rois.shape[0]], dtype='float32')
        if not has_fg:
            weight = weight - 1
        tgt_mask = paddle.stack(results)
        tgt_mask.stop_gradient = True
        fg_rois.stop_gradient = True

        mask_index.append(fg_inds)
        mask_rois.append(fg_rois)
        mask_rois_num.append(paddle.shape(fg_rois)[0])
        tgt_classes.append(fg_classes)
        tgt_masks.append(tgt_mask)
        tgt_weights.append(weight)

    mask_index = paddle.concat(mask_index)
    mask_rois_num = paddle.concat(mask_rois_num)
    tgt_classes = paddle.concat(tgt_classes, axis=0)
    tgt_masks = paddle.concat(tgt_masks, axis=0)
    tgt_weights = paddle.concat(tgt_weights, axis=0)

    return mask_rois, mask_rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights
