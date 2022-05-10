# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import paddle
from ppdet.core.workspace import register, serializable

from .target import rpn_anchor_target, generate_proposal_target, generate_mask_target, libra_generate_proposal_target
import numpy as np


@register
@serializable
class RPNTargetAssign(object):
    __shared__ = ['assign_on_cpu']
    """
    RPN targets assignment module

    The assignment consists of three steps:
        1. Match anchor and ground-truth box, label the anchor with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and 
           background
        3. Generate the targets for classification and regression branch


    Args:
        batch_size_per_im (int): Total number of RPN samples per image. 
            default 256
        fg_fraction (float): Fraction of anchors that is labeled
            foreground, default 0.5
        positive_overlap (float): Minimum overlap required between an anchor
            and ground-truth box for the (anchor, gt box) pair to be 
            a foreground sample. default 0.7
        negative_overlap (float): Maximum overlap allowed between an anchor
            and ground-truth box for the (anchor, gt box) pair to be 
            a background sample. default 0.3
        ignore_thresh(float): Threshold for ignoring the is_crowd ground-truth
            if the value is larger than zero.
        use_random (bool): Use random sampling to choose foreground and 
            background boxes, default true.
        assign_on_cpu (bool): In case the number of gt box is too large, 
            compute IoU on CPU, default false.
    """

    def __init__(self,
                 batch_size_per_im=256,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 ignore_thresh=-1.,
                 use_random=True,
                 assign_on_cpu=False):
        super(RPNTargetAssign, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.ignore_thresh = ignore_thresh
        self.use_random = use_random
        self.assign_on_cpu = assign_on_cpu

    def __call__(self, inputs, anchors):
        """
        inputs: ground-truth instances.
        anchor_box (Tensor): [num_anchors, 4], num_anchors are all anchors in all feature maps.
        """
        gt_boxes = inputs['gt_bbox']
        is_crowd = inputs.get('is_crowd', None)
        batch_size = len(gt_boxes)
        tgt_labels, tgt_bboxes, tgt_deltas = rpn_anchor_target(
            anchors,
            gt_boxes,
            self.batch_size_per_im,
            self.positive_overlap,
            self.negative_overlap,
            self.fg_fraction,
            self.use_random,
            batch_size,
            self.ignore_thresh,
            is_crowd,
            assign_on_cpu=self.assign_on_cpu)
        norm = self.batch_size_per_im * batch_size

        return tgt_labels, tgt_bboxes, tgt_deltas, norm


@register
class BBoxAssigner(object):
    __shared__ = ['num_classes', 'assign_on_cpu']
    """
    RCNN targets assignment module

    The assignment consists of three steps:
        1. Match RoIs and ground-truth box, label the RoIs with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and 
           background
        3. Generate the targets for classification and regression branch

    Args:
        batch_size_per_im (int): Total number of RoIs per image. 
            default 512 
        fg_fraction (float): Fraction of RoIs that is labeled
            foreground, default 0.25
        fg_thresh (float): Minimum overlap required between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a foreground sample. default 0.5
        bg_thresh (float): Maximum overlap allowed between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a background sample. default 0.5
        ignore_thresh(float): Threshold for ignoring the is_crowd ground-truth
            if the value is larger than zero.
        use_random (bool): Use random sampling to choose foreground and 
            background boxes, default true
        cascade_iou (list[iou]): The list of overlap to select foreground and
            background of each stage, which is only used In Cascade RCNN.
        num_classes (int): The number of class.
        assign_on_cpu (bool): In case the number of gt box is too large, 
            compute IoU on CPU, default false.
    """

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh=.5,
                 ignore_thresh=-1.,
                 use_random=True,
                 cascade_iou=[0.5, 0.6, 0.7],
                 num_classes=80,
                 assign_on_cpu=False):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.ignore_thresh = ignore_thresh
        self.use_random = use_random
        self.cascade_iou = cascade_iou
        self.num_classes = num_classes
        self.assign_on_cpu = assign_on_cpu

    def __call__(self,
                 rpn_rois,
                 rpn_rois_num,
                 inputs,
                 stage=0,
                 is_cascade=False):
        gt_classes = inputs['gt_class']
        gt_boxes = inputs['gt_bbox']
        is_crowd = inputs.get('is_crowd', None)
        # rois, tgt_labels, tgt_bboxes, tgt_gt_inds
        # new_rois_num
        outs = generate_proposal_target(
            rpn_rois, gt_classes, gt_boxes, self.batch_size_per_im,
            self.fg_fraction, self.fg_thresh, self.bg_thresh, self.num_classes,
            self.ignore_thresh, is_crowd, self.use_random, is_cascade,
            self.cascade_iou[stage], self.assign_on_cpu)
        rois = outs[0]
        rois_num = outs[-1]
        # tgt_labels, tgt_bboxes, tgt_gt_inds
        targets = outs[1:4]
        return rois, rois_num, targets


@register
class BBoxLibraAssigner(object):
    __shared__ = ['num_classes']
    """
    Libra-RCNN targets assignment module

    The assignment consists of three steps:
        1. Match RoIs and ground-truth box, label the RoIs with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and
           background
        3. Generate the targets for classification and regression branch

    Args:
        batch_size_per_im (int): Total number of RoIs per image.
            default 512
        fg_fraction (float): Fraction of RoIs that is labeled
            foreground, default 0.25
        fg_thresh (float): Minimum overlap required between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a foreground sample. default 0.5
        bg_thresh (float): Maximum overlap allowed between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a background sample. default 0.5
        use_random (bool): Use random sampling to choose foreground and
            background boxes, default true
        cascade_iou (list[iou]): The list of overlap to select foreground and
            background of each stage, which is only used In Cascade RCNN.
        num_classes (int): The number of class.
        num_bins (int): The number of libra_sample.
    """

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh=.5,
                 use_random=True,
                 cascade_iou=[0.5, 0.6, 0.7],
                 num_classes=80,
                 num_bins=3):
        super(BBoxLibraAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.use_random = use_random
        self.cascade_iou = cascade_iou
        self.num_classes = num_classes
        self.num_bins = num_bins

    def __call__(self,
                 rpn_rois,
                 rpn_rois_num,
                 inputs,
                 stage=0,
                 is_cascade=False):
        gt_classes = inputs['gt_class']
        gt_boxes = inputs['gt_bbox']
        # rois, tgt_labels, tgt_bboxes, tgt_gt_inds
        outs = libra_generate_proposal_target(
            rpn_rois, gt_classes, gt_boxes, self.batch_size_per_im,
            self.fg_fraction, self.fg_thresh, self.bg_thresh, self.num_classes,
            self.use_random, is_cascade, self.cascade_iou[stage], self.num_bins)
        rois = outs[0]
        rois_num = outs[-1]
        # tgt_labels, tgt_bboxes, tgt_gt_inds
        targets = outs[1:4]
        return rois, rois_num, targets


@register
@serializable
class MaskAssigner(object):
    __shared__ = ['num_classes', 'mask_resolution']
    """
    Mask targets assignment module

    The assignment consists of three steps:
        1. Select RoIs labels with foreground.
        2. Encode the RoIs and corresponding gt polygons to generate 
           mask target

    Args:
        num_classes (int): The number of class
        mask_resolution (int): The resolution of mask target, default 14
    """

    def __init__(self, num_classes=80, mask_resolution=14):
        super(MaskAssigner, self).__init__()
        self.num_classes = num_classes
        self.mask_resolution = mask_resolution

    def __call__(self, rois, tgt_labels, tgt_gt_inds, inputs):
        gt_segms = inputs['gt_poly']

        outs = generate_mask_target(gt_segms, rois, tgt_labels, tgt_gt_inds,
                                    self.num_classes, self.mask_resolution)

        # mask_rois, mask_rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights
        return outs


@register
class RBoxAssigner(object):
    """
    assigner of rbox
    Args:
        pos_iou_thr (float): threshold of pos samples
        neg_iou_thr (float): threshold of neg samples
        min_iou_thr (float): the min threshold of samples
        ignore_iof_thr (int): the ignored threshold
    """

    def __init__(self,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.4,
                 min_iou_thr=0.0,
                 ignore_iof_thr=-2):
        super(RBoxAssigner, self).__init__()

        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_iou_thr = min_iou_thr
        self.ignore_iof_thr = ignore_iof_thr

    def anchor_valid(self, anchors):
        """

        Args:
            anchor: M x 4

        Returns:

        """
        if anchors.ndim == 3:
            anchors = anchors.reshape(-1, anchors.shape[-1])
        assert anchors.ndim == 2
        anchor_num = anchors.shape[0]
        anchor_valid = np.ones((anchor_num), np.int32)
        anchor_inds = np.arange(anchor_num)
        return anchor_inds

    def rbox2delta(self,
                   proposals,
                   gt,
                   means=[0, 0, 0, 0, 0],
                   stds=[1, 1, 1, 1, 1]):
        """
        Args:
            proposals: tensor [N, 5]
            gt: gt [N, 5]
            means: means [5]
            stds: stds [5]
        Returns:

        """
        proposals = proposals.astype(np.float64)

        PI = np.pi

        gt_widths = gt[..., 2]
        gt_heights = gt[..., 3]
        gt_angle = gt[..., 4]

        proposals_widths = proposals[..., 2]
        proposals_heights = proposals[..., 3]
        proposals_angle = proposals[..., 4]

        coord = gt[..., 0:2] - proposals[..., 0:2]
        dx = (np.cos(proposals[..., 4]) * coord[..., 0] +
              np.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
        dy = (-np.sin(proposals[..., 4]) * coord[..., 0] +
              np.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
        dw = np.log(gt_widths / proposals_widths)
        dh = np.log(gt_heights / proposals_heights)
        da = (gt_angle - proposals_angle)

        da = (da + PI / 4) % PI - PI / 4
        da /= PI

        deltas = np.stack([dx, dy, dw, dh, da], axis=-1)
        means = np.array(means, dtype=deltas.dtype)
        stds = np.array(stds, dtype=deltas.dtype)
        deltas = (deltas - means) / stds
        deltas = deltas.astype(np.float32)
        return deltas

    def assign_anchor(self,
                      anchors,
                      gt_bboxes,
                      gt_lables,
                      pos_iou_thr,
                      neg_iou_thr,
                      min_iou_thr=0.0,
                      ignore_iof_thr=-2):
        """

        Args:
            anchors:
            gt_bboxes:[M, 5] rc,yc,w,h,angle
            gt_lables:

        Returns:

        """
        assert anchors.shape[1] == 4 or anchors.shape[1] == 5
        assert gt_bboxes.shape[1] == 4 or gt_bboxes.shape[1] == 5
        anchors_xc_yc = anchors
        gt_bboxes_xc_yc = gt_bboxes

        # calc rbox iou
        anchors_xc_yc = anchors_xc_yc.astype(np.float32)
        gt_bboxes_xc_yc = gt_bboxes_xc_yc.astype(np.float32)
        anchors_xc_yc = paddle.to_tensor(anchors_xc_yc)
        gt_bboxes_xc_yc = paddle.to_tensor(gt_bboxes_xc_yc)

        try:
            from rbox_iou_ops import rbox_iou
        except Exception as e:
            print("import custom_ops error, try install rbox_iou_ops " \
                  "following ppdet/ext_op/README.md", e)
            sys.stdout.flush()
            sys.exit(-1)

        iou = rbox_iou(gt_bboxes_xc_yc, anchors_xc_yc)
        iou = iou.numpy()
        iou = iou.T

        # every gt's anchor's index
        gt_bbox_anchor_inds = iou.argmax(axis=0)
        gt_bbox_anchor_iou = iou[gt_bbox_anchor_inds, np.arange(iou.shape[1])]
        gt_bbox_anchor_iou_inds = np.where(iou == gt_bbox_anchor_iou)[0]

        # every anchor's gt bbox's index
        anchor_gt_bbox_inds = iou.argmax(axis=1)
        anchor_gt_bbox_iou = iou[np.arange(iou.shape[0]), anchor_gt_bbox_inds]

        # (1) set labels=-2 as default
        labels = np.ones((iou.shape[0], ), dtype=np.int32) * ignore_iof_thr

        # (2) assign ignore
        labels[anchor_gt_bbox_iou < min_iou_thr] = ignore_iof_thr

        # (3) assign neg_ids -1
        assign_neg_ids1 = anchor_gt_bbox_iou >= min_iou_thr
        assign_neg_ids2 = anchor_gt_bbox_iou < neg_iou_thr
        assign_neg_ids = np.logical_and(assign_neg_ids1, assign_neg_ids2)
        labels[assign_neg_ids] = -1

        # anchor_gt_bbox_iou_inds
        # (4) assign max_iou as pos_ids >=0
        anchor_gt_bbox_iou_inds = anchor_gt_bbox_inds[gt_bbox_anchor_iou_inds]
        # gt_bbox_anchor_iou_inds = np.logical_and(gt_bbox_anchor_iou_inds, anchor_gt_bbox_iou >= min_iou_thr)
        labels[gt_bbox_anchor_iou_inds] = gt_lables[anchor_gt_bbox_iou_inds]

        # (5) assign >= pos_iou_thr as pos_ids
        iou_pos_iou_thr_ids = anchor_gt_bbox_iou >= pos_iou_thr
        iou_pos_iou_thr_ids_box_inds = anchor_gt_bbox_inds[iou_pos_iou_thr_ids]
        labels[iou_pos_iou_thr_ids] = gt_lables[iou_pos_iou_thr_ids_box_inds]
        return anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels

    def __call__(self, anchors, gt_bboxes, gt_labels, is_crowd):

        assert anchors.ndim == 2
        assert anchors.shape[1] == 5
        assert gt_bboxes.ndim == 2
        assert gt_bboxes.shape[1] == 5

        pos_iou_thr = self.pos_iou_thr
        neg_iou_thr = self.neg_iou_thr
        min_iou_thr = self.min_iou_thr
        ignore_iof_thr = self.ignore_iof_thr

        anchor_num = anchors.shape[0]

        gt_bboxes = gt_bboxes
        is_crowd_slice = is_crowd
        not_crowd_inds = np.where(is_crowd_slice == 0)

        # Step1: match anchor and gt_bbox
        anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels = self.assign_anchor(
            anchors, gt_bboxes,
            gt_labels.reshape(-1), pos_iou_thr, neg_iou_thr, min_iou_thr,
            ignore_iof_thr)

        # Step2: sample anchor
        pos_inds = np.where(labels >= 0)[0]
        neg_inds = np.where(labels == -1)[0]

        # Step3: make output
        anchors_num = anchors.shape[0]
        bbox_targets = np.zeros_like(anchors)
        bbox_weights = np.zeros_like(anchors)
        bbox_gt_bboxes = np.zeros_like(anchors)
        pos_labels = np.zeros(anchors_num, dtype=np.int32)
        pos_labels_weights = np.zeros(anchors_num, dtype=np.float32)

        pos_sampled_anchors = anchors[pos_inds]
        pos_sampled_gt_boxes = gt_bboxes[anchor_gt_bbox_inds[pos_inds]]
        if len(pos_inds) > 0:
            pos_bbox_targets = self.rbox2delta(pos_sampled_anchors,
                                               pos_sampled_gt_boxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_gt_bboxes[pos_inds, :] = pos_sampled_gt_boxes
            bbox_weights[pos_inds, :] = 1.0

            pos_labels[pos_inds] = labels[pos_inds]
            pos_labels_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            pos_labels_weights[neg_inds] = 1.0
        return (pos_labels, pos_labels_weights, bbox_targets, bbox_weights,
                bbox_gt_bboxes, pos_inds, neg_inds)
