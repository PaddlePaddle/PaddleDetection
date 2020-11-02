# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from ppdet.core.workspace import register
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
logger = logging.getLogger(__name__)

__all__ = ['YOLOv3Loss']


@register
class YOLOv3Loss(object):
    """
    Combined loss for YOLOv3 network

    Args:
        train_batch_size (int): training batch size
        ignore_thresh (float): threshold to ignore confidence loss
        label_smooth (bool): whether to use label smoothing
        use_fine_grained_loss (bool): whether use fine grained YOLOv3 loss
                                      instead of fluid.layers.yolov3_loss
    """
    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['use_fine_grained_loss', 'train_batch_size']

    def __init__(
            self,
            train_batch_size=8,
            batch_size=-1,  # stub for backward compatable
            ignore_thresh=0.7,
            label_smooth=True,
            use_fine_grained_loss=False,
            iou_loss=None,
            iou_aware_loss=None,
            downsample=[32, 16, 8],
            scale_x_y=1.,
            match_score=False):
        self._train_batch_size = train_batch_size
        self._ignore_thresh = ignore_thresh
        self._label_smooth = label_smooth
        self._use_fine_grained_loss = use_fine_grained_loss
        self._iou_loss = iou_loss
        self._iou_aware_loss = iou_aware_loss
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.match_score = match_score

        if batch_size != -1:
            logger.warn(
                "config YOLOv3Loss.batch_size is deprecated, "
                "training batch size should be set by TrainReader.batch_size")

    def __call__(self, outputs, gt_box, gt_label, gt_score, targets, anchors,
                 anchor_masks, mask_anchors, num_classes, prefix_name):
        if self._use_fine_grained_loss:
            return self._get_fine_grained_loss(
                outputs, targets, gt_box, self._train_batch_size, num_classes,
                mask_anchors, self._ignore_thresh)
        else:
            losses = []
            for i, output in enumerate(outputs):
                scale_x_y = self.scale_x_y if not isinstance(
                    self.scale_x_y, Sequence) else self.scale_x_y[i]
                anchor_mask = anchor_masks[i]
                loss = fluid.layers.yolov3_loss(
                    x=output,
                    gt_box=gt_box,
                    gt_label=gt_label,
                    gt_score=gt_score,
                    anchors=anchors,
                    anchor_mask=anchor_mask,
                    class_num=num_classes,
                    ignore_thresh=self._ignore_thresh,
                    downsample_ratio=self.downsample[i],
                    use_label_smooth=self._label_smooth,
                    scale_x_y=scale_x_y,
                    name=prefix_name + "yolo_loss" + str(i))

                losses.append(fluid.layers.reduce_mean(loss))

            return {'loss': sum(losses)}

    def _get_fine_grained_loss(self,
                               outputs,
                               targets,
                               gt_box,
                               train_batch_size,
                               num_classes,
                               mask_anchors,
                               ignore_thresh,
                               eps=1.e-10):
        """
        Calculate fine grained YOLOv3 loss

        Args:
            outputs ([Variables]): List of Variables, output of backbone stages
            targets ([Variables]): List of Variables, The targets for yolo
                                   loss calculatation.
            gt_box (Variable): The ground-truth boudding boxes.
            train_batch_size (int): The training batch size
            num_classes (int): class num of dataset
            mask_anchors ([[float]]): list of anchors in each output layer
            ignore_thresh (float): prediction bbox overlap any gt_box greater
                                   than ignore_thresh, objectness loss will
                                   be ignored.

        Returns:
            Type: dict
                xy_loss (Variable): YOLOv3 (x, y) coordinates loss
                wh_loss (Variable): YOLOv3 (w, h) coordinates loss
                obj_loss (Variable): YOLOv3 objectness score loss
                cls_loss (Variable): YOLOv3 classification loss

        """

        assert len(outputs) == len(targets), \
            "YOLOv3 output layer number not equal target number"

        loss_xys, loss_whs, loss_objs, loss_clss = [], [], [], []
        if self._iou_loss is not None:
            loss_ious = []
        if self._iou_aware_loss is not None:
            loss_iou_awares = []
        for i, (output, target,
                anchors) in enumerate(zip(outputs, targets, mask_anchors)):
            downsample = self.downsample[i]
            an_num = len(anchors) // 2
            if self._iou_aware_loss is not None:
                ioup, output = self._split_ioup(output, an_num, num_classes)
            x, y, w, h, obj, cls = self._split_output(output, an_num,
                                                      num_classes)
            tx, ty, tw, th, tscale, tobj, tcls = self._split_target(target)

            tscale_tobj = tscale * tobj

            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]

            if (abs(scale_x_y - 1.0) < eps):
                loss_x = fluid.layers.sigmoid_cross_entropy_with_logits(
                    x, tx) * tscale_tobj
                loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
                loss_y = fluid.layers.sigmoid_cross_entropy_with_logits(
                    y, ty) * tscale_tobj
                loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])
            else:
                dx = scale_x_y * fluid.layers.sigmoid(x) - 0.5 * (scale_x_y -
                                                                  1.0)
                dy = scale_x_y * fluid.layers.sigmoid(y) - 0.5 * (scale_x_y -
                                                                  1.0)
                loss_x = fluid.layers.abs(dx - tx) * tscale_tobj
                loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
                loss_y = fluid.layers.abs(dy - ty) * tscale_tobj
                loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])

            # NOTE: we refined loss function of (w, h) as L1Loss
            loss_w = fluid.layers.abs(w - tw) * tscale_tobj
            loss_w = fluid.layers.reduce_sum(loss_w, dim=[1, 2, 3])
            loss_h = fluid.layers.abs(h - th) * tscale_tobj
            loss_h = fluid.layers.reduce_sum(loss_h, dim=[1, 2, 3])
            if self._iou_loss is not None:
                loss_iou = self._iou_loss(x, y, w, h, tx, ty, tw, th, anchors,
                                          downsample, self._train_batch_size,
                                          scale_x_y)
                loss_iou = loss_iou * tscale_tobj
                loss_iou = fluid.layers.reduce_sum(loss_iou, dim=[1, 2, 3])
                loss_ious.append(fluid.layers.reduce_mean(loss_iou))

            if self._iou_aware_loss is not None:
                loss_iou_aware = self._iou_aware_loss(
                    ioup, x, y, w, h, tx, ty, tw, th, anchors, downsample,
                    self._train_batch_size, scale_x_y)
                loss_iou_aware = loss_iou_aware * tobj
                loss_iou_aware = fluid.layers.reduce_sum(
                    loss_iou_aware, dim=[1, 2, 3])
                loss_iou_awares.append(fluid.layers.reduce_mean(loss_iou_aware))

            loss_obj_pos, loss_obj_neg = self._calc_obj_loss(
                output, obj, tobj, gt_box, self._train_batch_size, anchors,
                num_classes, downsample, self._ignore_thresh, scale_x_y)

            loss_cls = fluid.layers.sigmoid_cross_entropy_with_logits(cls, tcls)
            loss_cls = fluid.layers.elementwise_mul(loss_cls, tobj, axis=0)
            loss_cls = fluid.layers.reduce_sum(loss_cls, dim=[1, 2, 3, 4])

            loss_xys.append(fluid.layers.reduce_mean(loss_x + loss_y))
            loss_whs.append(fluid.layers.reduce_mean(loss_w + loss_h))
            loss_objs.append(
                fluid.layers.reduce_mean(loss_obj_pos + loss_obj_neg))
            loss_clss.append(fluid.layers.reduce_mean(loss_cls))

        losses_all = {
            "loss_xy": fluid.layers.sum(loss_xys),
            "loss_wh": fluid.layers.sum(loss_whs),
            "loss_obj": fluid.layers.sum(loss_objs),
            "loss_cls": fluid.layers.sum(loss_clss),
        }
        if self._iou_loss is not None:
            losses_all["loss_iou"] = fluid.layers.sum(loss_ious)
        if self._iou_aware_loss is not None:
            losses_all["loss_iou_aware"] = fluid.layers.sum(loss_iou_awares)
        return losses_all

    def _split_ioup(self, output, an_num, num_classes):
        """
        Split output feature map to output, predicted iou
        along channel dimension
        """
        ioup = fluid.layers.slice(output, axes=[1], starts=[0], ends=[an_num])
        ioup = fluid.layers.sigmoid(ioup)
        oriout = fluid.layers.slice(
            output,
            axes=[1],
            starts=[an_num],
            ends=[an_num * (num_classes + 6)])
        return (ioup, oriout)

    def _split_output(self, output, an_num, num_classes):
        """
        Split output feature map to x, y, w, h, objectness, classification
        along channel dimension
        """
        x = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[0],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        y = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[1],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        w = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[2],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        h = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[3],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        obj = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[4],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        clss = []
        stride = output.shape[1] // an_num
        for m in range(an_num):
            clss.append(
                fluid.layers.slice(
                    output,
                    axes=[1],
                    starts=[stride * m + 5],
                    ends=[stride * m + 5 + num_classes]))
        cls = fluid.layers.transpose(
            fluid.layers.stack(
                clss, axis=1), perm=[0, 1, 3, 4, 2])

        return (x, y, w, h, obj, cls)

    def _split_target(self, target):
        """
        split target to x, y, w, h, objectness, classification
        along dimension 2

        target is in shape [N, an_num, 6 + class_num, H, W]
        """
        tx = target[:, :, 0, :, :]
        ty = target[:, :, 1, :, :]
        tw = target[:, :, 2, :, :]
        th = target[:, :, 3, :, :]

        tscale = target[:, :, 4, :, :]
        tobj = target[:, :, 5, :, :]

        tcls = fluid.layers.transpose(
            target[:, :, 6:, :, :], perm=[0, 1, 3, 4, 2])
        tcls.stop_gradient = True

        return (tx, ty, tw, th, tscale, tobj, tcls)

    def _calc_obj_loss(self, output, obj, tobj, gt_box, batch_size, anchors,
                       num_classes, downsample, ignore_thresh, scale_x_y):
        # A prediction bbox overlap any gt_bbox over ignore_thresh, 
        # objectness loss will be ignored, process as follows:

        # 1. get pred bbox, which is same with YOLOv3 infer mode, use yolo_box here
        # NOTE: img_size is set as 1.0 to get noramlized pred bbox
        bbox, prob = fluid.layers.yolo_box(
            x=output,
            img_size=fluid.layers.ones(
                shape=[batch_size, 2], dtype="int32"),
            anchors=anchors,
            class_num=num_classes,
            conf_thresh=0.,
            downsample_ratio=downsample,
            clip_bbox=False,
            scale_x_y=scale_x_y)

        # 2. split pred bbox and gt bbox by sample, calculate IoU between pred bbox
        #    and gt bbox in each sample
        if batch_size > 1:
            preds = fluid.layers.split(bbox, batch_size, dim=0)
            gts = fluid.layers.split(gt_box, batch_size, dim=0)
        else:
            preds = [bbox]
            gts = [gt_box]
            probs = [prob]
        ious = []
        for pred, gt in zip(preds, gts):

            def box_xywh2xyxy(box):
                x = box[:, 0]
                y = box[:, 1]
                w = box[:, 2]
                h = box[:, 3]
                return fluid.layers.stack(
                    [
                        x - w / 2.,
                        y - h / 2.,
                        x + w / 2.,
                        y + h / 2.,
                    ], axis=1)

            pred = fluid.layers.squeeze(pred, axes=[0])
            gt = box_xywh2xyxy(fluid.layers.squeeze(gt, axes=[0]))
            ious.append(fluid.layers.iou_similarity(pred, gt))

        iou = fluid.layers.stack(ious, axis=0)
        # 3. Get iou_mask by IoU between gt bbox and prediction bbox,
        #    Get obj_mask by tobj(holds gt_score), calculate objectness loss

        max_iou = fluid.layers.reduce_max(iou, dim=-1)
        iou_mask = fluid.layers.cast(max_iou <= ignore_thresh, dtype="float32")
        if self.match_score:
            max_prob = fluid.layers.reduce_max(prob, dim=-1)
            iou_mask = iou_mask * fluid.layers.cast(
                max_prob <= 0.25, dtype="float32")
        output_shape = fluid.layers.shape(output)
        an_num = len(anchors) // 2
        iou_mask = fluid.layers.reshape(iou_mask, (-1, an_num, output_shape[2],
                                                   output_shape[3]))
        iou_mask.stop_gradient = True

        # NOTE: tobj holds gt_score, obj_mask holds object existence mask
        obj_mask = fluid.layers.cast(tobj > 0., dtype="float32")
        obj_mask.stop_gradient = True

        # For positive objectness grids, objectness loss should be calculated
        # For negative objectness grids, objectness loss is calculated only iou_mask == 1.0
        loss_obj = fluid.layers.sigmoid_cross_entropy_with_logits(obj, obj_mask)
        loss_obj_pos = fluid.layers.reduce_sum(loss_obj * tobj, dim=[1, 2, 3])
        loss_obj_neg = fluid.layers.reduce_sum(
            loss_obj * (1.0 - obj_mask) * iou_mask, dim=[1, 2, 3])

        return loss_obj_pos, loss_obj_neg
