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
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NumpyArrayInitializer

from paddle import fluid
from ppdet.core.workspace import register

__all__ = ['YOLOv3Loss']


@register
class YOLOv3Loss(object):
    """
    Combined loss for YOLOv3 network

    Args:
        batch_size (int): training batch size
        ignore_thresh (float): threshold to ignore confidence loss
        label_smooth (bool): whether to use label smoothing
        use_fine_grained_loss (bool): whether use fine grained YOLOv3 loss
                                      instead of fluid.layers.yolov3_loss
    """
    __shared__ = ['use_fine_grained_loss']

    def __init__(self,
                 batch_size=8,
                 ignore_thresh=0.7,
                 label_smooth=True,
                 use_fine_grained_loss=False):
        self._batch_size = batch_size
        self._ignore_thresh = ignore_thresh
        self._label_smooth = label_smooth
        self._use_fine_grained_loss = use_fine_grained_loss
        self._MAX_WI = 608
        self._MAX_HI = 608

    def __call__(self, outputs, gt_box, gt_label, gt_score, targets, anchors,
                 anchor_masks, mask_anchors, num_classes, prefix_name):
        if self._use_fine_grained_loss:
            return self._get_fine_grained_loss(
                outputs, targets, gt_box, self._batch_size, num_classes,
                mask_anchors, self._ignore_thresh)
        else:
            losses = []
            downsample = 32
            for i, output in enumerate(outputs):
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
                    downsample_ratio=downsample,
                    use_label_smooth=self._label_smooth,
                    name=prefix_name + "yolo_loss" + str(i))
                losses.append(fluid.layers.reduce_mean(loss))
                downsample //= 2

            return {'loss': sum(losses)}

    def _get_fine_grained_loss(self, outputs, targets, gt_box, batch_size,
                               num_classes, mask_anchors, ignore_thresh, loss_weight=2.5):
        """
        Calculate fine grained YOLOv3 loss

        Args:
            outputs ([Variables]): List of Variables, output of backbone stages
            targets ([Variables]): List of Variables, The targets for yolo
                                   loss calculatation.
            gt_box (Variable): The ground-truth boudding boxes.
            batch_size (int): The training batch size
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

        downsample = 32
        loss_xys, loss_whs, loss_locs, loss_objs, loss_clss = [], [], [], [], []
        for i, (output, target,
                anchors) in enumerate(zip(outputs, targets, mask_anchors)):
            an_num = len(anchors) // 2
            x, y, w, h, obj, cls = self._split_output(output, an_num,
                                                      num_classes)
            tx, ty, tw, th, tscale, tobj, tcls = self._split_target(target)
            loss_giou = self._IoUloss(x, y, w, h, tx, ty, tw, th, anchors, downsample)

            tscale_tobj = tscale * tobj

            loss_x = fluid.layers.sigmoid_cross_entropy_with_logits(
                x, tx) * tscale_tobj
            loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
            loss_y = fluid.layers.sigmoid_cross_entropy_with_logits(
                y, ty) * tscale_tobj
            loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])
            # NOTE: we refined loss function of (w, h) as L1Loss
            loss_w = fluid.layers.abs(w - tw) * tscale_tobj
            loss_w = fluid.layers.reduce_sum(loss_w, dim=[1, 2, 3])
            loss_h = fluid.layers.abs(h - th) * tscale_tobj
            loss_h = fluid.layers.reduce_sum(loss_h, dim=[1, 2, 3])
            loss_giou = loss_giou * tscale_tobj * loss_weight
            loss_giou = fluid.layers.reduce_sum(loss_giou, dim=[1, 2, 3])

            loss_obj_pos, loss_obj_neg = self._calc_obj_loss(
                output, obj, tobj, gt_box, self._batch_size, anchors,
                num_classes, downsample, self._ignore_thresh)

            loss_cls = fluid.layers.sigmoid_cross_entropy_with_logits(cls, tcls)
            loss_cls = fluid.layers.elementwise_mul(loss_cls, tobj, axis=0)
            loss_cls = fluid.layers.reduce_sum(loss_cls, dim=[1, 2, 3, 4])

            loss_xys.append(fluid.layers.reduce_mean(loss_x + loss_y))
            loss_whs.append(fluid.layers.reduce_mean(loss_w + loss_h))
            loss_locs.append(fluid.layers.reduce_mean(loss_giou))
            loss_objs.append(
                fluid.layers.reduce_mean(loss_obj_pos + loss_obj_neg))
            loss_clss.append(fluid.layers.reduce_mean(loss_cls))

            downsample //= 2

        return {
            "loss_xy": fluid.layers.sum(loss_xys),
            "loss_wh": fluid.layers.sum(loss_whs),
            "loss_loc": fluid.layers.sum(loss_locs),
            "loss_obj": fluid.layers.sum(loss_objs),
            "loss_cls": fluid.layers.sum(loss_clss),
        }

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
                       num_classes, downsample, ignore_thresh):
        # A prediction bbox overlap any gt_bbox over ignore_thresh, 
        # objectness loss will be ignored, process as follows:

        # 1. get pred bbox, which is same with YOLOv3 infer mode, use yolo_box here
        # NOTE: img_size is set as 1.0 to get noramlized pred bbox
        bbox, _ = fluid.layers.yolo_box(
            x=output,
            img_size=fluid.layers.ones(
                shape=[batch_size, 2], dtype="int32"),
            anchors=anchors,
            class_num=num_classes,
            conf_thresh=0.,
            downsample_ratio=downsample,
            clip_bbox=False)

        # 2. split pred bbox and gt bbox by sample, calculate IoU between pred bbox
        #    and gt bbox in each sample
        if batch_size > 1:
            preds = fluid.layers.split(bbox, batch_size, dim=0)
            gts = fluid.layers.split(gt_box, batch_size, dim=0)
        else:
            preds = [bbox]
            gts = [gt_box]
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

    def _bbox_transform(self, dcx, dcy, dw, dh, anchors, downsample_ratio, is_gt):
        batch_size = self._batch_size
        grid_x = int(self._MAX_WI / downsample_ratio)
        grid_y = int(self._MAX_HI / downsample_ratio)

        shape_fmp = fluid.layers.shape(dcx)
        shape_fmp.stop_gradient = True
        # generate the grid_w x _grid_h center of feature map
        idx_i = np.array([[i for i in range(grid_x)]])
        idx_j = np.array([[j for j in range(grid_y)]]).transpose()
        gi_np = np.repeat(idx_i, grid_y, axis=0)
        gi_np = np.expand_dims(gi_np, axis=0)
        gi_np = np.expand_dims(gi_np, axis=0)
        gi_np = np.repeat(gi_np, 3, axis=1)
        gi_np = np.repeat(gi_np, batch_size, axis=0)
        gj_np = np.repeat(idx_j, grid_x, axis=1)
        gj_np = np.expand_dims(gj_np, axis=0)
        gj_np = np.expand_dims(gj_np, axis=0)
        gj_np = np.repeat(gj_np, 3, axis=1)
        gj_np = np.repeat(gj_np, batch_size, axis=0)
        gi_max = self._crate_tensor_from_numpy(gi_np.astype(np.float32))
        gi = fluid.layers.crop(x=gi_max, shape=dcx)
        gi.stop_gradient = True
        gj_max = self._crate_tensor_from_numpy(gj_np.astype(np.float32))
        gj = fluid.layers.crop(x=gj_max, shape=dcx)
        gj.stop_gradient = True

        grid_x_act = fluid.layers.cast(shape_fmp[3], dtype="float32")
        grid_x_act.stop_gradient = True
        grid_y_act = fluid.layers.cast(shape_fmp[2], dtype="float32")
        grid_y_act.stop_gradient = True
        if is_gt:
            cx = fluid.layers.elementwise_add(dcx, gi) / grid_x_act
            cx.gradient = True
            cy = fluid.layers.elementwise_add(dcy, gi) / grid_y_act
            cy.gradient = True
        else:
            dcx_sig = fluid.layers.sigmoid(dcx)
            cx_rel = fluid.layers.elementwise_add(dcx_sig, gi)
            dcy_sig = fluid.layers.sigmoid(dcy)
            cy_rel = fluid.layers.elementwise_add(dcy_sig, gj)
            cx = cx_rel / grid_x_act
            cy = cy_rel / grid_y_act

        anchor_w_np = np.array([anchors[0], anchors[2], anchors[4]])
        anchor_w_np = np.expand_dims(anchor_w_np, axis=0)
        anchor_w_np = np.expand_dims(anchor_w_np, axis=2)
        anchor_w_np = np.expand_dims(anchor_w_np, axis=3)
        anchor_w_np = np.repeat(anchor_w_np, grid_x, axis=2)
        anchor_w_np = np.repeat(anchor_w_np, grid_y, axis=3)
        anchor_w_max = self._crate_tensor_from_numpy(anchor_w_np.astype(np.float32))
        anchor_w = fluid.layers.crop(x=anchor_w_max, shape=dcx)
        anchor_w.stop_gradient = True
        anchor_h_np = np.array([anchors[1], anchors[3], anchors[5]])
        anchor_h_np = np.expand_dims(anchor_h_np, axis=0)
        anchor_h_np = np.expand_dims(anchor_h_np, axis=2)
        anchor_h_np = np.expand_dims(anchor_h_np, axis=3)
        anchor_h_np = np.repeat(anchor_h_np, grid_x, axis=2)
        anchor_h_np = np.repeat(anchor_h_np, grid_y, axis=3)
        anchor_h_max = self._crate_tensor_from_numpy(anchor_h_np.astype(np.float32))
        anchor_h = fluid.layers.crop(x=anchor_h_max, shape=dcx)
        anchor_h.stop_gradient = True
        # e^tw e^th
        exp_dw = fluid.layers.exp(dw)
        exp_dh = fluid.layers.exp(dh)
        pw = fluid.layers.elementwise_mul(exp_dw, anchor_w) / (fluid.layers.cast(shape_fmp[3], dtype="float32") * downsample_ratio)
        ph = fluid.layers.elementwise_mul(exp_dh, anchor_h) / (fluid.layers.cast(shape_fmp[2], dtype="float32") * downsample_ratio)
        if is_gt:
            exp_dw.stop_gradient = True
            exp_dh.stop_gradient = True
            pw.stop_gradient = True
            ph.stop_gradient = True
        
        pred_ctr_x = cx
        pred_ctr_y = cy
        pred_w = pw
        pred_h = ph

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        if is_gt:
            x1.stop_gradient = True
            y1.stop_gradient = True
            x2.stop_gradient = True
            y2.stop_gradient = True

        return x1, y1, x2, y2

    def _IoUloss(self, x, y, w, h, tx, ty, tw, th,
                 anchors, downsample_ratio):
        '''
        IoU loss referenced the paper https://arxiv.org/abs/1908.03851
        Loss = 1.0 - iou * iou
        '''
        eps = 1.e-10
        x1, y1, x2, y2 = self._bbox_transform(x, y, w, h, anchors, downsample_ratio, False)
        x1g, y1g, x2g, y2g = self._bbox_transform(tx, ty, tw, th, anchors, downsample_ratio, True)

        x2 = fluid.layers.elementwise_max(x1, x2)
        y2 = fluid.layers.elementwise_max(y1, y2)

        xkis1 = fluid.layers.elementwise_max(x1, x1g)
        ykis1 = fluid.layers.elementwise_max(y1, y1g)
        xkis2 = fluid.layers.elementwise_min(x2, x2g)
        ykis2 = fluid.layers.elementwise_min(y2, y2g)

        xc1 = fluid.layers.elementwise_min(x1, x1g)
        yc1 = fluid.layers.elementwise_min(y1, y1g)
        xc2 = fluid.layers.elementwise_max(x2, x2g)
        yc2 = fluid.layers.elementwise_max(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * fluid.layers.greater_than(
            xkis2, xkis1) * fluid.layers.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + eps
        iouk = intsctk / unionk
        loss_iou = 1. - iouk * iouk

        return loss_iou

    def _crate_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_parameter(
            attr=ParamAttr(),
            shape=numpy_array.shape,
            dtype=numpy_array.dtype,
            default_initializer=NumpyArrayInitializer(numpy_array))
        paddle_array.stop_gradient = True
        return paddle_array

