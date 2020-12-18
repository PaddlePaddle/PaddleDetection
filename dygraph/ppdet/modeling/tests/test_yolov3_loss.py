#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import unittest
import numpy as np
from scipy.special import logit
from scipy.special import expit

import paddle
from paddle import fluid
from paddle.fluid import core
# add python path of PadleDetection to sys.path
import os
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.modeling.loss import YOLOv3Loss
from ppdet.data.transform.op_helper import jaccard_overlap
import random
import numpy as np


def _split_ioup(output, an_num, num_classes):
    """
    Split output feature map to output, predicted iou
    along channel dimension
    """
    ioup = fluid.layers.slice(output, axes=[1], starts=[0], ends=[an_num])
    ioup = fluid.layers.sigmoid(ioup)
    oriout = fluid.layers.slice(
        output, axes=[1], starts=[an_num], ends=[an_num * (num_classes + 6)])
    return (ioup, oriout)


def _split_output(output, an_num, num_classes):
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


def _split_target(target):
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
    tcls = fluid.layers.transpose(target[:, :, 6:, :, :], perm=[0, 1, 3, 4, 2])
    tcls.stop_gradient = True
    return (tx, ty, tw, th, tscale, tobj, tcls)


def _calc_obj_loss(output, obj, tobj, gt_box, batch_size, anchors, num_classes,
                   downsample, ignore_thresh, scale_x_y):
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


def fine_grained_loss(output,
                      target,
                      gt_box,
                      batch_size,
                      num_classes,
                      anchors,
                      ignore_thresh,
                      downsample,
                      scale_x_y=1.,
                      eps=1e-10):
    an_num = len(anchors) // 2
    x, y, w, h, obj, cls = _split_output(output, an_num, num_classes)
    tx, ty, tw, th, tscale, tobj, tcls = _split_target(target)

    tscale_tobj = tscale * tobj

    scale_x_y = scale_x_y

    if (abs(scale_x_y - 1.0) < eps):
        loss_x = fluid.layers.sigmoid_cross_entropy_with_logits(
            x, tx) * tscale_tobj
        loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
        loss_y = fluid.layers.sigmoid_cross_entropy_with_logits(
            y, ty) * tscale_tobj
        loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])
    else:
        dx = scale_x_y * fluid.layers.sigmoid(x) - 0.5 * (scale_x_y - 1.0)
        dy = scale_x_y * fluid.layers.sigmoid(y) - 0.5 * (scale_x_y - 1.0)
        loss_x = fluid.layers.abs(dx - tx) * tscale_tobj
        loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
        loss_y = fluid.layers.abs(dy - ty) * tscale_tobj
        loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])

    # NOTE: we refined loss function of (w, h) as L1Loss
    loss_w = fluid.layers.abs(w - tw) * tscale_tobj
    loss_w = fluid.layers.reduce_sum(loss_w, dim=[1, 2, 3])
    loss_h = fluid.layers.abs(h - th) * tscale_tobj
    loss_h = fluid.layers.reduce_sum(loss_h, dim=[1, 2, 3])

    loss_obj_pos, loss_obj_neg = _calc_obj_loss(
        output, obj, tobj, gt_box, batch_size, anchors, num_classes, downsample,
        ignore_thresh, scale_x_y)

    loss_cls = fluid.layers.sigmoid_cross_entropy_with_logits(cls, tcls)
    loss_cls = fluid.layers.elementwise_mul(loss_cls, tobj, axis=0)
    loss_cls = fluid.layers.reduce_sum(loss_cls, dim=[1, 2, 3, 4])

    loss_xys = fluid.layers.reduce_mean(loss_x + loss_y)
    loss_whs = fluid.layers.reduce_mean(loss_w + loss_h)
    loss_objs = fluid.layers.reduce_mean(loss_obj_pos + loss_obj_neg)
    loss_clss = fluid.layers.reduce_mean(loss_cls)

    losses_all = {
        "loss_xy": fluid.layers.sum(loss_xys),
        "loss_wh": fluid.layers.sum(loss_whs),
        "loss_loc": fluid.layers.sum(loss_xys) + fluid.layers.sum(loss_whs),
        "loss_obj": fluid.layers.sum(loss_objs),
        "loss_cls": fluid.layers.sum(loss_clss),
    }
    return losses_all, x, y, tx, ty


def gt2yolotarget(gt_bbox, gt_class, gt_score, anchors, mask, num_classes, size,
                  stride):
    grid_h, grid_w = size
    h, w = grid_h * stride, grid_w * stride
    an_hw = np.array(anchors) / np.array([[w, h]])
    target = np.zeros(
        (len(mask), 6 + num_classes, grid_h, grid_w), dtype=np.float32)
    for b in range(gt_bbox.shape[0]):
        gx, gy, gw, gh = gt_bbox[b, :]
        cls = gt_class[b]
        score = gt_score[b]
        if gw <= 0. or gh <= 0. or score <= 0.:
            continue

        # find best match anchor index
        best_iou = 0.
        best_idx = -1
        for an_idx in range(an_hw.shape[0]):
            iou = jaccard_overlap([0., 0., gw, gh],
                                  [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
            if iou > best_iou:
                best_iou = iou
                best_idx = an_idx

        gi = int(gx * grid_w)
        gj = int(gy * grid_h)

        # gtbox should be regresed in this layes if best match 
        # anchor index in anchor mask of this layer
        if best_idx in mask:
            best_n = mask.index(best_idx)

            # x, y, w, h, scale
            target[best_n, 0, gj, gi] = gx * grid_w - gi
            target[best_n, 1, gj, gi] = gy * grid_h - gj
            target[best_n, 2, gj, gi] = np.log(gw * w / anchors[best_idx][0])
            target[best_n, 3, gj, gi] = np.log(gh * h / anchors[best_idx][1])
            target[best_n, 4, gj, gi] = 2.0 - gw * gh

            # objectness record gt_score
            # if target[best_n, 5, gj, gi] > 0:
            #     print('find 1 duplicate')
            target[best_n, 5, gj, gi] = score

            # classification
            target[best_n, 6 + cls, gj, gi] = 1.

    return target


class TestYolov3LossOp(unittest.TestCase):
    def setUp(self):
        self.initTestCase()
        x = np.random.uniform(0, 1, self.x_shape).astype('float64')
        gtbox = np.random.random(size=self.gtbox_shape).astype('float64')
        gtlabel = np.random.randint(0, self.class_num, self.gtbox_shape[:2])
        gtmask = np.random.randint(0, 2, self.gtbox_shape[:2])
        gtbox = gtbox * gtmask[:, :, np.newaxis]
        gtlabel = gtlabel * gtmask

        gtscore = np.ones(self.gtbox_shape[:2]).astype('float64')
        if self.gtscore:
            gtscore = np.random.random(self.gtbox_shape[:2]).astype('float64')

        target = []
        for box, label, score in zip(gtbox, gtlabel, gtscore):
            target.append(
                gt2yolotarget(box, label, score, self.anchors, self.anchor_mask,
                              self.class_num, (self.h, self.w
                                               ), self.downsample_ratio))

        self.target = np.array(target).astype('float64')

        self.mask_anchors = []
        for i in self.anchor_mask:
            self.mask_anchors.extend(self.anchors[i])
        self.x = x
        self.gtbox = gtbox
        self.gtlabel = gtlabel
        self.gtscore = gtscore

    def initTestCase(self):
        self.b = 8
        self.h = 19
        self.w = 19
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                        [59, 119], [116, 90], [156, 198], [373, 326]]
        self.anchor_mask = [6, 7, 8]
        self.na = len(self.anchor_mask)
        self.class_num = 80
        self.ignore_thresh = 0.7
        self.downsample_ratio = 32
        self.x_shape = (self.b, len(self.anchor_mask) * (5 + self.class_num),
                        self.h, self.w)
        self.gtbox_shape = (self.b, 40, 4)
        self.gtscore = True
        self.use_label_smooth = False
        self.scale_x_y = 1.

    def test_loss(self):
        x, gtbox, gtlabel, gtscore, target = self.x, self.gtbox, self.gtlabel, self.gtscore, self.target
        yolo_loss = YOLOv3Loss(
            ignore_thresh=self.ignore_thresh,
            label_smooth=self.use_label_smooth,
            num_classes=self.class_num,
            downsample=self.downsample_ratio,
            scale_x_y=self.scale_x_y)
        x = paddle.to_tensor(x.astype(np.float32))
        gtbox = paddle.to_tensor(gtbox.astype(np.float32))
        gtlabel = paddle.to_tensor(gtlabel.astype(np.float32))
        gtscore = paddle.to_tensor(gtscore.astype(np.float32))
        t = paddle.to_tensor(target.astype(np.float32))
        anchor = [self.anchors[i] for i in self.anchor_mask]
        (yolo_loss1, px, py, tx, ty) = fine_grained_loss(
            output=x,
            target=t,
            gt_box=gtbox,
            batch_size=self.b,
            num_classes=self.class_num,
            anchors=self.mask_anchors,
            ignore_thresh=self.ignore_thresh,
            downsample=self.downsample_ratio,
            scale_x_y=self.scale_x_y)
        yolo_loss2 = yolo_loss.yolov3_loss(
            x, t, gtbox, anchor, self.downsample_ratio, self.scale_x_y)
        for k in yolo_loss2:
            self.assertAlmostEqual(
                yolo_loss1[k].numpy()[0],
                yolo_loss2[k].numpy()[0],
                delta=1e-2,
                msg=k)


class TestYolov3LossNoGTScore(TestYolov3LossOp):
    def initTestCase(self):
        self.b = 1
        self.h = 76
        self.w = 76
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                        [59, 119], [116, 90], [156, 198], [373, 326]]
        self.anchor_mask = [0, 1, 2]
        self.na = len(self.anchor_mask)
        self.class_num = 80
        self.ignore_thresh = 0.7
        self.downsample_ratio = 8
        self.x_shape = (self.b, len(self.anchor_mask) * (5 + self.class_num),
                        self.h, self.w)
        self.gtbox_shape = (self.b, 40, 4)
        self.gtscore = False
        self.use_label_smooth = False
        self.scale_x_y = 1.


class TestYolov3LossWithScaleXY(TestYolov3LossOp):
    def initTestCase(self):
        self.b = 5
        self.h = 38
        self.w = 38
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                        [59, 119], [116, 90], [156, 198], [373, 326]]
        self.anchor_mask = [3, 4, 5]
        self.na = len(self.anchor_mask)
        self.class_num = 80
        self.ignore_thresh = 0.7
        self.downsample_ratio = 16
        self.x_shape = (self.b, len(self.anchor_mask) * (5 + self.class_num),
                        self.h, self.w)
        self.gtbox_shape = (self.b, 40, 4)
        self.gtscore = True
        self.use_label_smooth = False
        self.scale_x_y = 1.2


if __name__ == "__main__":
    unittest.main()
