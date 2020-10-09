# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid
from ppdet.core.workspace import register
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

__all__ = ['YOLOv5Loss']


@register
class YOLOv5Loss(object):
    def __init__(self,
                 train_batch_size=8,
                 batch_size=8,
                 giou_ratio=1.0,
                 balance=[4., 1., 0.4],
                 loss_weights=[0.05, 0.5, 1.0]):
        super(YOLOv5Loss, self).__init__()
        self.train_batch_size = train_batch_size
        self.balance = balance
        self.loss_weights = loss_weights
        self.giou_ratio = giou_ratio

    def _create_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_global_var(
            shape=numpy_array.shape, value=0., dtype=numpy_array.dtype)
        fluid.layers.assign(numpy_array, paddle_array)
        return paddle_array

    def _bbox_iou(self,
                  box1,
                  box2,
                  x1y1x2y2=True,
                  GIoU=False,
                  DIoU=False,
                  CIoU=False,
                  eps=1e-9):
        if x1y1x2y2:
            x1, y1, x2, y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            x1g, y1g, x2g, y2g = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        else:
            x1, x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            y1, y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            x1g, x2g = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            y1g, y2g = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        # intersection
        xi1 = fluid.layers.elementwise_max(x1, x1g)
        yi1 = fluid.layers.elementwise_max(y1, y1g)
        xi2 = fluid.layers.elementwise_min(x2, x2g)
        yi2 = fluid.layers.elementwise_min(y2, y2g)

        w1, h1 = x2 - x1, y2 - y1 + eps
        w2, h2 = x2g - x1g, y2g - y1g + eps
        inter = (xi2 - xi1) * (yi2 - yi1)
        inter = inter * fluid.layers.greater_than(
            xi2, xi1) * fluid.layers.greater_than(yi2, yi1)
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        if GIoU or DIoU or CIoU:
            xc1 = fluid.layers.elementwise_min(x1, x1g)
            yc1 = fluid.layers.elementwise_min(y1, y1g)
            xc2 = fluid.layers.elementwise_max(x2, x2g)
            yc2 = fluid.layers.elementwise_max(y2, y2g)

            cw = xc2 - xc1
            ch = yc2 - yc1
            if CIoU or DIoU:
                c2 = cw**2 + ch**2 + eps
                rho2 = ((x1g + x2g - x1 - x2)**2 + (y1g + y2g - y1 - y2)**2) / 4
                if DIoU:
                    return iou - rho2 / c2
                elif CIoU:
                    arctan = fluid.layers.atan(w2 / h2) - fluid.layers.atan(w1 /
                                                                            h1)
                    ar_loss = (2 * arctan / np.pi)**2
                    alpha = ar_loss / (1 - iou + ar_loss + eps)
                    alpha.stop_gradient = True
                    return iou - rho2 / c2 - alpha * ar_loss
            else:
                c_area = cw * ch + eps
                return iou - (c_area - union) / c_area
        else:
            return iou

    def __call__(self, outputs, targets, gt_box, gt_label, anchors, num_classes,
                 strides):
        loss_boxes, loss_objs, loss_clss = [], [], []
        no = 5 + num_classes
        for i, (output, target) in enumerate(zip(outputs, targets)):
            anchor = anchors[i]
            na = len(anchor)
            output_shape = fluid.layers.shape(output)
            bs, c, h, w = output_shape[0], output_shape[1], output_shape[
                2], output_shape[3]
            output = fluid.layers.reshape(output, [bs, na, no, h, w])
            output = fluid.layers.transpose(output, perm=[0, 1, 3, 4, 2])
            anchor = (np.array(anchor) / strides[i]).reshape(
                (1, 3, 1, 1, 2)).astype(np.float32)
            anchor = self._create_tensor_from_numpy(anchor)
            xy = fluid.layers.sigmoid(output[:, :, :, :, 0:2]) * 2 - 0.5
            wh = (fluid.layers.sigmoid(output[:, :, :, :, 2:4]) * 2)**2 * anchor
            target = fluid.layers.reshape(target, [bs, na, no, h, w])
            target = fluid.layers.transpose(target, perm=[0, 1, 3, 4, 2])
            pbox = fluid.layers.concat([xy, wh], axis=-1)
            tbox = target[:, :, :, :, 0:4]
            pbox = fluid.layers.reshape(pbox, [-1, 4])
            tbox = fluid.layers.reshape(tbox, [-1, 4])
            mask = fluid.layers.reshape(target[:, :, :, :, 4], [-1])
            nm = fluid.layers.reduce_sum(mask) + 0.00000001
            giou = self._bbox_iou(pbox, tbox, x1y1x2y2=False, CIoU=True)
            # fluid.layers.Print(fluid.layers.reduce_min(giou))
            # fluid.layers.Print(fluid.layers.reduce_max(giou))
            loss_box = fluid.layers.reduce_sum(
                (1 - giou) *
                mask) * self.loss_weights[0] * self.train_batch_size
            # fluid.layers.Print(loss_box)
            loss_boxes.append(loss_box / nm)
            pcls = output[:, :, :, :, 5:]
            tcls = target[:, :, :, :, 5:]
            pcls = fluid.layers.reshape(pcls, [-1, num_classes])
            tcls = fluid.layers.reshape(tcls, [-1, num_classes])
            loss_cls = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(pcls, tcls),
                dim=-1)
            loss_cls = fluid.layers.reduce_sum(
                loss_cls * mask) * self.loss_weights[1] * self.train_batch_size
            loss_clss.append(loss_cls / nm)
            pobj = output[:, :, :, :, 4]
            pobj = fluid.layers.reshape(pobj, [-1])
            tobj = (1 - self.giou_ratio) * mask + self.giou_ratio * giou
            tobj = fluid.layers.clamp(tobj, min=0.)
            tobj.stop_gradient = True
            loss_obj = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    pobj, tobj)) * self.balance[i] * self.train_batch_size
            loss_objs.append(loss_obj * self.loss_weights[2])

        losses_all = {
            'loss_box': fluid.layers.sum(loss_boxes),
            'loss_obj': fluid.layers.sum(loss_objs),
            'loss_cls': fluid.layers.sum(loss_clss)
        }
        return losses_all
