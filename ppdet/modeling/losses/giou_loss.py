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
from ppdet.core.workspace import register, serializable

__all__ = ['GiouLoss']


@register
@serializable
class GiouLoss(object):

    __shared__ = ['num_classes']

    def __init__(self,
                 loss_weight=10.,
                 is_cls_agnostic=False,
                 num_classes=81,
                 max_height=608,
                 max_width=608,
                 use_xiou_loss_yolo=False):
        super(GiouLoss, self).__init__()
        self.loss_weight = loss_weight
        self.is_cls_agnostic = is_cls_agnostic
        self.num_classes = num_classes
        self._MAX_HI = max_height
        self._MAX_WI = max_width
        self._use_xiou_loss_yolo = use_xiou_loss_yolo

    def __call__(self, *args, **kwargs):
        if self._use_xiou_loss_yolo:
            _x, _y, _w, _h, _tx, _ty, _tw, _th, _anchors, _downsample_ratio, _batch_size = args[
                0], args[1], args[2], args[3], args[4], args[5], args[6], args[
                    7], args[8], args[9], args[10]
            return self._giou_loss_yolo(
                _x, _y, _w, _h, _tx, _ty, _tw, _th, _anchors, _downsample_ratio,
                _batch_size, self.loss_weight, self._MAX_HI, self._MAX_WI)
        else:
            x, y, inside_weight, outside_weight = kwargs['x'], kwargs[
                'y'], kwargs['inside_weight'], kwargs['outside_weight']
            return self.giou_loss(
                x,
                y,
                inside_weight,
                outside_weight,
                self.loss_weight,
                self.is_cls_agnostic,
                self.num_classes,
                bbox_reg_weight=[0.1, 0.1, 0.2, 0.2])

    def giou_loss(self, x, y, inside_weight, outside_weight, loss_weight,
                  is_cls_agnostic, num_classes, bbox_reg_weight):
        eps = 1.e-10
        x1, y1, x2, y2 = self.bbox_transform(x, bbox_reg_weight)
        x1g, y1g, x2g, y2g = self.bbox_transform(y, bbox_reg_weight)

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

        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + eps
        iouk = intsctk / unionk

        area_c = (xc2 - xc1) * (yc2 - yc1) + eps
        miouk = iouk - ((area_c - unionk) / area_c)

        iou_weights = 1
        if inside_weight is not None and outside_weight is not None:
            inside_weight = fluid.layers.reshape(inside_weight, shape=(-1, 4))
            outside_weight = fluid.layers.reshape(outside_weight, shape=(-1, 4))

            inside_weight = fluid.layers.reduce_mean(inside_weight, dim=1)
            outside_weight = fluid.layers.reduce_mean(outside_weight, dim=1)

            iou_weights = inside_weight * outside_weight

        class_weight = 2 if self.is_cls_agnostic else self.num_classes
        iouk = fluid.layers.reduce_mean((1 - iouk) * iou_weights) * class_weight
        miouk = fluid.layers.reduce_mean(
            (1 - miouk) * iou_weights) * class_weight

        return miouk * self.loss_weight

    def bbox_transform(self, deltas, weights):
        wx, wy, ww, wh = weights

        deltas = fluid.layers.reshape(deltas, shape=(0, -1, 4))

        dx = fluid.layers.slice(deltas, axes=[2], starts=[0], ends=[1]) * wx
        dy = fluid.layers.slice(deltas, axes=[2], starts=[1], ends=[2]) * wy
        dw = fluid.layers.slice(deltas, axes=[2], starts=[2], ends=[3]) * ww
        dh = fluid.layers.slice(deltas, axes=[2], starts=[3], ends=[4]) * wh

        dw = fluid.layers.clip(dw, -1.e10, np.log(1000. / 16))
        dh = fluid.layers.clip(dh, -1.e10, np.log(1000. / 16))

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = fluid.layers.exp(dw)
        pred_h = fluid.layers.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        x1 = fluid.layers.reshape(x1, shape=(-1, ))
        y1 = fluid.layers.reshape(y1, shape=(-1, ))
        x2 = fluid.layers.reshape(x2, shape=(-1, ))
        y2 = fluid.layers.reshape(y2, shape=(-1, ))

        return x1, y1, x2, y2

    def _giou_loss_yolo(self, x, y, w, h, tx, ty, tw, th, anchors,
                        downsample_ratio, batch_size, loss_weight, MAX_HI,
                        MAX_WI):
        eps = 1.e-10
        x1, y1, x2, y2 = self._bbox_transform(
            x, y, w, h, anchors, downsample_ratio, batch_size, False)
        x1g, y1g, x2g, y2g = self._bbox_transform(
            tx, ty, tw, th, anchors, downsample_ratio, batch_size, True)

        # A and B
        xkis1 = fluid.layers.elementwise_max(x1, x1g)
        ykis1 = fluid.layers.elementwise_max(y1, y1g)
        xkis2 = fluid.layers.elementwise_min(x2, x2g)
        ykis2 = fluid.layers.elementwise_min(y2, y2g)
        # A or B
        xc1 = fluid.layers.elementwise_min(x1, x1g)
        yc1 = fluid.layers.elementwise_min(y1, y1g)
        xc2 = fluid.layers.elementwise_max(x2, x2g)
        yc2 = fluid.layers.elementwise_max(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * fluid.layers.greater_than(
            xkis2, xkis1) * fluid.layers.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + eps
        iouk = intsctk / unionk

        #giou loss
        area_c = (xc2 - xc1) * (yc2 - yc1) + eps
        loss_giou = 1 - iouk + ((area_c - unionk) / area_c)
        loss_giou = loss_giou * self.loss_weight

        return loss_giou

    def _bbox_transform(self, dcx, dcy, dw, dh, anchors, downsample_ratio,
                        batch_size, is_gt):
        grid_x = int(self._MAX_WI / downsample_ratio)
        grid_y = int(self._MAX_HI / downsample_ratio)
        an_num = len(anchors) // 2

        shape_fmp = fluid.layers.shape(dcx)
        shape_fmp.stop_gradient = True
        # generate the grid_w x grid_h center of feature map
        idx_i = np.array([[i for i in range(grid_x)]])
        idx_j = np.array([[j for j in range(grid_y)]]).transpose()
        gi_np = np.repeat(idx_i, grid_y, axis=0)
        gi_np = np.reshape(gi_np, newshape=[1, 1, grid_y, grid_x])
        gi_np = np.tile(gi_np, reps=[batch_size, an_num, 1, 1])
        gj_np = np.repeat(idx_j, grid_x, axis=1)
        gj_np = np.reshape(gj_np, newshape=[1, 1, grid_y, grid_x])
        gj_np = np.tile(gj_np, reps=[batch_size, an_num, 1, 1])
        gi_max = self._create_tensor_from_numpy(gi_np.astype(np.float32))
        gi = fluid.layers.crop(x=gi_max, shape=dcx)
        gi.stop_gradient = True
        gj_max = self._create_tensor_from_numpy(gj_np.astype(np.float32))
        gj = fluid.layers.crop(x=gj_max, shape=dcx)
        gj.stop_gradient = True

        grid_x_act = fluid.layers.cast(shape_fmp[3], dtype="float32")
        grid_x_act.stop_gradient = True
        grid_y_act = fluid.layers.cast(shape_fmp[2], dtype="float32")
        grid_y_act.stop_gradient = True
        if is_gt:
            cx = fluid.layers.elementwise_add(dcx, gi) / grid_x_act
            cx.gradient = True
            cy = fluid.layers.elementwise_add(dcy, gj) / grid_y_act
            cy.gradient = True
        else:
            dcx_sig = fluid.layers.sigmoid(dcx)
            cx = fluid.layers.elementwise_add(dcx_sig, gi) / grid_x_act
            dcy_sig = fluid.layers.sigmoid(dcy)
            cy = fluid.layers.elementwise_add(dcy_sig, gj) / grid_y_act

        anchor_w_ = [anchors[i] for i in range(0, len(anchors)) if i % 2 == 0]
        anchor_w_np = np.array(anchor_w_)
        anchor_w_np = np.reshape(anchor_w_np, newshape=[1, an_num, 1, 1])
        anchor_w_np = np.tile(anchor_w_np, reps=[batch_size, 1, grid_y, grid_x])
        anchor_w_max = self._create_tensor_from_numpy(
            anchor_w_np.astype(np.float32))
        anchor_w = fluid.layers.crop(x=anchor_w_max, shape=dcx)
        anchor_w.stop_gradient = True
        anchor_h_ = [anchors[i] for i in range(0, len(anchors)) if i % 2 == 1]
        anchor_h_np = np.array(anchor_h_)
        anchor_h_np = np.reshape(anchor_h_np, newshape=[1, an_num, 1, 1])
        anchor_h_np = np.tile(anchor_h_np, reps=[batch_size, 1, grid_y, grid_x])
        anchor_h_max = self._create_tensor_from_numpy(
            anchor_h_np.astype(np.float32))
        anchor_h = fluid.layers.crop(x=anchor_h_max, shape=dcx)
        anchor_h.stop_gradient = True
        # e^tw e^th
        exp_dw = fluid.layers.exp(dw)
        exp_dh = fluid.layers.exp(dh)
        pw = fluid.layers.elementwise_mul(exp_dw, anchor_w) / \
            (grid_x_act * downsample_ratio)
        ph = fluid.layers.elementwise_mul(exp_dh, anchor_h) / \
            (grid_y_act * downsample_ratio)
        if is_gt:
            exp_dw.stop_gradient = True
            exp_dh.stop_gradient = True
            pw.stop_gradient = True
            ph.stop_gradient = True

        x1 = cx - 0.5 * pw
        y1 = cy - 0.5 * ph
        x2 = cx + 0.5 * pw
        y2 = cy + 0.5 * ph
        if is_gt:
            x1.stop_gradient = True
            y1.stop_gradient = True
            x2.stop_gradient = True
            y2.stop_gradient = True

        return x1, y1, x2, y2

    def _create_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_parameter(
            attr=ParamAttr(),
            shape=numpy_array.shape,
            dtype=numpy_array.dtype,
            default_initializer=NumpyArrayInitializer(numpy_array))
        paddle_array.stop_gradient = True
        return paddle_array
