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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NumpyArrayInitializer

from paddle import fluid
from ppdet.core.workspace import register, serializable

__all__ = ['IouLoss']


@register
@serializable
class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 max_height=608,
                 max_width=608,
                 ciou_term=False,
                 loss_square=True):
        self._loss_weight = loss_weight
        self._MAX_HI = max_height
        self._MAX_WI = max_width
        self.ciou_term = ciou_term
        self.loss_square = loss_square

    def __call__(self,
                 x,
                 y,
                 w,
                 h,
                 tx,
                 ty,
                 tw,
                 th,
                 anchors,
                 downsample_ratio,
                 batch_size,
                 scale_x_y=1.,
                 ioup=None,
                 eps=1.e-10):
        '''
        Args:
            x  | y | w | h  ([Variables]): the output of yolov3 for encoded x|y|w|h
            tx |ty |tw |th  ([Variables]): the target of yolov3 for encoded x|y|w|h
            anchors ([float]): list of anchors for current output layer
            downsample_ratio (float): the downsample ratio for current output layer
            batch_size (int): training batch size
            eps (float): the decimal to prevent the denominator eqaul zero
        '''
        pred = self._bbox_transform(x, y, w, h, anchors, downsample_ratio,
                                    batch_size, False, scale_x_y, eps)
        gt = self._bbox_transform(tx, ty, tw, th, anchors, downsample_ratio,
                                  batch_size, True, scale_x_y, eps)
        iouk = self._iou(pred, gt, ioup, eps)
        if self.loss_square:
            loss_iou = 1. - iouk * iouk
        else:
            loss_iou = 1. - iouk
        loss_iou = loss_iou * self._loss_weight

        return loss_iou

    def _iou(self, pred, gt, ioup=None, eps=1.e-10):
        x1, y1, x2, y2 = pred
        x1g, y1g, x2g, y2g = gt
        x2 = fluid.layers.elementwise_max(x1, x2)
        y2 = fluid.layers.elementwise_max(y1, y2)

        xkis1 = fluid.layers.elementwise_max(x1, x1g)
        ykis1 = fluid.layers.elementwise_max(y1, y1g)
        xkis2 = fluid.layers.elementwise_min(x2, x2g)
        ykis2 = fluid.layers.elementwise_min(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * fluid.layers.greater_than(
            xkis2, xkis1) * fluid.layers.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + eps
        iouk = intsctk / unionk
        if self.ciou_term:
            ciou = self.get_ciou_term(pred, gt, iouk, eps)
            iouk = iouk - ciou
        return iouk

    def get_ciou_term(self, pred, gt, iouk, eps):
        x1, y1, x2, y2 = pred
        x1g, y1g, x2g, y2g = gt

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) + fluid.layers.cast((x2 - x1) == 0, 'float32')
        h = (y2 - y1) + fluid.layers.cast((y2 - y1) == 0, 'float32')

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g
        hg = y2g - y1g

        # A or B
        xc1 = fluid.layers.elementwise_min(x1, x1g)
        yc1 = fluid.layers.elementwise_min(y1, y1g)
        xc2 = fluid.layers.elementwise_max(x2, x2g)
        yc2 = fluid.layers.elementwise_max(y2, y2g)

        # DIOU term
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        diou_term = (dist_intersection + eps) / (dist_union + eps)
        # CIOU term
        ciou_term = 0
        ar_gt = wg / hg
        ar_pred = w / h
        arctan = fluid.layers.atan(ar_gt) - fluid.layers.atan(ar_pred)
        ar_loss = 4. / np.pi / np.pi * arctan * arctan
        alpha = ar_loss / (1 - iouk + ar_loss + eps)
        alpha.stop_gradient = True
        ciou_term = alpha * ar_loss
        return diou_term + ciou_term

    def _bbox_transform(self, dcx, dcy, dw, dh, anchors, downsample_ratio,
                        batch_size, is_gt, scale_x_y, eps):
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
            dcy_sig = fluid.layers.sigmoid(dcy)
            if (abs(scale_x_y - 1.0) > eps):
                dcx_sig = scale_x_y * dcx_sig - 0.5 * (scale_x_y - 1)
                dcy_sig = scale_x_y * dcy_sig - 0.5 * (scale_x_y - 1)
            cx = fluid.layers.elementwise_add(dcx_sig, gi) / grid_x_act
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
        paddle_array.persistable = False
        return paddle_array
