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
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.modeling.ops import MultiClassNMS, MultiClassSoftNMS, MatrixNMS
from ppdet.modeling.losses.yolo_loss import YOLOv3Loss
from ppdet.core.workspace import register
from ppdet.modeling.ops import DropBlock
from .iou_aware import get_iou_aware_score
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from ppdet.utils.check import check_version

__all__ = ['YOLOv5Head']


@register
class YOLOv5Head(object):

    __inject__ = ['nms', 'yolo_loss']
    __shared__ = ['num_classes', 'weight_prefix_name']

    def __init__(self,
                 anchors=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                          [72, 146], [142, 110], [192, 243], [459, 401]],
                 anchor_masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 num_classes=80,
                 yolo_loss="YOLOv5Loss",
                 weight_prefix_name='',
                 stride=[8, 16, 32],
                 start=24,
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__):

        self.anchors = self._parse_anchors(anchors, anchor_masks)
        self.anchor_masks = anchor_masks
        self.num_classes = num_classes
        self.yolo_loss = yolo_loss
        self.prefix = weight_prefix_name
        self.stride = stride
        self.start = start
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)

    def _create_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_global_var(
            shape=numpy_array.shape, value=0., dtype=numpy_array.dtype)
        fluid.layers.assign(numpy_array, paddle_array)
        return paddle_array

    def _parse_anchors(self, anchors, anchor_masks):
        output = []
        for anchor_mask in anchor_masks:
            output.append([anchors[i] for i in anchor_mask])
        return output

    def _get_outputs(self, inputs):
        outputs = []
        for i, x in enumerate(inputs):
            c_out = len(self.anchor_masks[i]) * (self.num_classes + 5)
            output = fluid.layers.conv2d(
                x,
                c_out,
                1,
                1,
                0,
                act=None,
                param_attr=ParamAttr(
                    name=self.prefix + '.{}.m.{}.weight'.format(self.start, i)),
                bias_attr=ParamAttr(
                    regularizer=L2Decay(0.0),
                    name=self.prefix + '.{}.m.{}.bias'.format(self.start, i)))
            outputs.append(output)

        return outputs

    def get_loss(self, inputs, gt_box, gt_label, gt_score, targets):
        outputs = self._get_outputs(inputs)
        return self.yolo_loss(outputs, targets, gt_box, gt_label, self.anchors,
                              self.num_classes, self.stride)

    def get_prediction(self,
                       inputs,
                       im_size,
                       im_scale,
                       im_pad,
                       exclude_nms=False):
        outputs = self._get_outputs(inputs)
        boxes, scores = [], []
        for i, output in enumerate(outputs):
            output = fluid.layers.sigmoid(output)
            output_shape = fluid.layers.shape(output)
            bs, c, h, w = output_shape[0], output_shape[1], output_shape[
                2], output_shape[3]
            na = len(self.anchor_masks[i])
            no = self.num_classes + 5
            output = fluid.layers.reshape(output, [bs, na, no, h, w])
            output = fluid.layers.transpose(output, perm=[0, 1, 3, 4, 2])
            grid = self._make_grid(w, h)
            # decode
            xy = (output[:, :, :, :, 0:2] * 2 - 0.5 + grid) * self.stride[i]
            anchor = np.array(self.anchors[i]).reshape(
                (1, 3, 1, 1, 2)).astype(np.float32)
            anchor = self._create_tensor_from_numpy(anchor)
            wh = (output[:, :, :, :, 2:4] * 2)**2 * anchor
            box = self._xywh2xxyy(xy, wh)
            box = fluid.layers.reshape(box, (bs, -1, 4))
            box = self._scale_box(box, im_scale, im_pad)
            box = self._clip_box(box, im_size)
            boxes.append(box)
            # calculate prop
            objectness = output[:, :, :, :, 4:5]
            cls_p = output[:, :, :, :, 5:] * objectness
            score = fluid.layers.reshape(cls_p, (bs, -1, self.num_classes))
            scores.append(score)

        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=1)
        if exclude_nms:
            return {'bbox': yolo_scores}
        if type(self.nms) is not MultiClassSoftNMS:
            yolo_scores = fluid.layers.transpose(yolo_scores, perm=[0, 2, 1])
        pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        return {'bbox': pred}

    def _make_grid(self, nx, ny):
        start = self._create_tensor_from_numpy(np.array([0], dtype=np.int32))
        step = self._create_tensor_from_numpy(np.array([1], dtype=np.int32))
        yv, xv = fluid.layers.meshgrid([
            fluid.layers.arange(start, ny, step), fluid.layers.arange(start, nx,
                                                                      step)
        ])
        grid = fluid.layers.stack([xv, yv], axis=2)
        return fluid.layers.reshape(grid, (1, 1, ny, nx, 2))

    def _xywh2xxyy(self, xy, wh):
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        return fluid.layers.concat([x1y1, x2y2], axis=-1)

    def _scale_box(self, box, im_scale, im_pad):
        x1 = (box[:, :, 0:1] - im_pad[:, 1:2]) * im_scale[:, 1:2]
        y1 = (box[:, :, 1:2] - im_pad[:, 0:1]) * im_scale[:, 0:1]
        x2 = (box[:, :, 2:3] - im_pad[:, 1:2]) * im_scale[:, 1:2] - 1
        y2 = (box[:, :, 3:4] - im_pad[:, 0:1]) * im_scale[:, 0:1] - 1
        return fluid.layers.concat([x1, y1, x2, y2], axis=-1)

    def _clip_box(self, box, im_size):
        bs = fluid.layers.shape(box)[0]
        outputs = []
        for i in range(1):
            s = fluid.layers.cast(im_size[i], dtype=np.float32)
            x1 = fluid.layers.clamp(box[i, :, 0:1], min=0., max=s[1])
            y1 = fluid.layers.clamp(box[i, :, 1:2], min=0., max=s[0])
            x2 = fluid.layers.clamp(box[i, :, 2:3], min=0., max=s[1])
            y2 = fluid.layers.clamp(box[i, :, 3:4], min=0., max=s[0])
            output = fluid.layers.concat([x1, y1, x2, y2], axis=-1)
            outputs.append(output)
        return fluid.layers.stack(outputs, axis=0)
