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

import paddle
import paddle.nn.functional as F
import math


def xywh2xyxy(box):
    out = paddle.zeros_like(box)
    out[:, :, 0:2] = box[:, :, 0:2] - box[:, :, 2:4] / 2
    out[:, :, 2:4] = box[:, :, 0:2] + box[:, :, 2:4] / 2
    return out


def make_grid(h, w, dtype):
    yv, xv = paddle.meshgrid([paddle.arange(h), paddle.arange(w)])
    return paddle.stack((xv, yv), 2).cast(dtype=dtype)


def decode_yolo(box, anchor, downsample_ratio):
    """decode yolo box

    Args:
        box (Tensor): pred with the shape [b, h, w, na, 4]
        anchor (list): anchor with the shape [na, 2]
        downsample_ratio (int): downsample ratio, default 32
        scale (float): scale, default 1.
    
    Return:
        box (Tensor): decoded box, with the shape [b, h, w, na, 4]
    """
    h, w, na = box.shape[1:4]
    grid = make_grid(h, w, box.dtype).reshape((1, h, w, 1, 2))
    box[:, :, :, :, 0:2] = box[:, :, :, :, :2] + grid
    box[:, :, :, :, 0] = box[:, :, :, :, 0] / w
    box[:, :, :, :, 1] = box[:, :, :, :, 1] / h

    anchor = paddle.to_tensor(anchor)
    anchor = paddle.cast(anchor, box.dtype)
    anchor = anchor.reshape((1, 1, 1, na, 2))
    box[:, :, :, :, 2:4] = paddle.exp(box[:, :, :, :, 2:4]) * anchor
    box[:, :, :, :, 2] = box[:, :, :, :, 2] / (downsample_ratio * w)
    box[:, :, :, :, 3] = box[:, :, :, :, 3] / (downsample_ratio * h)
    return box


def iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]
    
    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = paddle.maximum(px1y1, gx1y1)
    x2y2 = paddle.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """calculate the iou of box1 and box2

    Args:
        box1 (Tensor): box1 with the shape (N, M, 4)
        box2 (Tensor): box1 with the shape (N, M, 4)
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape (N, M)
    """
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = paddle.maximum(px1y1, gx1y1)
    x2y2 = paddle.minimum(px2y2, gx2y2)

    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    iou = overlap / union
    if giou or ciou or diou:
        # convex w, h
        cwh = paddle.maximum(px2y2, gx2y2) - paddle.minimum(px1y1, gx1y1)
        if ciou or diou:
            # convex diagonal squared
            c2 = (cwh**2).sum(2) + eps
            # center distance
            rho2 = ((px1y1 + px2y2 - gx1y1 - gx2y2)**2).sum(2) / 4
            if diou:
                return iou - rho2 / c2
            elif ciou:
                wh1 = px2y2 - px1y1
                wh2 = gx2y2 - gx1y1
                w1, h1 = wh1[:, :, 0], wh1[:, :, 1] + eps
                w2, h2 = wh2[:, :, 0], wh2[:, :, 1] + eps
                v = (4 / math.pi**2) * paddle.pow(
                    paddle.atan(w1 / h1) - paddle.atan(w2 / h2), 2)
                alpha = v / (1 + eps - iou + v)
                alpha.stop_gradient = True
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cwh.prod(2) + eps
            return iou - (c_area - union) / c_area
    else:
        return iou
