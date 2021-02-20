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

import math
import paddle


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * paddle.log(tgt_w / src_w)
    dh = wh * paddle.log(tgt_h / src_h)

    deltas = paddle.stack((dx, dy, dw, dh), axis=1)
    return deltas


def delta2bbox(deltas, boxes, weights):
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into paddle.exp()
    dw = paddle.clip(dw, max=clip_scale)
    dh = paddle.clip(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = paddle.exp(dw) * widths.unsqueeze(1)
    pred_h = paddle.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = paddle.stack(pred_boxes, axis=-1)

    return pred_boxes


def expand_bbox(bboxes, scale):
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    bboxes_exp = np.zeros(bboxes.shape, dtype=np.float32)
    bboxes_exp[:, 0] = x_c - w_half
    bboxes_exp[:, 2] = x_c + w_half
    bboxes_exp[:, 1] = y_c - h_half
    bboxes_exp[:, 3] = y_c + h_half

    return bboxes_exp


def clip_bbox(boxes, im_shape):
    h, w = im_shape[0], im_shape[1]
    x1 = boxes[:, 0].clip(0, w)
    y1 = boxes[:, 1].clip(0, h)
    x2 = boxes[:, 2].clip(0, w)
    y2 = boxes[:, 3].clip(0, h)
    return paddle.stack([x1, y1, x2, y2], axis=1)


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = paddle.logical_and(w > min_size, w > min_size)
    if return_mask:
        return mask
    keep = paddle.nonzero(mask).flatten()
    return keep


def bbox_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def bbox_overlaps(boxes1, boxes2):
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)

    xy_max = paddle.minimum(
        paddle.unsqueeze(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
    xy_min = paddle.maximum(
        paddle.unsqueeze(boxes1, 1)[:, :, :2], boxes2[:, :2])
    width_height = xy_max - xy_min
    width_height = width_height.clip(min=0)
    inter = width_height.prod(axis=2)

    overlaps = paddle.where(inter > 0, inter /
                            (paddle.unsqueeze(area1, 1) + area2 - inter),
                            paddle.zeros_like(inter))
    return overlaps
