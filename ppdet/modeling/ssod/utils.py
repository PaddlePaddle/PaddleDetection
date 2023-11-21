#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F


def align_weak_strong_shape(data_weak, data_strong):
    max_shape_x = max(data_strong['image'].shape[2],
                      data_weak['image'].shape[2])
    max_shape_y = max(data_strong['image'].shape[3],
                      data_weak['image'].shape[3])

    scale_x_s = max_shape_x / data_strong['image'].shape[2]
    scale_y_s = max_shape_y / data_strong['image'].shape[3]
    scale_x_w = max_shape_x / data_weak['image'].shape[2]
    scale_y_w = max_shape_y / data_weak['image'].shape[3]
    target_size = [max_shape_x, max_shape_y]

    if scale_x_s != 1 or scale_y_s != 1:
        data_strong['image'] = F.interpolate(
            data_strong['image'],
            size=target_size,
            mode='bilinear',
            align_corners=False)
        if 'gt_bbox' in data_strong:
            gt_bboxes = data_strong['gt_bbox'].numpy()
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_s
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_s
            data_strong['gt_bbox'] = paddle.to_tensor(gt_bboxes)

    if scale_x_w != 1 or scale_y_w != 1:
        data_weak['image'] = F.interpolate(
            data_weak['image'],
            size=target_size,
            mode='bilinear',
            align_corners=False)
        if 'gt_bbox' in data_weak:
            gt_bboxes = data_weak['gt_bbox'].numpy()
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_w
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_w
            data_weak['gt_bbox'] = paddle.to_tensor(gt_bboxes)
    return data_weak, data_strong


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    pt = pred_sigmoid
    zerolabel = paddle.zeros_like(pt)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos],
        reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss


def filter_invalid(bbox, label=None, score=None, thr=0.0, min_size=0):
    if score.numel() > 0:
        soft_score = score.max(-1)
        valid = soft_score >= thr
        bbox = bbox[valid]

        if label is not None:
            label = label[valid]
        score = score[valid]
    if min_size is not None and bbox.shape[0] > 0:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]

        if label is not None:
            label = label[valid]
            score = score[valid]

    return bbox, label, score
