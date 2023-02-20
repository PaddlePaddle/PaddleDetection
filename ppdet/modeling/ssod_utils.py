#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import warnings
from collections import Counter, Mapping, Sequence
from numbers import Number
from typing import List, Optional, Tuple, Union, Dict
from functools import partial

import numpy as np
import paddle
from six.moves import map, zip



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
            gt_bboxes = data_strong['gt_bbox']
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_s
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_s
            data_strong['gt_bbox'] = gt_bboxes

    if scale_x_w != 1 or scale_y_w != 1:
        data_weak['image'] = F.interpolate(
            data_weak['image'],
            size=target_size,
            mode='bilinear',
            align_corners=False)
        if 'gt_bbox' in data_weak:
            gt_bboxes = data_weak['gt_bbox']
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_w
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_w
            data_weak['gt_bbox'] = gt_bboxes
    return data_weak, data_strong


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.reshape([N, -1, K, H, W]).transpose([0, 3, 4, 1, 2])
    tensor = tensor.reshape([N, -1, K])
    return tensor


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



def filter_invalid(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score.numel() > 0:
        # valid = score > thr
        valid = score >= thr
        # if valid.shape[0] == 1 :
        #     bbox = bbox if valid.item() else paddle.expand(paddle.to_tensor([])[:, None], (-1, 4))
        # else:
        bbox = bbox[valid]

        if label is not None:
            # if valid.shape[0] == 1 :
            #     label = label if valid.item() else paddle.to_tensor([])
            # else:
            label = label[valid]
        # bbox = bbox[valid]
        # if label is not None:
        #     label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None and bbox.shape[0] > 0:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        valid = (bw > min_size) & (bh > min_size)

        # if valid.shape[0] == 1 :
        #     bbox = bbox if valid.item() else paddle.expand(paddle.to_tensor([])[:, None], (-1, 4))
        # else:
        bbox = bbox[valid]

        if label is not None:
            # if valid.shape[0] == 1 :
            #     label = label if valid.item() else paddle.to_tensor([])
            # else:
            label = label[valid]
            
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask


def weighted_loss(loss: dict, weight, ignore_keys=[], warmup=0):
    if len(loss) == 0:
        return {}

    if isinstance(weight, Mapping):
        for k, v in weight.items():
            for name, loss_item in loss.items():
                if (k in name) and ("loss" in name):
                    loss[name] = sequence_mul(loss[name], v)
    elif isinstance(weight, Number):
        for name, loss_item in loss.items():
            if "loss" in name:
                if not is_match(name, ignore_keys):
                    loss[name] = sequence_mul(loss[name], weight)
                else:
                    loss[name] = sequence_mul(loss[name], 0.0)
    else:
        raise NotImplementedError()

    total_loss = paddle.add_n(list(loss.values()))
    loss.update({'loss': total_loss})
    return loss

def sequence_mul(obj, multiplier):
    if isinstance(obj, Sequence):
        return [o * multiplier for o in obj]
    else:
        return obj * multiplier
        
def is_match(word, word_list):
    for keyword in word_list:
        if keyword in word:
            return True
    return False