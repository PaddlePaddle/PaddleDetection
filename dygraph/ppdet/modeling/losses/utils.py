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
import math


def bbox_overlap(box1, box2, eps=1e-10):
    """calculate the iou of box1 and box2
    Args:
        box1 (Tensor): box1 with the shape (..., 4)
        box2 (Tensor): box1 with the shape (..., 4)
        eps (float): epsilon to avoid divide by zero
    Return:
        iou (Tensor): iou of box1 and box2
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xkis1 = paddle.maximum(x1, x1g)
    ykis1 = paddle.maximum(y1, y1g)
    xkis2 = paddle.minimum(x2, x2g)
    ykis2 = paddle.minimum(y2, y2g)
    w_inter = (xkis2 - xkis1).clip(0)
    h_inter = (ykis2 - ykis1).clip(0)
    overlap = w_inter * h_inter

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - overlap + eps
    iou = overlap / union

    return iou, overlap, union