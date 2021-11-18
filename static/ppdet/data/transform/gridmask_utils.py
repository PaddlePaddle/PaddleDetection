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

# The code is based on:
# https://github.com/dvlab-research/GridMask/blob/master/detection_grid/maskrcnn_benchmark/data/transforms/grid.py

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from PIL import Image


class GridMask(object):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 prob=0.7,
                 upper_iter=360000):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.st_prob = prob
        self.upper_iter = upper_iter

    def __call__(self, x, curr_iter):
        self.prob = self.st_prob * min(1, 1.0 * curr_iter / self.upper_iter)
        if np.random.rand() > self.prob:
            return x
        # image should be C, H, W format
        h, w, _ = x.shape
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2
                    + w].astype(np.float32)

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask, axis=-1)
        if self.offset:
            offset = (2 * (np.random.rand(h, w) - 0.5)).astype(np.float32)
            x = (x * mask + offset * (1 - mask)).astype(x.dtype)
        else:
            x = (x * mask).astype(x.dtype)

        return x
