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

# The code is based on 
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/anchor_generator.py

import math

import paddle
import paddle.nn as nn

from ppdet.core.workspace import register

__all__ = ['AnchorGenerator', 'RetinaAnchorGenerator']


@register
class AnchorGenerator(nn.Layer):
    """
    Generate anchors according to the feature maps

    Args:
        anchor_sizes (list[float] | list[list[float]]): The anchor sizes at 
            each feature point. list[float] means all feature levels share the 
            same sizes. list[list[float]] means the anchor sizes for 
            each level. The sizes stand for the scale of input size.
        aspect_ratios (list[float] | list[list[float]]): The aspect ratios at
            each feature point. list[float] means all feature levels share the
            same ratios. list[list[float]] means the aspect ratios for
            each level.
        strides (list[float]): The strides of feature maps which generate 
            anchors
        offset (float): The offset of the coordinate of anchors, default 0.
        
    """

    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[16.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.variance = variance
        self.cell_anchors = self._calculate_anchors(len(strides))
        self.offset = offset

    def _broadcast_params(self, params, num_features):
        if not isinstance(params[0], (list, tuple)):  # list[float]
            return [params] * num_features
        if len(params) == 1:
            return list(params) * num_features
        return params

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return paddle.to_tensor(anchors, dtype='float32')

    def _calculate_anchors(self, num_features):
        sizes = self._broadcast_params(self.anchor_sizes, num_features)
        aspect_ratios = self._broadcast_params(self.aspect_ratios, num_features)
        cell_anchors = [
            self.generate_cell_anchors(s, a)
            for s, a in zip(sizes, aspect_ratios)
        ]
        [
            self.register_buffer(
                t.name, t, persistable=False) for t in cell_anchors
        ]
        return cell_anchors

    def _create_grid_offsets(self, size, stride, offset):
        grid_height, grid_width = size[0], size[1]
        shifts_x = paddle.arange(
            offset * stride, grid_width * stride, step=stride, dtype='float32')
        shifts_y = paddle.arange(
            offset * stride, grid_height * stride, step=stride, dtype='float32')
        shift_y, shift_x = paddle.meshgrid(shifts_y, shifts_x)
        shift_x = paddle.reshape(shift_x, [-1])
        shift_y = paddle.reshape(shift_y, [-1])
        return shift_x, shift_y

    def _grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides,
                                              self.cell_anchors):
            shift_x, shift_y = self._create_grid_offsets(size, stride,
                                                         self.offset)
            shifts = paddle.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
            shifts = paddle.reshape(shifts, [-1, 1, 4])
            base_anchors = paddle.reshape(base_anchors, [1, -1, 4])

            anchors.append(paddle.reshape(shifts + base_anchors, [-1, 4]))

        return anchors

    def forward(self, input):
        grid_sizes = [paddle.shape(feature_map)[-2:] for feature_map in input]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps

    @property
    def num_anchors(self):
        """
        Returns:
            int: number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                For FPN models, `num_anchors` on every feature map is the same.
        """
        return len(self.cell_anchors[0])


@register
class RetinaAnchorGenerator(AnchorGenerator):
    def __init__(self,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[8.0, 16.0, 32.0, 64.0, 128.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.0):
        anchor_sizes = []
        for s in strides:
            anchor_sizes.append([
                s * octave_base_scale * 2**(i/scales_per_octave) \
                for i in range(scales_per_octave)])
        super(RetinaAnchorGenerator, self).__init__(
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            strides=strides,
            variance=variance,
            offset=offset)
