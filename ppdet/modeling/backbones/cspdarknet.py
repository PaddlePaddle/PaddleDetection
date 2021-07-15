# Copyright (c) 2021 DBdetection Authors. All Rights Reserved.

from __future__ import absolute_import

from __future__ import division
from __future__ import print_function

import paddle.nn as nn
from ppdet.core.workspace import register, serializable
from ..architectures.common import parse_model
from ..shape_spec import ShapeSpec
from ..architectures.common import Focus, Conv, C3, SPP

__all__ = ['CSPDarknet']

@register
@serializable
class CSPDarknet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 modules=[
                     [-1, 1, Focus, [64, 3]],  # 0-P1/2
                     [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
                     [-1, 3, C3, [128]],
                     [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
                     [-1, 9, C3, [256]],
                     [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
                     [-1, 9, C3, [512]],
                     [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
                     [-1, 1, SPP, [1024, [5, 9, 13]]],
                     [-1, 3, C3, [1024, False]],  # 9
                 ],
                 depth_multiple=0.33,
                 width_multiple=0.5,
                 feature_maps=[4, 6, 9],
                 norm_type='bn'):
        super(CSPDarknet, self).__init__()
        self.model, self.save, self.ch = parse_model(
            modules, depth_multiple, width_multiple)
        self.feature_maps = feature_maps
        self._out_channels = [self.ch[i] for i in feature_maps]

    def forward(self, inputs):
        outs = []
        x = inputs['image']
        for i, m in enumerate(self.model):
            x = m(x)  # run
            if i in self.feature_maps:
                outs.append(x)
        return outs

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
