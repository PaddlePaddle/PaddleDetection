from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from paddle import fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.base import to_variable

from ppdet.core.workspace import register
from ppdet.utils.data_structure import BufferDict

__all__ = ['BaseArch']


@register
class BaseArch(Layer):
    def __init__(self, mode='train', *args, **kwargs):
        super(BaseArch, self).__init__()
        self.mode = mode

    def forward(self, inputs, inputs_keys, mode='train'):
        raise NotImplementedError("Should implement forward method!")

    def loss(self, inputs):
        raise NotImplementedError("Should implement loss method!")

    def infer(self, inputs):
        raise NotImplementedError("Should implement infer method!")

    def build_inputs(self, inputs, inputs_keys):
        gbd = BufferDict()
        for i, k in enumerate(inputs_keys):
            v = to_variable(np.array([x[i] for x in inputs]))
            gbd.set(k, v)
        return gbd
