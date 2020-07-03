from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.base import to_variable
from ppdet.core.workspace import register
from ppdet.utils.data_structure import BufferDict

__all__ = ['BaseArch']


@register
class BaseArch(Layer):
    def __init__(self, *args, **kwargs):
        super(BaseArch, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, inputs, inputs_keys):
        self.gbd = BufferDict()
        self.gbd.update(self.kwargs)
        assert self.gbd[
            'mode'] is not None, "Please specify mode train or infer in config file!"
        if self.gbd['open_debug'] is None:
            self.gbd['open_debug'] = False

        self.build_inputs(inputs, inputs_keys)

        self.model_arch()

        self.gbd.debug()

        if self.gbd['mode'] == 'train':
            out = self.loss()
        elif self.gbd['mode'] == 'infer':
            out = self.infer()
        else:
            raise "Now, only support train or infer mode!"
        return out

    def build_inputs(self, inputs, inputs_keys):
        for i, k in enumerate(inputs_keys):
            v = to_variable(np.array([x[i] for x in inputs]))
            self.gbd.set(k, v)

    def model_arch(self, ):
        raise NotImplementedError("Should implement model_arch method!")

    def loss(self, ):
        raise NotImplementedError("Should implement loss method!")

    def infer(self, ):
        raise NotImplementedError("Should implement infer method!")
