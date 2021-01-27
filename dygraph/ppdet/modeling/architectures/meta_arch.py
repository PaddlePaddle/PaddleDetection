from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
from ppdet.core.workspace import register

__all__ = ['BaseArch']


@register
class BaseArch(nn.Layer):
    def __init__(self):
        super(BaseArch, self).__init__()

    def forward(self, inputs):
        self.inputs = inputs
        self.model_arch()

        if self.training:
            out = self.get_loss()
        else:
            out = self.get_pred()
        return out

    def build_inputs(self, data, input_def):
        inputs = {}
        for i, k in enumerate(input_def):
            inputs[k] = data[i]
        return inputs

    def model_arch(self, ):
        pass

    def get_loss(self, ):
        raise NotImplementedError("Should implement get_loss method!")

    def get_pred(self, ):
        raise NotImplementedError("Should implement get_pred method!")
