from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from ppdet.core.workspace import register

__all__ = ['BaseArch']


@register
class BaseArch(nn.Layer):
    def __init__(self, data_format='NCHW'):
        super(BaseArch, self).__init__()
        self.data_format = data_format
        self.inputs = {}
        self.fuse_norm = False

    def load_meanstd(self, cfg_transform):
        self.scale = 1.
        self.mean = paddle.to_tensor([0.485, 0.456, 0.406]).reshape(
            (1, 3, 1, 1))
        self.std = paddle.to_tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        for item in cfg_transform:
            if 'NormalizeImage' in item:
                self.mean = paddle.to_tensor(item['NormalizeImage'][
                    'mean']).reshape((1, 3, 1, 1))
                self.std = paddle.to_tensor(item['NormalizeImage'][
                    'std']).reshape((1, 3, 1, 1))
                if item['NormalizeImage'].get('is_scale', True):
                    self.scale = 1. / 255.
                break
        if self.data_format == 'NHWC':
            self.mean = self.mean.reshape(1, 1, 1, 3)
            self.std = self.std.reshape(1, 1, 1, 3)

    def forward(self, inputs):
        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = paddle.transpose(image, [0, 2, 3, 1])

        if self.fuse_norm:
            image = inputs['image']
            self.inputs['image'] = (image * self.scale - self.mean) / self.std
            self.inputs['im_shape'] = inputs['im_shape']
            self.inputs['scale_factor'] = inputs['scale_factor']
        else:
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
