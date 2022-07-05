# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is the paddle implementation of MobileOne block, see: https://arxiv.org/pdf/2206.04040.pdf. 
Some codes are based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
Ths copyright of microsoft/Swin-Transformer is as follows:
MIT License [see LICENSE for details]
"""

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant

from ppdet.modeling.ops import get_act_fn
from ppdet.modeling.layers import ConvNormLayer


class MobileOneBlock(nn.Layer):
    def __init__(
            self,
            ch_in,
            ch_out,
            stride,
            kernel_size,
            conv_num=1,
            norm_type='bn',
            norm_decay=0.,
            norm_groups=32,
            bias_on=False,
            lr_scale=1.,
            freeze_norm=False,
            initializer=Normal(
                mean=0., std=0.01),
            skip_quant=False,
            act='relu', ):
        super(MobileOneBlock, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.k = conv_num

        self.depth_conv = nn.LayerList()
        self.point_conv = nn.LayerList()
        for _ in range(self.k):
            self.depth_conv.append(
                ConvNormLayer(
                    ch_in,
                    ch_in,
                    kernel_size,
                    stride=stride,
                    groups=ch_in,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    norm_groups=norm_groups,
                    bias_on=bias_on,
                    lr_scale=lr_scale,
                    freeze_norm=freeze_norm,
                    initializer=initializer,
                    skip_quant=skip_quant))
            self.point_conv.append(
                ConvNormLayer(
                    ch_in,
                    ch_out,
                    1,
                    stride=1,
                    groups=1,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    norm_groups=norm_groups,
                    bias_on=bias_on,
                    lr_scale=lr_scale,
                    freeze_norm=freeze_norm,
                    initializer=initializer,
                    skip_quant=skip_quant))
        self.rbr_1x1 = ConvNormLayer(
            ch_in,
            ch_in,
            1,
            stride=self.stride,
            groups=ch_in,
            norm_type=norm_type,
            norm_decay=norm_decay,
            norm_groups=norm_groups,
            bias_on=bias_on,
            lr_scale=lr_scale,
            freeze_norm=freeze_norm,
            initializer=initializer,
            skip_quant=skip_quant)
        self.rbr_identity_st1 = nn.BatchNorm2D(
            num_features=ch_in,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(
                0.0))) if ch_in == ch_out and self.stride == 1 else None
        self.rbr_identity_st2 = nn.BatchNorm2D(
            num_features=ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(
                0.0))) if ch_in == ch_out and self.stride == 1 else None
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        if hasattr(self, "conv1") and hasattr(self, "conv2"):
            y = self.act(self.conv2(self.act(self.conv1(x))))
        else:
            if self.rbr_identity_st1 is None:
                id_out_st1 = 0
            else:
                id_out_st1 = self.rbr_identity_st1(x)

            x1_1 = 0
            for i in range(self.k):
                x1_1 += self.depth_conv[i](x)

            x1_2 = self.rbr_1x1(x)
            x1 = self.act(x1_1 + x1_2 + id_out_st1)

            if self.rbr_identity_st2 is None:
                id_out_st2 = 0
            else:
                id_out_st2 = self.rbr_identity_st2(x1)

            x2_1 = 0
            for i in range(self.k):
                x2_1 += self.point_conv[i](x1)
            y = self.act(x2_1 + id_out_st2)

        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv1'):
            self.conv1 = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_in,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.ch_in,
                bias_attr=ParamAttr(
                    initializer=Constant(value=0.), learning_rate=1.))
        if not hasattr(self, 'conv2'):
            self.conv2 = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=1,
                stride=1,
                padding='SAME',
                groups=1,
                bias_attr=ParamAttr(
                    initializer=Constant(value=0.), learning_rate=1.))

        conv1_kernel, conv1_bias, conv2_kernel, conv2_bias = self.get_equivalent_kernel_bias(
        )
        self.conv1.weight.set_value(conv1_kernel)
        self.conv1.bias.set_value(conv1_bias)
        self.conv2.weight.set_value(conv2_kernel)
        self.conv2.bias.set_value(conv2_bias)
        self.__delattr__('depth_conv')
        self.__delattr__('point_conv')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity_st1'):
            self.__delattr__('rbr_identity_st1')
        if hasattr(self, 'rbr_identity_st2'):
            self.__delattr__('rbr_identity_st2')

    def get_equivalent_kernel_bias(self):
        st1_kernel3x3, st1_bias3x3 = self._fuse_bn_tensor(self.depth_conv)
        st1_kernel1x1, st1_bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        st1_kernelid, st1_biasid = self._fuse_bn_tensor(
            self.rbr_identity_st1, kernel_size=self.kernel_size)

        st2_kernel1x1, st2_bias1x1 = self._fuse_bn_tensor(self.point_conv)
        st2_kernelid, st2_biasid = self._fuse_bn_tensor(
            self.rbr_identity_st2, kernel_size=1)

        conv1_kernel = st1_kernel3x3 + self._pad_1x1_to_3x3_tensor(
            st1_kernel1x1) + st1_kernelid

        conv1_bias = st1_bias3x3 + st1_bias1x1 + st1_biasid

        conv2_kernel = st2_kernel1x1 + st2_kernelid
        conv2_bias = st2_bias1x1 + st2_biasid

        return conv1_kernel, conv1_bias, conv2_kernel, conv2_bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            padding_size = (self.kernel_size - 1) // 2
            return nn.functional.pad(
                kernel1x1,
                [padding_size, padding_size, padding_size, padding_size])

    def _fuse_bn_tensor(self, branch, kernel_size=3):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.LayerList):
            fused_kernels = []
            fused_bias = []
            for block in branch:
                kernel = block.conv.weight
                running_mean = block.norm._mean
                running_var = block.norm._variance
                gamma = block.norm.weight
                beta = block.norm.bias
                eps = block.norm._epsilon

                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape((-1, 1, 1, 1))

                fused_kernels.append(kernel * t)
                fused_bias.append(beta - running_mean * gamma / std)

            return sum(fused_kernels), sum(fused_bias)

        elif isinstance(branch, ConvNormLayer):
            kernel = branch.conv.weight
            running_mean = branch.norm._mean
            running_var = branch.norm._variance
            gamma = branch.norm.weight
            beta = branch.norm.bias
            eps = branch.norm._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            input_dim = self.ch_in if kernel_size == 1 else 1
            kernel_value = paddle.zeros(
                shape=[self.ch_in, input_dim, kernel_size, kernel_size],
                dtype='float32')
            if kernel_size > 1:
                for i in range(self.ch_in):
                    kernel_value[i, i % input_dim, (kernel_size - 1) // 2, (
                        kernel_size - 1) // 2] = 1
            elif kernel_size == 1:
                for i in range(self.ch_in):
                    kernel_value[i, i % input_dim, 0, 0] = 1
            else:
                raise ValueError("Invalid kernel size recieved!")
            kernel = paddle.to_tensor(kernel_value, place=branch.weight.place)
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))

        return kernel * t, beta - running_mean * gamma / std
