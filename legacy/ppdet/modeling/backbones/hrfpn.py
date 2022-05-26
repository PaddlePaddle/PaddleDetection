# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr

from ppdet.core.workspace import register

__all__ = ['HRFPN']


@register
class HRFPN(object):
    """
    HRNet, see https://arxiv.org/abs/1908.07919

    Args:
        num_chan (int): number of feature channels
        pooling_type (str): pooling type of downsampling
        share_conv (bool): whethet to share conv for different layers' reduction
        spatial_scale (list): feature map scaling factor
    """

    def __init__(
            self,
            num_chan=256,
            pooling_type="avg",
            share_conv=False,
            spatial_scale=[1. / 64, 1. / 32, 1. / 16, 1. / 8, 1. / 4], ):
        self.num_chan = num_chan
        self.pooling_type = pooling_type
        self.share_conv = share_conv
        self.spatial_scale = spatial_scale
        return

    def get_output(self, body_dict):
        num_out = len(self.spatial_scale)
        body_name_list = list(body_dict.keys())

        num_backbone_stages = len(body_name_list)

        outs = []
        outs.append(body_dict[body_name_list[0]])

        # resize
        for i in range(1, len(body_dict)):
            resized = self.resize_input_tensor(body_dict[body_name_list[i]],
                                               outs[0], 2**i)
            outs.append(resized)

        # concat
        out = fluid.layers.concat(outs, axis=1)

        # reduction
        out = fluid.layers.conv2d(
            input=out,
            num_filters=self.num_chan,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr=ParamAttr(name='hrfpn_reduction_weights'),
            bias_attr=False)

        # conv
        outs = [out]
        for i in range(1, num_out):
            outs.append(
                self.pooling(
                    out, size=2**i, stride=2**i,
                    pooling_type=self.pooling_type))
        outputs = []

        for i in range(num_out):
            conv_name = "shared_fpn_conv" if self.share_conv else "shared_fpn_conv_" + str(
                i)
            conv = fluid.layers.conv2d(
                input=outs[i],
                num_filters=self.num_chan,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=ParamAttr(name=conv_name + "_weights"),
                bias_attr=False)
            outputs.append(conv)

        for idx in range(0, num_out - len(body_name_list)):
            body_name_list.append("fpn_res5_sum_subsampled_{}x".format(2**(idx +
                                                                           1)))

        outputs = outputs[::-1]
        body_name_list = body_name_list[::-1]

        res_dict = OrderedDict([(body_name_list[k], outputs[k])
                                for k in range(len(body_name_list))])
        return res_dict, self.spatial_scale

    def resize_input_tensor(self, body_input, ref_output, scale):
        shape = fluid.layers.shape(ref_output)
        shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
        out_shape_ = shape_hw
        out_shape = fluid.layers.cast(out_shape_, dtype='int32')
        out_shape.stop_gradient = True
        body_output = fluid.layers.resize_bilinear(
            body_input, scale=scale, out_shape=out_shape)
        return body_output

    def pooling(self, input, size, stride, pooling_type):
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=size,
            pool_stride=stride,
            pool_type=pooling_type)
        return pool
