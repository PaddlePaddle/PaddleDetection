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
import copy
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling.ops import ConvNorm

__all__ = ['ACFPN']


@register
class ACFPN(object):
    """
    Attention-guided Context Feature Pyramid Network for Object Detection,
        see https://arxiv.org/abs/2005.11475

    Args:
        num_chan (int): number of feature channels
        min_level (int): lowest level of the backbone feature map to use
        max_level (int): highest level of the backbone feature map to use
        spatial_scale (list): feature map scaling factor
        has_extra_convs (bool): whether has extral convolutions in higher levels
        norm_type (str|None): normalization type, 'bn'/'sync_bn'/'affine_channel'
        use_c5 (bool): whether to use C5 as the feature map.
        norm_groups (int): group number of group norm.
    """
    __shared__ = ['norm_type', 'freeze_norm']

    def __init__(self,
                 num_chan=256,
                 min_level=2,
                 max_level=6,
                 spatial_scale=[1. / 32., 1. / 16., 1. / 8., 1. / 4.],
                 has_extra_convs=False,
                 norm_type=None,
                 freeze_norm=False,
                 use_c5=True,
                 norm_groups=32):
        self.freeze_norm = freeze_norm
        self.num_chan = num_chan
        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale
        self.has_extra_convs = has_extra_convs
        self.norm_type = norm_type
        self.use_c5 = use_c5
        self.norm_groups = norm_groups

    def _add_topdown_lateral(self, body_name, body_input, upper_output):
        lateral_name = 'fpn_inner_' + body_name + '_lateral'
        topdown_name = 'fpn_topdown_' + body_name
        fan = body_input.shape[1]
        if self.norm_type:
            initializer = Xavier(fan_out=fan)
            lateral = ConvNorm(
                body_input,
                self.num_chan,
                1,
                initializer=initializer,
                norm_type=self.norm_type,
                freeze_norm=self.freeze_norm,
                name=lateral_name,
                norm_name=lateral_name)
        else:
            lateral = fluid.layers.conv2d(
                body_input,
                self.num_chan,
                1,
                param_attr=ParamAttr(
                    name=lateral_name + "_w", initializer=Xavier(fan_out=fan)),
                bias_attr=ParamAttr(
                    name=lateral_name + "_b",
                    learning_rate=2.,
                    regularizer=L2Decay(0.)),
                name=lateral_name)
        topdown = fluid.layers.resize_nearest(
            upper_output, scale=2., name=topdown_name)

        return lateral + topdown

    def dense_aspp_block(self, input, num_filters1, num_filters2, dilation_rate,
                         dropout_prob, name):

        conv = ConvNorm(
            input,
            num_filters=num_filters1,
            filter_size=1,
            stride=1,
            groups=1,
            norm_decay=0.,
            norm_type='gn',
            norm_groups=self.norm_groups,
            dilation=dilation_rate,
            lr_scale=1,
            freeze_norm=False,
            act="relu",
            norm_name=name + "_gn",
            initializer=None,
            bias_attr=False,
            name=name + "_gn")

        conv = fluid.layers.conv2d(
            conv,
            num_filters2,
            filter_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
            act="relu",
            param_attr=ParamAttr(name=name + "_conv_w"),
            bias_attr=ParamAttr(name=name + "_conv_b"), )

        if dropout_prob > 0:
            conv = fluid.layers.dropout(conv, dropout_prob=dropout_prob)

        return conv

    def dense_aspp(self, input, name=None):
        dropout0 = 0.1
        d_feature0 = 512
        d_feature1 = 256

        aspp3 = self.dense_aspp_block(
            input,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp3',
            dilation_rate=3)
        conv = fluid.layers.concat([aspp3, input], axis=1)

        aspp6 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp6',
            dilation_rate=6)
        conv = fluid.layers.concat([aspp6, conv], axis=1)

        aspp12 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp12',
            dilation_rate=12)
        conv = fluid.layers.concat([aspp12, conv], axis=1)

        aspp18 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp18',
            dilation_rate=18)
        conv = fluid.layers.concat([aspp18, conv], axis=1)

        aspp24 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp24',
            dilation_rate=24)

        conv = fluid.layers.concat(
            [aspp3, aspp6, aspp12, aspp18, aspp24], axis=1)

        conv = ConvNorm(
            conv,
            num_filters=self.num_chan,
            filter_size=1,
            stride=1,
            groups=1,
            norm_decay=0.,
            norm_type='gn',
            norm_groups=self.norm_groups,
            dilation=1,
            lr_scale=1,
            freeze_norm=False,
            act="relu",
            norm_name=name + "_dense_aspp_reduce_gn",
            initializer=None,
            bias_attr=False,
            name=name + "_dense_aspp_reduce_gn")

        return conv

    def get_output(self, body_dict):
        """
        Add FPN onto backbone.

        Args:
            body_dict(OrderedDict): Dictionary of variables and each element is the
                output of backbone.

        Return:
            fpn_dict(OrderedDict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.
        """
        spatial_scale = copy.deepcopy(self.spatial_scale)
        body_name_list = list(body_dict.keys())[::-1]
        num_backbone_stages = len(body_name_list)
        self.fpn_inner_output = [[] for _ in range(num_backbone_stages)]
        fpn_inner_name = 'fpn_inner_' + body_name_list[0]
        body_input = body_dict[body_name_list[0]]
        fan = body_input.shape[1]
        if self.norm_type:
            initializer = Xavier(fan_out=fan)
            self.fpn_inner_output[0] = ConvNorm(
                body_input,
                self.num_chan,
                1,
                initializer=initializer,
                norm_type=self.norm_type,
                freeze_norm=self.freeze_norm,
                name=fpn_inner_name,
                norm_name=fpn_inner_name)
        else:
            self.fpn_inner_output[0] = fluid.layers.conv2d(
                body_input,
                self.num_chan,
                1,
                param_attr=ParamAttr(
                    name=fpn_inner_name + "_w",
                    initializer=Xavier(fan_out=fan)),
                bias_attr=ParamAttr(
                    name=fpn_inner_name + "_b",
                    learning_rate=2.,
                    regularizer=L2Decay(0.)),
                name=fpn_inner_name)

        self.fpn_inner_output[0] += self.dense_aspp(
            self.fpn_inner_output[0], name="acfpn")

        for i in range(1, num_backbone_stages):
            body_name = body_name_list[i]
            body_input = body_dict[body_name]
            top_output = self.fpn_inner_output[i - 1]
            fpn_inner_single = self._add_topdown_lateral(body_name, body_input,
                                                         top_output)
            self.fpn_inner_output[i] = fpn_inner_single
        fpn_dict = {}
        fpn_name_list = []
        for i in range(num_backbone_stages):
            fpn_name = 'fpn_' + body_name_list[i]
            fan = self.fpn_inner_output[i].shape[1] * 3 * 3
            if self.norm_type:
                initializer = Xavier(fan_out=fan)
                fpn_output = ConvNorm(
                    self.fpn_inner_output[i],
                    self.num_chan,
                    3,
                    initializer=initializer,
                    norm_type=self.norm_type,
                    freeze_norm=self.freeze_norm,
                    name=fpn_name,
                    norm_name=fpn_name)
            else:
                fpn_output = fluid.layers.conv2d(
                    self.fpn_inner_output[i],
                    self.num_chan,
                    filter_size=3,
                    padding=1,
                    param_attr=ParamAttr(
                        name=fpn_name + "_w", initializer=Xavier(fan_out=fan)),
                    bias_attr=ParamAttr(
                        name=fpn_name + "_b",
                        learning_rate=2.,
                        regularizer=L2Decay(0.)),
                    name=fpn_name)
            fpn_dict[fpn_name] = fpn_output
            fpn_name_list.append(fpn_name)
        if not self.has_extra_convs and self.max_level - self.min_level == len(
                spatial_scale):
            body_top_name = fpn_name_list[0]
            body_top_extension = fluid.layers.pool2d(
                fpn_dict[body_top_name],
                1,
                'max',
                pool_stride=2,
                name=body_top_name + '_subsampled_2x')
            fpn_dict[body_top_name + '_subsampled_2x'] = body_top_extension
            fpn_name_list.insert(0, body_top_name + '_subsampled_2x')
            spatial_scale.insert(0, spatial_scale[0] * 0.5)
        # Coarser FPN levels introduced for RetinaNet
        highest_backbone_level = self.min_level + len(spatial_scale) - 1
        if self.has_extra_convs and self.max_level > highest_backbone_level:
            if self.use_c5:
                fpn_blob = body_dict[body_name_list[0]]
            else:
                fpn_blob = fpn_dict[fpn_name_list[0]]
            for i in range(highest_backbone_level + 1, self.max_level + 1):
                fpn_blob_in = fpn_blob
                fpn_name = 'fpn_' + str(i)
                if i > highest_backbone_level + 1:
                    fpn_blob_in = fluid.layers.relu(fpn_blob)
                fan = fpn_blob_in.shape[1] * 3 * 3
                fpn_blob = fluid.layers.conv2d(
                    input=fpn_blob_in,
                    num_filters=self.num_chan,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    param_attr=ParamAttr(
                        name=fpn_name + "_w", initializer=Xavier(fan_out=fan)),
                    bias_attr=ParamAttr(
                        name=fpn_name + "_b",
                        learning_rate=2.,
                        regularizer=L2Decay(0.)),
                    name=fpn_name)
                fpn_dict[fpn_name] = fpn_blob
                fpn_name_list.insert(0, fpn_name)
                spatial_scale.insert(0, spatial_scale[0] * 0.5)
        res_dict = OrderedDict([(k, fpn_dict[k]) for k in fpn_name_list])
        return res_dict, spatial_scale
