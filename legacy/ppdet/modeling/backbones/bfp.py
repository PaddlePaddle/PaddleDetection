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

from .nonlocal_helper import add_space_nonlocal
from .fpn import FPN

__all__ = ['BFP']


@register
class BFP(object):
    """
    Libra R-CNN, see https://arxiv.org/abs/1904.02701
    Args:
        base_neck (dict): basic neck before balanced feature pyramid (bfp)
        refine_level (int): index of integration and refine level of bfp
        refine_type (str): refine type, None, conv or nonlocal
        nonlocal_reduction (float): channel reduction level if refine_type is nonlocal
        with_bias (bool): whether the nonlocal module contains bias
        with_scale (bool): whether to scale feature in nonlocal module or not
    """
    __inject__ = ['base_neck']

    def __init__(self,
                 base_neck=FPN().__dict__,
                 refine_level=2,
                 refine_type="nonlocal",
                 nonlocal_reduction=1,
                 with_bias=True,
                 with_scale=False):
        if isinstance(base_neck, dict):
            self.base_neck = FPN(**base_neck)
        self.refine_level = refine_level
        self.refine_type = refine_type
        self.nonlocal_reduction = nonlocal_reduction
        self.with_bias = with_bias
        self.with_scale = with_scale

    def get_output(self, body_dict):
        # top-down order
        res_dict, spatial_scale = self.base_neck.get_output(body_dict)
        res_dict = self.get_output_bfp(res_dict)
        return res_dict, spatial_scale

    def get_output_bfp(self, body_dict):
        body_name_list = list(body_dict.keys())
        num_backbone_stages = len(body_name_list)

        self.num_levels = len(body_dict)

        # step 1: gather multi-level features by resize and average
        feats = []
        refine_level_name = body_name_list[self.refine_level]

        for i in range(self.num_levels):
            curr_fpn_name = body_name_list[i]
            pool_stride = 2**(i - self.refine_level)
            pool_size = [
                body_dict[refine_level_name].shape[2],
                body_dict[refine_level_name].shape[3]
            ]
            if i > self.refine_level:
                gathered = fluid.layers.pool2d(
                    input=body_dict[curr_fpn_name],
                    pool_type='max',
                    pool_size=pool_stride,
                    pool_stride=pool_stride,
                    ceil_mode=True, )
            else:
                gathered = self._resize_input_tensor(
                    body_dict[curr_fpn_name], body_dict[refine_level_name],
                    1.0 / pool_stride)
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type == "conv":
            bsf = fluid.layers.conv2d(
                bsf,
                bsf.shape[1],
                filter_size=3,
                padding=1,
                param_attr=ParamAttr(name="bsf_w"),
                bias_attr=ParamAttr(name="bsf_b"),
                name="bsf")
        elif self.refine_type == "nonlocal":
            dim_in = bsf.shape[1]
            nonlocal_name = "nonlocal_bsf"
            bsf = add_space_nonlocal(
                bsf,
                bsf.shape[1],
                bsf.shape[1],
                nonlocal_name,
                int(bsf.shape[1] / self.nonlocal_reduction),
                with_bias=self.with_bias,
                with_scale=self.with_scale)

        # step 3: scatter refined features to multi-levels by a residual path
        fpn_dict = {}
        fpn_name_list = []
        for i in range(self.num_levels):
            curr_fpn_name = body_name_list[i]
            pool_stride = 2**(self.refine_level - i)
            if i >= self.refine_level:
                residual = self._resize_input_tensor(
                    bsf, body_dict[curr_fpn_name], 1.0 / pool_stride)
            else:
                residual = fluid.layers.pool2d(
                    input=bsf,
                    pool_type='max',
                    pool_size=pool_stride,
                    pool_stride=pool_stride,
                    ceil_mode=True, )

            fpn_dict[curr_fpn_name] = residual + body_dict[curr_fpn_name]
            fpn_name_list.append(curr_fpn_name)

        res_dict = OrderedDict([(k, fpn_dict[k]) for k in fpn_name_list])
        return res_dict

    def _resize_input_tensor(self, body_input, ref_output, scale):
        shape = fluid.layers.shape(ref_output)
        shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
        out_shape_ = shape_hw
        out_shape = fluid.layers.cast(out_shape_, dtype='int32')
        out_shape.stop_gradient = True
        body_output = fluid.layers.resize_nearest(
            body_input, scale=scale, out_shape=out_shape)
        return body_output
