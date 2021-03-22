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

import math

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register, serializable
from .resnext import ResNeXt

__all__ = ['SENet', 'SENetC5']


@register
@serializable
class SENet(ResNeXt):
    """
    Squeeze-and-Excitation Networks, see https://arxiv.org/abs/1709.01507
    Args:
        depth (int): SENet depth, should be 50, 101, 152
        groups (int): group convolution cardinality
        group_width (int): width of each group convolution
        freeze_at (int): freeze the backbone at which stage
        norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
        freeze_norm (bool): freeze normalization layers
        norm_decay (float): weight decay for normalization layer weights
        variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
        feature_maps (list): index of the stages whose feature maps are returned
        dcn_v2_stages (list): index of stages who select deformable conv v2
    """

    def __init__(self,
                 depth=50,
                 groups=64,
                 group_width=4,
                 freeze_at=2,
                 norm_type='affine_channel',
                 freeze_norm=True,
                 norm_decay=0.,
                 variant='d',
                 feature_maps=[2, 3, 4, 5],
                 dcn_v2_stages=[],
                 std_senet=False,
                 weight_prefix_name=''):
        super(SENet, self).__init__(depth, groups, group_width, freeze_at,
                                    norm_type, freeze_norm, norm_decay, variant,
                                    feature_maps)
        if depth < 152:
            self.stage_filters = [128, 256, 512, 1024]
        else:
            self.stage_filters = [256, 512, 1024, 2048]
        self.reduction_ratio = 16
        self.std_senet = std_senet
        self._c1_out_chan_num = 128
        self._model_type = 'SEResNeXt'
        self.dcn_v2_stages = dcn_v2_stages

    def _squeeze_excitation(self, input, num_channels, name=None):
        mixed_precision_enabled = mixed_precision_global_state() is not None
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_type='avg',
            global_pooling=True,
            use_cudnn=mixed_precision_enabled)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=int(num_channels / self.reduction_ratio),
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale


@register
@serializable
class SENetC5(SENet):
    __doc__ = SENet.__doc__

    def __init__(self,
                 depth=50,
                 groups=64,
                 group_width=4,
                 freeze_at=2,
                 norm_type='affine_channel',
                 freeze_norm=True,
                 norm_decay=0.,
                 variant='d',
                 feature_maps=[5],
                 weight_prefix_name=''):
        super(SENetC5, self).__init__(depth, groups, group_width, freeze_at,
                                      norm_type, freeze_norm, norm_decay,
                                      variant, feature_maps)
        self.severed_head = True
