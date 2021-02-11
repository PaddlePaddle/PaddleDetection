# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F
from paddle import ParamAttr
import paddle.nn as nn
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable

__all__ = ['HRFPN']


@register
class HRFPN(nn.Layer):
    """
    Args:
        in_channel (int): number of input feature channels from backbone
        out_channel (int): number of output feature channels
        share_conv (bool): whether to share conv for different layers' reduction
        spatial_scale (list): feature map scaling factor
    """

    def __init__(
            self,
            in_channel=270,
            out_channel=256,
            share_conv=False,
            spatial_scale=[1. / 4, 1. / 8, 1. / 16, 1. / 32, 1. / 64], ):
        super(HRFPN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.share_conv = share_conv
        self.spatial_scale = spatial_scale

        self.reduction = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            weight_attr=ParamAttr(name='hrfpn_reduction_weights'),
            bias_attr=False)
        self.num_out = len(self.spatial_scale)
        if share_conv:
            self.fpn_conv = nn.Conv2D(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
                weight_attr=ParamAttr(name='fpn_conv_weights'),
                bias_attr=False)
        else:
            self.fpn_conv = []
            for i in range(self.num_out):
                conv_name = "fpn_conv_" + str(i)
                conv = self.add_sublayer(
                    conv_name,
                    nn.Conv2D(
                        in_channels=out_channel,
                        out_channels=out_channel,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(name=conv_name + "_weights"),
                        bias_attr=False))
                self.fpn_conv.append(conv)

    def forward(self, body_feats):
        num_backbone_stages = len(body_feats)

        outs = []
        outs.append(body_feats[0])

        # resize
        for i in range(1, num_backbone_stages):
            resized = F.interpolate(
                body_feats[i], scale_factor=2**i, mode='bilinear')
            outs.append(resized)

        # concat
        out = paddle.concat(outs, axis=1)
        assert out.shape[
            1] == self.in_channel, 'in_channel should be {}, be received {}'.format(
                out.shape[1], self.in_channel)

        # reduction
        out = self.reduction(out)

        # conv
        outs = [out]
        for i in range(1, self.num_out):
            outs.append(F.avg_pool2d(out, kernel_size=2**i, stride=2**i))
        outputs = []

        for i in range(self.num_out):
            conv_func = self.fpn_conv if self.share_conv else self.fpn_conv[i]
            conv = conv_func(outs[i])
            outputs.append(conv)

        fpn_feat = [outputs[k] for k in range(self.num_out)]
        return fpn_feat, self.spatial_scale
