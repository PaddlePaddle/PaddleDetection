import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant, Xavier
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register, serializable


@register
@serializable
class FPN(Layer):
    def __init__(self,
                 in_channels,
                 out_channel,
                 min_level=0,
                 max_level=4,
                 spatial_scale=[0.25, 0.125, 0.0625, 0.03125]):

        super(FPN, self).__init__()
        self.lateral_convs = []
        self.fpn_convs = []
        fan = out_channel * 3 * 3

        for i in range(min_level, max_level):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i]
            lateral = self.add_sublayer(
                lateral_name,
                Conv2D(
                    num_channels=in_c,
                    num_filters=out_channel,
                    filter_size=1,
                    param_attr=ParamAttr(
                        #name=lateral_name+'_w', 
                        initializer=Xavier(fan_out=in_c)),
                    bias_attr=ParamAttr(
                        #name=lateral_name+'_b', 
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))
            self.lateral_convs.append(lateral)

            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            fpn_conv = self.add_sublayer(
                fpn_name,
                Conv2D(
                    num_channels=out_channel,
                    num_filters=out_channel,
                    filter_size=3,
                    padding=1,
                    param_attr=ParamAttr(
                        #name=fpn_name+'_w', 
                        initializer=Xavier(fan_out=fan)),
                    bias_attr=ParamAttr(
                        #name=fpn_name+'_b', 
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))
            self.fpn_convs.append(fpn_conv)

        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale

    def forward(self, body_feats):
        laterals = []
        for lvl in range(self.min_level, self.max_level):
            laterals.append(self.lateral_convs[lvl](body_feats[lvl]))

        for lvl in range(self.max_level - 1, self.min_level, -1):
            upsample = fluid.layers.resize_nearest(laterals[lvl], scale=2.)
            laterals[lvl - 1] = laterals[lvl - 1] + upsample

        fpn_output = []
        for lvl in range(self.min_level, self.max_level):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        extension = fluid.layers.pool2d(fpn_output[-1], 1, 'max', pool_stride=2)

        spatial_scale = self.spatial_scale + [self.spatial_scale[-1] * 0.5]
        fpn_output.append(extension)
        return fpn_output, spatial_scale
