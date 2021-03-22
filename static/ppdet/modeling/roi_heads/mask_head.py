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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling.ops import ConvNorm

__all__ = ['MaskHead']


@register
class MaskHead(object):
    """
    RCNN mask head
    Args:
        num_convs (int): num of convolutions, 4 for FPN, 1 otherwise
        conv_dim (int): num of channels after first convolution
        resolution (int): size of the output mask
        dilation (int): dilation rate
        num_classes (int): number of output classes
    """

    __shared__ = ['num_classes']

    def __init__(self,
                 num_convs=0,
                 conv_dim=256,
                 resolution=14,
                 dilation=1,
                 num_classes=81,
                 norm_type=None):
        super(MaskHead, self).__init__()
        self.num_convs = num_convs
        self.conv_dim = conv_dim
        self.resolution = resolution
        self.dilation = dilation
        self.num_classes = num_classes
        self.norm_type = norm_type

    def _mask_conv_head(self, roi_feat, num_convs, norm_type):
        if norm_type == 'gn':
            for i in range(num_convs):
                layer_name = "mask_inter_feat_" + str(i + 1)
                fan = self.conv_dim * 3 * 3
                initializer = MSRA(uniform=False, fan_in=fan)
                roi_feat = ConvNorm(
                    roi_feat,
                    self.conv_dim,
                    3,
                    act='relu',
                    dilation=self.dilation,
                    initializer=initializer,
                    norm_type=self.norm_type,
                    name=layer_name,
                    norm_name=layer_name)
        else:
            for i in range(num_convs):
                layer_name = "mask_inter_feat_" + str(i + 1)
                fan = self.conv_dim * 3 * 3
                initializer = MSRA(uniform=False, fan_in=fan)
                roi_feat = fluid.layers.conv2d(
                    input=roi_feat,
                    num_filters=self.conv_dim,
                    filter_size=3,
                    padding=1 * self.dilation,
                    act='relu',
                    stride=1,
                    dilation=self.dilation,
                    name=layer_name,
                    param_attr=ParamAttr(
                        name=layer_name + '_w', initializer=initializer),
                    bias_attr=ParamAttr(
                        name=layer_name + '_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.)))
        fan = roi_feat.shape[1] * 2 * 2
        feat = fluid.layers.conv2d_transpose(
            input=roi_feat,
            num_filters=self.conv_dim,
            filter_size=2,
            stride=2,
            act='relu',
            param_attr=ParamAttr(
                name='conv5_mask_w',
                initializer=MSRA(
                    uniform=False, fan_in=fan)),
            bias_attr=ParamAttr(
                name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))
        return feat

    def _get_output(self, roi_feat):
        class_num = self.num_classes
        # configure the conv number for FPN if necessary
        head_feat = self._mask_conv_head(roi_feat, self.num_convs,
                                         self.norm_type)
        fan = class_num
        mask_logits = fluid.layers.conv2d(
            input=head_feat,
            num_filters=class_num,
            filter_size=1,
            act=None,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w',
                initializer=MSRA(
                    uniform=False, fan_in=fan)),
            bias_attr=ParamAttr(
                name="mask_fcn_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return mask_logits

    def get_loss(self, roi_feat, mask_int32):
        mask_logits = self._get_output(roi_feat)
        num_classes = self.num_classes
        resolution = self.resolution
        dim = num_classes * resolution * resolution
        mask_logits = fluid.layers.reshape(mask_logits, (-1, dim))

        mask_label = fluid.layers.cast(x=mask_int32, dtype='float32')
        mask_label.stop_gradient = True
        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=mask_logits, label=mask_label, ignore_index=-1, normalize=True)
        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')
        return {'loss_mask': loss_mask}

    def get_prediction(self, roi_feat, bbox_pred):
        """
        Get prediction mask in test stage.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            bbox_pred (Variable): predicted bbox.

        Returns:
            mask_pred (Variable): Prediction mask with shape
                [N, num_classes, resolution, resolution].
        """
        mask_logits = self._get_output(roi_feat)
        mask_prob = fluid.layers.sigmoid(mask_logits)
        mask_prob = fluid.layers.lod_reset(mask_prob, bbox_pred)
        return mask_prob
