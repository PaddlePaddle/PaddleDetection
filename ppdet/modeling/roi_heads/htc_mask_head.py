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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling.ops import ConvNorm

__all__ = ['HTCMaskHead']


@register
class HTCMaskHead(object):
    """
    htc mask head
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
                 norm_type=None,
                 lr_ratio=2.0,
                 share_mask_conv=False):
        super(HTCMaskHead, self).__init__()
        self.num_convs = num_convs
        self.conv_dim = conv_dim
        self.resolution = resolution
        self.dilation = dilation
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.lr_ratio = lr_ratio
        self.share_mask_conv = share_mask_conv

    def _mask_conv_head(self,
                        roi_feat,
                        num_convs,
                        norm_type,
                        wb_scalar=1.0,
                        name=''):
        if norm_type == 'gn':
            for i in range(num_convs):
                layer_name = "mask_inter_feat_" + str(i + 1)
                if not self.share_mask_conv:
                    layer_name += name
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
                if not self.share_mask_conv:
                    layer_name += name
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
                        learning_rate=wb_scalar * self.lr_ratio,
                        regularizer=L2Decay(0.)))
        return roi_feat

    def get_output(self,
                   roi_feat,
                   res_feat=None,
                   return_logits=True,
                   return_feat=False,
                   wb_scalar=1.0,
                   name=''):
        class_num = self.num_classes
        if res_feat is not None:
            res_feat = fluid.layers.conv2d(
                res_feat, roi_feat.shape[1], 1, name='res_net' + name)
            roi_feat = fluid.layers.sum([roi_feat, res_feat])
        # configure the conv number for FPN if necessary
        head_feat = self._mask_conv_head(roi_feat, self.num_convs,
                                         self.norm_type, wb_scalar, name)

        if return_logits:
            fan0 = roi_feat.shape[1] * 2 * 2
            up_head_feat = fluid.layers.conv2d_transpose(
                input=head_feat,
                num_filters=self.conv_dim,
                filter_size=2,
                stride=2,
                act='relu',
                param_attr=ParamAttr(
                    name='conv5_mask_w' + name,
                    initializer=MSRA(
                        uniform=False, fan_in=fan0)),
                bias_attr=ParamAttr(
                    name='conv5_mask_b' + name,
                    learning_rate=wb_scalar * self.lr_ratio,
                    regularizer=L2Decay(0.)))

            fan = class_num
            mask_logits = fluid.layers.conv2d(
                input=up_head_feat,
                num_filters=class_num,
                filter_size=1,
                act=None,
                param_attr=ParamAttr(
                    name='mask_fcn_logits_w' + name,
                    initializer=MSRA(
                        uniform=False, fan_in=fan)),
                bias_attr=ParamAttr(
                    name="mask_fcn_logits_b" + name,
                    learning_rate=wb_scalar * self.lr_ratio,
                    regularizer=L2Decay(0.)))
            if return_feat:
                return mask_logits, head_feat
            else:
                return mask_logits

        if return_feat:
            return head_feat

    def get_loss(self,
                 mask_logits_list,
                 mask_int32_list,
                 cascade_loss_weights=[1.0, 0.5, 0.25]):
        num_classes = self.num_classes
        resolution = self.resolution
        dim = num_classes * resolution * resolution
        loss_mask_dict = {}
        for i, (mask_logits, mask_int32
                ) in enumerate(zip(mask_logits_list, mask_int32_list)):

            mask_logits = fluid.layers.reshape(mask_logits, (-1, dim))
            mask_label = fluid.layers.cast(x=mask_int32, dtype='float32')
            mask_label.stop_gradient = True
            loss_name = 'loss_mask_' + str(i)
            loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=mask_logits,
                label=mask_label,
                ignore_index=-1,
                normalize=True,
                name=loss_name)
            loss_mask = fluid.layers.reduce_sum(
                loss_mask) * cascade_loss_weights[i]
            loss_mask_dict[loss_name] = loss_mask
        return loss_mask_dict

    def get_prediction(self, mask_logits, bbox_pred):
        """
        Get prediction mask in test stage.

        Args:
            mask_logits (Variable): mask head output features.
            bbox_pred (Variable): predicted bbox.

        Returns:
            mask_pred (Variable): Prediction mask with shape
                [N, num_classes, resolution, resolution].
        """
        mask_prob = fluid.layers.sigmoid(mask_logits)
        mask_prob = fluid.layers.lod_reset(mask_prob, bbox_pred)
        return mask_prob
