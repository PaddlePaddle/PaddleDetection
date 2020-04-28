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

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import TruncatedNormal, Constant
from paddle.fluid.regularizer import L2Decay
from ppdet.modeling.ops import RetinaOutputDecoder

from ppdet.core.workspace import register

__all__ = ['EfficientHead']


@register
class EfficientHead(object):
    """
    EfficientDet Head

    Args:
        output_decoder (object): `RetinaOutputDecoder` instance.
        repeat (int): Number of convolution layers.
        num_chan (int): Number of octave output channels.
        prior_prob (float): Initial value of the class prediction layer bias.
        num_anchors (int): Number of anchors per cell.
        num_classes (int): Number of classes.
        gamma (float): Gamma parameter for focal loss.
        alpha (float): Alpha parameter for focal loss.
        sigma (float): Sigma parameter for smooth l1 loss.
    """
    __inject__ = ['output_decoder']
    __shared__ = ['num_classes']

    def __init__(self,
                 output_decoder=RetinaOutputDecoder().__dict__,
                 repeat=3,
                 num_chan=64,
                 prior_prob=0.01,
                 num_anchors=9,
                 num_classes=81,
                 gamma=1.5,
                 alpha=0.25,
                 delta=0.1):
        super(EfficientHead, self).__init__()
        self.output_decoder = output_decoder
        self.repeat = repeat
        self.num_chan = num_chan
        self.prior_prob = prior_prob
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        if isinstance(output_decoder, dict):
            self.output_decoder = RetinaOutputDecoder(**output_decoder)

    def _get_output(self, body_feats):
        def separable_conv(inputs, num_chan, bias_init=None, name=''):
            dw_conv_name = name + '_dw'
            pw_conv_name = name + '_pw'
            in_chan = inputs.shape[1]
            fan_in = np.sqrt(1. / (in_chan * 3 * 3))
            feat = fluid.layers.conv2d(
                input=inputs,
                num_filters=in_chan,
                groups=in_chan,
                filter_size=3,
                stride=1,
                padding='SAME',
                param_attr=ParamAttr(
                    name=dw_conv_name + '_w',
                    initializer=TruncatedNormal(scale=fan_in)),
                bias_attr=False)
            fan_in = np.sqrt(1. / in_chan)
            feat = fluid.layers.conv2d(
                input=feat,
                num_filters=num_chan,
                filter_size=1,
                stride=1,
                param_attr=ParamAttr(
                    name=pw_conv_name + '_w',
                    initializer=TruncatedNormal(scale=fan_in)),
                bias_attr=ParamAttr(
                    name=pw_conv_name + '_b',
                    initializer=bias_init,
                    regularizer=L2Decay(0.)))
            return feat

        def subnet(inputs, prefix, level):
            feat = inputs
            for i in range(self.repeat):
                # NOTE share weight across FPN levels
                conv_name = '{}_pred_conv_{}'.format(prefix, i)
                feat = separable_conv(feat, self.num_chan, name=conv_name)
                # NOTE batch norm params are not shared
                bn_name = '{}_pred_bn_{}_{}'.format(prefix, level, i)
                feat = fluid.layers.batch_norm(
                    input=feat,
                    act='swish',
                    momentum=0.997,
                    epsilon=1e-4,
                    moving_mean_name=bn_name + '_mean',
                    moving_variance_name=bn_name + '_variance',
                    param_attr=ParamAttr(
                        name=bn_name + '_w',
                        initializer=Constant(value=1.),
                        regularizer=L2Decay(0.)),
                    bias_attr=ParamAttr(
                        name=bn_name + '_b', regularizer=L2Decay(0.)))
            return feat

        cls_preds = []
        box_preds = []
        for l, feat in enumerate(body_feats):
            cls_out = subnet(feat, 'cls', l)
            box_out = subnet(feat, 'box', l)

            bias_init = float(-np.log((1 - self.prior_prob) / self.prior_prob))
            bias_init = Constant(value=bias_init)
            cls_pred = separable_conv(
                cls_out,
                self.num_anchors * (self.num_classes - 1),
                bias_init=bias_init,
                name='cls_pred')
            cls_pred = fluid.layers.transpose(cls_pred, perm=[0, 2, 3, 1])
            cls_pred = fluid.layers.reshape(
                cls_pred, shape=(0, -1, self.num_classes - 1))
            cls_preds.append(cls_pred)

            box_pred = separable_conv(
                box_out, self.num_anchors * 4, name='box_pred')
            box_pred = fluid.layers.transpose(box_pred, perm=[0, 2, 3, 1])
            box_pred = fluid.layers.reshape(box_pred, shape=(0, -1, 4))
            box_preds.append(box_pred)

        return cls_preds, box_preds

    def get_prediction(self, body_feats, anchors, im_info):
        cls_preds, box_preds = self._get_output(body_feats)
        cls_preds = [fluid.layers.sigmoid(pred) for pred in cls_preds]
        pred_result = self.output_decoder(
            bboxes=box_preds,
            scores=cls_preds,
            anchors=anchors,
            im_info=im_info)
        return {'bbox': pred_result}

    def get_loss(self, body_feats, gt_labels, gt_targets, fg_num):
        cls_preds, box_preds = self._get_output(body_feats)
        fg_num = fluid.layers.reduce_sum(fg_num, name='fg_num')
        fg_num.stop_gradient = True

        cls_pred = fluid.layers.concat(cls_preds, axis=1)
        box_pred = fluid.layers.concat(box_preds, axis=1)
        cls_pred_reshape = fluid.layers.reshape(
            cls_pred, shape=(-1, self.num_classes - 1))
        gt_labels_reshape = fluid.layers.reshape(gt_labels, shape=(-1, 1))
        loss_cls = fluid.layers.sigmoid_focal_loss(
            x=cls_pred_reshape,
            label=gt_labels_reshape,
            fg_num=fg_num,
            gamma=self.gamma,
            alpha=self.alpha)
        loss_cls = fluid.layers.reduce_sum(loss_cls)

        loss_bbox = fluid.layers.huber_loss(
            input=box_pred, label=gt_targets, delta=self.delta)
        mask = fluid.layers.expand(gt_labels, expand_times=[1, 1, 4]) > 0
        loss_bbox *= fluid.layers.cast(mask, 'float32')
        loss_bbox = fluid.layers.reduce_sum(loss_bbox) / (fg_num * 4)

        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}
