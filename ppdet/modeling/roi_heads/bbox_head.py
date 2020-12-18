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
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA

from ppdet.modeling.ops import MultiClassNMS
from ppdet.modeling.ops import ConvNorm
from ppdet.modeling.losses import SmoothL1Loss
from ppdet.core.workspace import register, serializable
from ppdet.experimental import mixed_precision_global_state

__all__ = ['BBoxHead', 'TwoFCHead', 'XConvNormHead']


@register
@serializable
class BoxCoder(object):
    __op__ = fluid.layers.box_coder
    __append_doc__ = True

    def __init__(self,
                 prior_box_var=[0.1, 0.1, 0.2, 0.2],
                 code_type='decode_center_size',
                 box_normalized=False,
                 axis=1):
        super(BoxCoder, self).__init__()
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.axis = axis


@register
class XConvNormHead(object):
    """
    RCNN head with serveral convolution layers

    Args:
        conv_num (int): num of convolution layers for the rcnn head
        conv_dim (int): num of filters for the conv layers
        mlp_dim (int): num of filters for the fc layers
    """
    __shared__ = ['norm_type', 'freeze_norm']

    def __init__(self,
                 num_conv=4,
                 conv_dim=256,
                 mlp_dim=1024,
                 norm_type=None,
                 freeze_norm=False):
        super(XConvNormHead, self).__init__()
        self.conv_dim = conv_dim
        self.mlp_dim = mlp_dim
        self.num_conv = num_conv
        self.norm_type = norm_type
        self.freeze_norm = freeze_norm

    def __call__(self, roi_feat):
        conv = roi_feat
        fan = self.conv_dim * 3 * 3
        initializer = MSRA(uniform=False, fan_in=fan)
        for i in range(self.num_conv):
            name = 'bbox_head_conv' + str(i)
            conv = ConvNorm(
                conv,
                self.conv_dim,
                3,
                act='relu',
                initializer=initializer,
                norm_type=self.norm_type,
                freeze_norm=self.freeze_norm,
                name=name,
                norm_name=name)
        fan = conv.shape[1] * conv.shape[2] * conv.shape[3]
        head_heat = fluid.layers.fc(input=conv,
                                    size=self.mlp_dim,
                                    act='relu',
                                    name='fc6' + name,
                                    param_attr=ParamAttr(
                                        name='fc6%s_w' % name,
                                        initializer=Xavier(fan_out=fan)),
                                    bias_attr=ParamAttr(
                                        name='fc6%s_b' % name,
                                        learning_rate=2,
                                        regularizer=L2Decay(0.)))
        return head_heat


@register
class TwoFCHead(object):
    """
    RCNN head with two Fully Connected layers

    Args:
        mlp_dim (int): num of filters for the fc layers
    """

    def __init__(self, mlp_dim=1024):
        super(TwoFCHead, self).__init__()
        self.mlp_dim = mlp_dim

    def __call__(self, roi_feat):
        fan = roi_feat.shape[1] * roi_feat.shape[2] * roi_feat.shape[3]

        mixed_precision_enabled = mixed_precision_global_state() is not None

        if mixed_precision_enabled:
            roi_feat = fluid.layers.cast(roi_feat, 'float16')

        fc6 = fluid.layers.fc(input=roi_feat,
                              size=self.mlp_dim,
                              act='relu',
                              name='fc6',
                              param_attr=ParamAttr(
                                  name='fc6_w',
                                  initializer=Xavier(fan_out=fan)),
                              bias_attr=ParamAttr(
                                  name='fc6_b',
                                  learning_rate=2.,
                                  regularizer=L2Decay(0.)))
        head_feat = fluid.layers.fc(input=fc6,
                                    size=self.mlp_dim,
                                    act='relu',
                                    name='fc7',
                                    param_attr=ParamAttr(
                                        name='fc7_w', initializer=Xavier()),
                                    bias_attr=ParamAttr(
                                        name='fc7_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))

        if mixed_precision_enabled:
            head_feat = fluid.layers.cast(head_feat, 'float32')

        return head_feat


@register
class BBoxHead(object):
    """
    RCNN bbox head

    Args:
        head (object): the head module instance, e.g., `ResNetC5`, `TwoFCHead`
        box_coder (object): `BoxCoder` instance
        nms (object): `MultiClassNMS` instance
        num_classes: number of output classes
    """
    __inject__ = ['head', 'box_coder', 'nms', 'bbox_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 head,
                 box_coder=BoxCoder().__dict__,
                 nms=MultiClassNMS().__dict__,
                 bbox_loss=SmoothL1Loss().__dict__,
                 num_classes=81):
        super(BBoxHead, self).__init__()
        self.head = head
        self.num_classes = num_classes
        self.box_coder = box_coder
        self.nms = nms
        self.bbox_loss = bbox_loss
        if isinstance(box_coder, dict):
            self.box_coder = BoxCoder(**box_coder)
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        if isinstance(bbox_loss, dict):
            self.bbox_loss = SmoothL1Loss(**bbox_loss)
        self.head_feat = None

    def get_head_feat(self, input=None):
        """
        Get the bbox head feature map.
        """

        if input is not None:
            feat = self.head(input)
            if isinstance(feat, OrderedDict):
                feat = list(feat.values())[0]
            self.head_feat = feat
        return self.head_feat

    def _get_output(self, roi_feat):
        """
        Get bbox head output.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.

        Returns:
            cls_score(Variable): Output of rpn head with shape of
                [N, num_anchors, H, W].
            bbox_pred(Variable): Output of rpn head with shape of
                [N, num_anchors * 4, H, W].
        """
        head_feat = self.get_head_feat(roi_feat)
        # when ResNetC5 output a single feature map
        if not isinstance(self.head, TwoFCHead) and not isinstance(
                self.head, XConvNormHead):
            head_feat = fluid.layers.pool2d(
                head_feat, pool_type='avg', global_pooling=True)
        cls_score = fluid.layers.fc(input=head_feat,
                                    size=self.num_classes,
                                    act=None,
                                    name='cls_score',
                                    param_attr=ParamAttr(
                                        name='cls_score_w',
                                        initializer=Normal(
                                            loc=0.0, scale=0.01)),
                                    bias_attr=ParamAttr(
                                        name='cls_score_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(input=head_feat,
                                    size=4 * self.num_classes,
                                    act=None,
                                    name='bbox_pred',
                                    param_attr=ParamAttr(
                                        name='bbox_pred_w',
                                        initializer=Normal(
                                            loc=0.0, scale=0.001)),
                                    bias_attr=ParamAttr(
                                        name='bbox_pred_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))
        return cls_score, bbox_pred

    def get_loss(self, roi_feat, labels_int32, bbox_targets,
                 bbox_inside_weights, bbox_outside_weights):
        """
        Get bbox_head loss.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            labels_int32(Variable): Class label of a RoI with shape [P, 1].
                P is the number of RoI.
            bbox_targets(Variable): Box label of a RoI with shape
                [P, 4 * class_nums].
            bbox_inside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.
            bbox_outside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.

        Return:
            Type: Dict
                loss_cls(Variable): bbox_head loss.
                loss_bbox(Variable): bbox_head loss.
        """

        cls_score, bbox_pred = self._get_output(roi_feat)

        labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=cls_score, label=labels_int64, numeric_stable_mode=True)
        loss_cls = fluid.layers.reduce_mean(loss_cls)
        loss_bbox = self.bbox_loss(
            x=bbox_pred,
            y=bbox_targets,
            inside_weight=bbox_inside_weights,
            outside_weight=bbox_outside_weights)
        loss_bbox = fluid.layers.reduce_mean(loss_bbox)
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}

    def get_prediction(self,
                       roi_feat,
                       rois,
                       im_info,
                       im_shape,
                       return_box_score=False):
        """
        Get prediction bounding box in test stage.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            rois (Variable): Output of generate_proposals in rpn head.
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists of im_height,
                im_width, im_scale.
            im_shape (Variable): Actual shape of original image with shape
                [B, 3]. B is the number of images, each element consists of
                original_height, original_width, 1

        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each
                row has 6 values: [label, confidence, xmin, ymin, xmax, ymax].
                N is the total number of prediction.
        """
        cls_score, bbox_pred = self._get_output(roi_feat)

        im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
        im_scale = fluid.layers.sequence_expand(im_scale, rois)
        boxes = rois / im_scale
        cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
        bbox_pred = fluid.layers.reshape(bbox_pred, (-1, self.num_classes, 4))
        decoded_box = self.box_coder(prior_box=boxes, target_box=bbox_pred)
        cliped_box = fluid.layers.box_clip(input=decoded_box, im_info=im_shape)
        if return_box_score:
            return {'bbox': cliped_box, 'score': cls_prob}
        pred_result = self.nms(bboxes=cliped_box, scores=cls_prob)
        return {'bbox': pred_result}
