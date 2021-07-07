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

import math

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant

from ppdet.modeling.ops import ConvNorm, DeformConvNorm
from ppdet.modeling.ops import MultiClassNMS

from ppdet.core.workspace import register

__all__ = ['FCOSHead']


@register
class FCOSHead(object):
    """
    FCOSHead
    Args:
        num_classes       (int): Number of classes
        fpn_stride       (list): The stride of each FPN Layer
        prior_prob      (float): Used to set the bias init for the class prediction layer
        num_convs         (int): The layer number in fcos head
        norm_type         (str): Normalization type, 'bn'/'sync_bn'/'affine_channel'
        fcos_loss      (object): Instance of 'FCOSLoss'
        norm_reg_targets (bool): Normalization the regression target if true
        centerness_on_reg(bool): The prediction of centerness on regression or clssification branch
        use_dcn_in_tower (bool): Ues deformable conv on FCOSHead if true
        nms            (object): Instance of 'MultiClassNMS'
    """
    __inject__ = ['fcos_loss', 'nms']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 num_convs=4,
                 norm_type="gn",
                 fcos_loss=None,
                 norm_reg_targets=False,
                 centerness_on_reg=False,
                 use_dcn_in_tower=False,
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__):
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride[::-1]
        self.prior_prob = prior_prob
        self.num_convs = num_convs
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.norm_type = norm_type
        self.fcos_loss = fcos_loss
        self.batch_size = 8
        self.nms = nms
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)

    def _fcos_head(self, features, fpn_stride, fpn_scale, is_training=False):
        """
        Args:
            features (Variables): feature map from FPN
            fpn_stride     (int): the stride of current feature map
            is_training   (bool): whether is train or test mode
        """
        subnet_blob_cls = features
        subnet_blob_reg = features
        in_channles = features.shape[1]
        if self.use_dcn_in_tower:
            conv_norm = DeformConvNorm
        else:
            conv_norm = ConvNorm
        for lvl in range(0, self.num_convs):
            conv_cls_name = 'fcos_head_cls_tower_conv_{}'.format(lvl)
            subnet_blob_cls = conv_norm(
                input=subnet_blob_cls,
                num_filters=in_channles,
                filter_size=3,
                stride=1,
                norm_type=self.norm_type,
                act='relu',
                initializer=Normal(
                    loc=0., scale=0.01),
                bias_attr=True,
                norm_name=conv_cls_name + "_norm",
                name=conv_cls_name)
            conv_reg_name = 'fcos_head_reg_tower_conv_{}'.format(lvl)
            subnet_blob_reg = conv_norm(
                input=subnet_blob_reg,
                num_filters=in_channles,
                filter_size=3,
                stride=1,
                norm_type=self.norm_type,
                act='relu',
                initializer=Normal(
                    loc=0., scale=0.01),
                bias_attr=True,
                norm_name=conv_reg_name + "_norm",
                name=conv_reg_name)
        conv_cls_name = "fcos_head_cls"
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        cls_logits = fluid.layers.conv2d(
            input=subnet_blob_cls,
            num_filters=self.num_classes,
            filter_size=3,
            stride=1,
            padding=1,
            param_attr=ParamAttr(
                name=conv_cls_name + "_weights",
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=conv_cls_name + "_bias",
                initializer=Constant(value=bias_init_value)),
            name=conv_cls_name)
        conv_reg_name = "fcos_head_reg"
        bbox_reg = fluid.layers.conv2d(
            input=subnet_blob_reg,
            num_filters=4,
            filter_size=3,
            stride=1,
            padding=1,
            param_attr=ParamAttr(
                name=conv_reg_name + "_weights",
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=conv_reg_name + "_bias", initializer=Constant(value=0)),
            name=conv_reg_name)
        bbox_reg = bbox_reg * fpn_scale
        if self.norm_reg_targets:
            bbox_reg = fluid.layers.relu(bbox_reg)
            if not is_training:
                bbox_reg = bbox_reg * fpn_stride
        else:
            bbox_reg = fluid.layers.exp(bbox_reg)

        conv_centerness_name = "fcos_head_centerness"
        if self.centerness_on_reg:
            subnet_blob_ctn = subnet_blob_reg
        else:
            subnet_blob_ctn = subnet_blob_cls
        centerness = fluid.layers.conv2d(
            input=subnet_blob_ctn,
            num_filters=1,
            filter_size=3,
            stride=1,
            padding=1,
            param_attr=ParamAttr(
                name=conv_centerness_name + "_weights",
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=conv_centerness_name + "_bias",
                initializer=Constant(value=0)),
            name=conv_centerness_name)
        return cls_logits, bbox_reg, centerness

    def _get_output(self, body_feats, is_training=False):
        """
        Args:
            body_feates (list): the list of fpn feature maps
            is_training (bool): whether is train or test mode
        Return:
            cls_logits (Variables): prediction for classification
            bboxes_reg (Variables): prediction for bounding box
            centerness (Variables): prediction for ceterness
        """
        cls_logits = []
        bboxes_reg = []
        centerness = []
        assert len(body_feats) == len(self.fpn_stride), \
            "The size of body_feats is not equal to size of fpn_stride"
        for fpn_name, fpn_stride in zip(body_feats, self.fpn_stride):
            features = body_feats[fpn_name]
            scale = fluid.layers.create_parameter(
                shape=[1, ],
                dtype="float32",
                name="%s_scale_on_reg" % fpn_name,
                default_initializer=fluid.initializer.Constant(1.))
            cls_pred, bbox_pred, ctn_pred = self._fcos_head(
                features, fpn_stride, scale, is_training=is_training)
            cls_logits.append(cls_pred)
            bboxes_reg.append(bbox_pred)
            centerness.append(ctn_pred)
        return cls_logits, bboxes_reg, centerness

    def _compute_locations(self, features):
        """
        Args:
            features (list): List of Variables for FPN feature maps
        Return:
            Anchor points for each feature map pixel
        """
        locations = []
        for lvl, fpn_name in enumerate(features):
            feature = features[fpn_name]
            shape_fm = fluid.layers.shape(feature)
            shape_fm.stop_gradient = True
            h = shape_fm[2]
            w = shape_fm[3]
            fpn_stride = self.fpn_stride[lvl]
            shift_x = fluid.layers.range(
                0, w * fpn_stride, fpn_stride, dtype='float32')
            shift_y = fluid.layers.range(
                0, h * fpn_stride, fpn_stride, dtype='float32')
            shift_x = fluid.layers.unsqueeze(shift_x, axes=[0])
            shift_y = fluid.layers.unsqueeze(shift_y, axes=[1])
            shift_x = fluid.layers.expand_as(
                shift_x, target_tensor=feature[0, 0, :, :])
            shift_y = fluid.layers.expand_as(
                shift_y, target_tensor=feature[0, 0, :, :])
            shift_x.stop_gradient = True
            shift_y.stop_gradient = True
            shift_x = fluid.layers.reshape(shift_x, shape=[-1])
            shift_y = fluid.layers.reshape(shift_y, shape=[-1])
            location = fluid.layers.stack(
                [shift_x, shift_y], axis=-1) + fpn_stride // 2
            location.stop_gradient = True
            locations.append(location)
        return locations

    def __merge_hw(self, input, ch_type="channel_first"):
        """
        Args:
            input (Variables): Feature map whose H and W will be merged into one dimension
            ch_type     (str): channel_first / channel_last
        Return:
            new_shape (Variables): The new shape after h and w merged into one dimension
        """
        shape_ = fluid.layers.shape(input)
        bs = shape_[0]
        ch = shape_[1]
        hi = shape_[2]
        wi = shape_[3]
        img_size = hi * wi
        img_size.stop_gradient = True
        if ch_type == "channel_first":
            new_shape = fluid.layers.concat([bs, ch, img_size])
        elif ch_type == "channel_last":
            new_shape = fluid.layers.concat([bs, img_size, ch])
        else:
            raise KeyError("Wrong ch_type %s" % ch_type)
        new_shape.stop_gradient = True
        return new_shape

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn,
                                 im_info):
        """
        Args:
            locations (Variables): anchor points for current layer
            box_cls   (Variables): categories prediction
            box_reg   (Variables): bounding box prediction
            box_ctn   (Variables): centerness prediction
            im_info   (Variables): [h, w, scale] for input images
        Return:
            box_cls_ch_last  (Variables): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Variables): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        act_shape_cls = self.__merge_hw(box_cls)
        box_cls_ch_last = fluid.layers.reshape(
            x=box_cls,
            shape=[self.batch_size, self.num_classes, -1],
            actual_shape=act_shape_cls)
        box_cls_ch_last = fluid.layers.sigmoid(box_cls_ch_last)
        act_shape_reg = self.__merge_hw(box_reg, "channel_last")
        box_reg_ch_last = fluid.layers.transpose(box_reg, perm=[0, 2, 3, 1])
        box_reg_ch_last = fluid.layers.reshape(
            x=box_reg_ch_last,
            shape=[self.batch_size, -1, 4],
            actual_shape=act_shape_reg)
        act_shape_ctn = self.__merge_hw(box_ctn)
        box_ctn_ch_last = fluid.layers.reshape(
            x=box_ctn,
            shape=[self.batch_size, 1, -1],
            actual_shape=act_shape_ctn)
        box_ctn_ch_last = fluid.layers.sigmoid(box_ctn_ch_last)

        box_reg_decoding = fluid.layers.stack(
            [
                locations[:, 0] - box_reg_ch_last[:, :, 0],
                locations[:, 1] - box_reg_ch_last[:, :, 1],
                locations[:, 0] + box_reg_ch_last[:, :, 2],
                locations[:, 1] + box_reg_ch_last[:, :, 3]
            ],
            axis=1)
        box_reg_decoding = fluid.layers.transpose(
            box_reg_decoding, perm=[0, 2, 1])
        # recover the location to original image
        im_scale = im_info[:, 2]
        box_reg_decoding = box_reg_decoding / im_scale
        box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last
        return box_cls_ch_last, box_reg_decoding

    def _post_processing(self, locations, cls_logits, bboxes_reg, centerness,
                         im_info):
        """
        Args:
            locations   (list): List of Variables composed by center of each anchor point
            cls_logits  (list): List of Variables for class prediction
            bboxes_reg  (list): List of Variables for bounding box prediction
            centerness  (list): List of Variables for centerness prediction
            im_info(Variables): [h, w, scale] for input images
        Return:
            pred (LoDTensor): predicted bounding box after nms,
                the shape is n x 6, last dimension is [label, score, xmin, ymin, xmax, ymax]
        """
        pred_boxes_ = []
        pred_scores_ = []
        for _, (
                pts, cls, box, ctn
        ) in enumerate(zip(locations, cls_logits, bboxes_reg, centerness)):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, im_info)
            pred_boxes_.append(pred_boxes_lvl)
            pred_scores_.append(pred_scores_lvl)
        pred_boxes = fluid.layers.concat(pred_boxes_, axis=1)
        pred_scores = fluid.layers.concat(pred_scores_, axis=2)
        pred = self.nms(pred_boxes, pred_scores)
        return pred

    def get_loss(self, input, tag_labels, tag_bboxes, tag_centerness):
        """
        Calculate the loss for FCOS
        Args:
            input           (list): List of Variables for feature maps from FPN layers
            tag_labels     (Variables): category targets for each anchor point
            tag_bboxes     (Variables): bounding boxes  targets for positive samples
            tag_centerness (Variables): centerness targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
                regression loss and centerness regression loss
        """
        cls_logits, bboxes_reg, centerness = self._get_output(
            input, is_training=True)
        loss = self.fcos_loss(cls_logits, bboxes_reg, centerness, tag_labels,
                              tag_bboxes, tag_centerness)
        return loss

    def get_prediction(self, input, im_info):
        """
        Decode the prediction
        Args:
            input           (list): List of Variables for feature maps from FPN layers
            im_info(Variables): [h, w, scale] for input images
        Return:
            the bounding box prediction
        """
        cls_logits, bboxes_reg, centerness = self._get_output(
            input, is_training=False)
        locations = self._compute_locations(input)
        pred = self._post_processing(locations, cls_logits, bboxes_reg,
                                     centerness, im_info)
        return {"bbox": pred}
