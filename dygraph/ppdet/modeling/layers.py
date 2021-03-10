#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import six
import numpy as np
from numbers import Integral

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle import to_tensor
from paddle.nn import Conv2D, BatchNorm2D, GroupNorm
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, serializable
from ppdet.modeling.bbox_utils import delta2bbox
from . import ops

from paddle.vision.ops import DeformConv2D


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 name=None):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = ParamAttr(
                initializer=Constant(0.),
                name='{}._conv_offset.bias'.format(name))
        else:
            offset_bias_attr = ParamAttr(
                initializer=Constant(0.),
                learning_rate=lr_scale,
                regularizer=regularizer,
                name='{}._conv_offset.bias'.format(name))
        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(
                initializer=Constant(0.0),
                name='{}._conv_offset.weight'.format(name)),
            bias_attr=offset_bias_attr)

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = ParamAttr(
                name=name + "_bias",
                initializer=Constant(value=0),
                regularizer=L2Decay(0.),
                learning_rate=2.)
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=dcn_bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 norm_name=None,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=Normal(
                     mean=0., std=0.01),
                 name=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn']

        if bias_on:
            bias_attr = ParamAttr(
                name=name + "_bias",
                initializer=Constant(value=0.),
                learning_rate=lr_scale)
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=1,
                weight_attr=ParamAttr(
                    name=name + "_weight",
                    initializer=initializer,
                    learning_rate=1.),
                bias_attr=bias_attr)
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=1,
                weight_attr=ParamAttr(
                    name=name + "_weight",
                    initializer=initializer,
                    learning_rate=1.),
                bias_attr=True,
                lr_scale=2.,
                regularizer=L2Decay(norm_decay),
                name=name)

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            name=norm_name + "_scale",
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        bias_attr = ParamAttr(
            name=norm_name + "_offset",
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2D(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


@register
@serializable
class AnchorGeneratorRPN(object):
    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 stride=[16.0, 16.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 anchor_start_size=None):
        super(AnchorGeneratorRPN, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.variance = variance
        self.anchor_start_size = anchor_start_size

    def __call__(self, input, level=None):
        anchor_sizes = self.anchor_sizes if (
            level is None or self.anchor_start_size is None) else (
                self.anchor_start_size * 2**level)
        stride = self.stride if (
            level is None or self.anchor_start_size is None) else (
                self.stride[0] * (2.**level), self.stride[1] * (2.**level))
        anchor, var = ops.anchor_generator(
            input=input,
            anchor_sizes=anchor_sizes,
            aspect_ratios=self.aspect_ratios,
            stride=stride,
            variance=self.variance)
        return anchor, var


@register
@serializable
class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=[8, 16, 32, 64, 100, 300],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
                 min_ratio=15,
                 max_ratio=90,
                 base_size=300,
                 min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
                 max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
                 offset=0.5,
                 flip=True,
                 clip=False,
                 min_max_aspect_ratios_order=False):
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_size = base_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order

        if self.min_sizes == [] and self.max_sizes == []:
            num_layer = len(aspect_ratios)
            step = int(
                math.floor(((self.max_ratio - self.min_ratio)) / (num_layer - 2
                                                                  )))
            for ratio in six.moves.range(self.min_ratio, self.max_ratio + 1,
                                         step):
                self.min_sizes.append(self.base_size * ratio / 100.)
                self.max_sizes.append(self.base_size * (ratio + step) / 100.)
            self.min_sizes = [self.base_size * .10] + self.min_sizes
            self.max_sizes = [self.base_size * .20] + self.max_sizes

        self.num_priors = []
        for aspect_ratio, min_size, max_size in zip(
                aspect_ratios, self.min_sizes, self.max_sizes):
            if isinstance(min_size, (list, tuple)):
                self.num_priors.append(
                    len(_to_list(min_size)) + len(_to_list(max_size)))
            else:
                self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(
                    _to_list(min_size)) + len(_to_list(max_size)))

    def __call__(self, inputs, image):
        boxes = []
        for input, min_size, max_size, aspect_ratio, step in zip(
                inputs, self.min_sizes, self.max_sizes, self.aspect_ratios,
                self.steps):
            box, _ = ops.prior_box(
                input=input,
                image=image,
                min_sizes=_to_list(min_size),
                max_sizes=_to_list(max_size),
                aspect_ratios=aspect_ratio,
                flip=self.flip,
                clip=self.clip,
                steps=[step, step],
                offset=self.offset,
                min_max_aspect_ratios_order=self.min_max_aspect_ratios_order)
            boxes.append(paddle.reshape(box, [-1, 4]))
        return boxes


@register
@serializable
class RCNNBox(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 prior_box_var=[10., 10., 5., 5.],
                 code_type="decode_center_size",
                 box_normalized=False,
                 num_classes=80):
        super(RCNNBox, self).__init__()
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.num_classes = num_classes

    def __call__(self, bbox_head_out, rois, im_shape, scale_factor):
        bbox_pred, cls_prob = bbox_head_out
        roi, rois_num = rois
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        scale_list = []
        origin_shape_list = []
        for idx, roi_per_im in enumerate(roi):
            rois_num_per_im = rois_num[idx]
            expand_im_shape = paddle.expand(im_shape[idx, :],
                                            [rois_num_per_im, 2])
            origin_shape_list.append(expand_im_shape)

        origin_shape = paddle.concat(origin_shape_list)

        # bbox_pred.shape: [N, C*4]
        # C=num_classes in faster/mask rcnn(bbox_head), C=1 in cascade rcnn(cascade_head)
        bbox = paddle.concat(roi)
        if bbox.shape[0] == 0:
            bbox = paddle.zeros([0, bbox_pred.shape[1]], dtype='float32')
        else:
            bbox = delta2bbox(bbox_pred, bbox, self.prior_box_var)
        scores = cls_prob[:, :-1]

        # bbox.shape: [N, C, 4]
        # bbox.shape[1] must be equal to scores.shape[1]
        bbox_num_class = bbox.shape[1]
        if bbox_num_class == 1:
            bbox = paddle.tile(bbox, [1, self.num_classes, 1])

        origin_h = paddle.unsqueeze(origin_shape[:, 0], axis=1)
        origin_w = paddle.unsqueeze(origin_shape[:, 1], axis=1)
        zeros = paddle.zeros_like(origin_h)
        x1 = paddle.maximum(paddle.minimum(bbox[:, :, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(bbox[:, :, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(bbox[:, :, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(bbox[:, :, 3], origin_h), zeros)
        bbox = paddle.stack([x1, y1, x2, y2], axis=-1)
        bboxes = (bbox, rois_num)
        return bboxes, scores


@register
@serializable
class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=False,
                 nms_eta=1.0,
                 return_rois_num=True):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_rois_num = return_rois_num

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape 
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,] 
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1. 
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        if background_label > -1:
            kwargs.update({'background_label': background_label})
        return ops.multiclass_nms(bboxes, score, **kwargs)


@register
@serializable
class MatrixNMS(object):
    __append_doc__ = True

    def __init__(self,
                 score_threshold=.05,
                 post_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 use_gaussian=False,
                 gaussian_sigma=2.,
                 normalized=False,
                 background_label=0):
        super(MatrixNMS, self).__init__()
        self.score_threshold = score_threshold
        self.post_threshold = post_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.normalized = normalized
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        self.background_label = background_label

    def __call__(self, bbox, score, *args):
        return ops.matrix_nms(
            bboxes=bbox,
            scores=score,
            score_threshold=self.score_threshold,
            post_threshold=self.post_threshold,
            nms_top_k=self.nms_top_k,
            keep_top_k=self.keep_top_k,
            use_gaussian=self.use_gaussian,
            gaussian_sigma=self.gaussian_sigma,
            background_label=self.background_label,
            normalized=self.normalized)


@register
@serializable
class YOLOBox(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 conf_thresh=0.005,
                 downsample_ratio=32,
                 clip_bbox=True,
                 scale_x_y=1.):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.downsample_ratio = downsample_ratio
        self.clip_bbox = clip_bbox
        self.scale_x_y = scale_x_y

    def __call__(self,
                 yolo_head_out,
                 anchors,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes_list = []
        scores_list = []
        origin_shape = im_shape / scale_factor
        origin_shape = paddle.cast(origin_shape, 'int32')
        for i, head_out in enumerate(yolo_head_out):
            boxes, scores = ops.yolo_box(head_out, origin_shape, anchors[i],
                                         self.num_classes, self.conf_thresh,
                                         self.downsample_ratio // 2**i,
                                         self.clip_bbox, self.scale_x_y)
            boxes_list.append(boxes)
            scores_list.append(paddle.transpose(scores, perm=[0, 2, 1]))
        yolo_boxes = paddle.concat(boxes_list, axis=1)
        yolo_scores = paddle.concat(scores_list, axis=2)
        return yolo_boxes, yolo_scores


@register
@serializable
class SSDBox(object):
    def __init__(self, is_normalized=True):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)

    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes, scores = preds
        outputs = []
        for box, score, prior_box in zip(boxes, scores, prior_boxes):
            pb_w = prior_box[:, 2] - prior_box[:, 0] + self.norm_delta
            pb_h = prior_box[:, 3] - prior_box[:, 1] + self.norm_delta
            pb_x = prior_box[:, 0] + pb_w * 0.5
            pb_y = prior_box[:, 1] + pb_h * 0.5
            out_x = pb_x + box[:, :, 0] * pb_w * 0.1
            out_y = pb_y + box[:, :, 1] * pb_h * 0.1
            out_w = paddle.exp(box[:, :, 2] * 0.2) * pb_w
            out_h = paddle.exp(box[:, :, 3] * 0.2) * pb_h

            if self.is_normalized:
                h = paddle.unsqueeze(
                    im_shape[:, 0] / scale_factor[:, 0], axis=-1)
                w = paddle.unsqueeze(
                    im_shape[:, 1] / scale_factor[:, 1], axis=-1)
                output = paddle.stack(
                    [(out_x - out_w / 2.) * w, (out_y - out_h / 2.) * h,
                     (out_x + out_w / 2.) * w, (out_y + out_h / 2.) * h],
                    axis=-1)
            else:
                output = paddle.stack(
                    [
                        out_x - out_w / 2., out_y - out_h / 2.,
                        out_x + out_w / 2. - 1., out_y + out_h / 2. - 1.
                    ],
                    axis=-1)
            outputs.append(output)
        boxes = paddle.concat(outputs, axis=1)

        scores = F.softmax(paddle.concat(scores, axis=1))
        scores = paddle.transpose(scores, [0, 2, 1])

        return boxes, scores


@register
@serializable
class AnchorGrid(object):
    """Generate anchor grid

    Args:
        image_size (int or list): input image size, may be a single integer or
            list of [h, w]. Default: 512
        min_level (int): min level of the feature pyramid. Default: 3
        max_level (int): max level of the feature pyramid. Default: 7
        anchor_base_scale: base anchor scale. Default: 4
        num_scales: number of anchor scales. Default: 3
        aspect_ratios: aspect ratios. default: [[1, 1], [1.4, 0.7], [0.7, 1.4]]
    """

    def __init__(self,
                 image_size=512,
                 min_level=3,
                 max_level=7,
                 anchor_base_scale=4,
                 num_scales=3,
                 aspect_ratios=[[1, 1], [1.4, 0.7], [0.7, 1.4]]):
        super(AnchorGrid, self).__init__()
        if isinstance(image_size, Integral):
            self.image_size = [image_size, image_size]
        else:
            self.image_size = image_size
        for dim in self.image_size:
            assert dim % 2 ** max_level == 0, \
                "image size should be multiple of the max level stride"
        self.min_level = min_level
        self.max_level = max_level
        self.anchor_base_scale = anchor_base_scale
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios

    @property
    def base_cell(self):
        if not hasattr(self, '_base_cell'):
            self._base_cell = self.make_cell()
        return self._base_cell

    def make_cell(self):
        scales = [2**(i / self.num_scales) for i in range(self.num_scales)]
        scales = np.array(scales)
        ratios = np.array(self.aspect_ratios)
        ws = np.outer(scales, ratios[:, 0]).reshape(-1, 1)
        hs = np.outer(scales, ratios[:, 1]).reshape(-1, 1)
        anchors = np.hstack((-0.5 * ws, -0.5 * hs, 0.5 * ws, 0.5 * hs))
        return anchors

    def make_grid(self, stride):
        cell = self.base_cell * stride * self.anchor_base_scale
        x_steps = np.arange(stride // 2, self.image_size[1], stride)
        y_steps = np.arange(stride // 2, self.image_size[0], stride)
        offset_x, offset_y = np.meshgrid(x_steps, y_steps)
        offset_x = offset_x.flatten()
        offset_y = offset_y.flatten()
        offsets = np.stack((offset_x, offset_y, offset_x, offset_y), axis=-1)
        offsets = offsets[:, np.newaxis, :]
        return (cell + offsets).reshape(-1, 4)

    def generate(self):
        return [
            self.make_grid(2**l)
            for l in range(self.min_level, self.max_level + 1)
        ]

    def __call__(self):
        if not hasattr(self, '_anchor_vars'):
            anchor_vars = []
            helper = LayerHelper('anchor_grid')
            for idx, l in enumerate(range(self.min_level, self.max_level + 1)):
                stride = 2**l
                anchors = self.make_grid(stride)
                var = helper.create_parameter(
                    attr=ParamAttr(name='anchors_{}'.format(idx)),
                    shape=anchors.shape,
                    dtype='float32',
                    stop_gradient=True,
                    default_initializer=NumpyArrayInitializer(anchors))
                anchor_vars.append(var)
                var.persistable = True
            self._anchor_vars = anchor_vars

        return self._anchor_vars


@register
@serializable
class FCOSBox(object):
    __shared__ = ['num_classes', 'batch_size']

    def __init__(self, num_classes=80, batch_size=1):
        super(FCOSBox, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _merge_hw(self, inputs, ch_type="channel_first"):
        """
        Args:
            inputs (Variables): Feature map whose H and W will be merged into one dimension
            ch_type     (str): channel_first / channel_last
        Return:
            new_shape (Variables): The new shape after h and w merged into one dimension
        """
        shape_ = paddle.shape(inputs)
        bs, ch, hi, wi = shape_[0], shape_[1], shape_[2], shape_[3]
        img_size = hi * wi
        img_size.stop_gradient = True
        if ch_type == "channel_first":
            new_shape = paddle.concat([bs, ch, img_size])
        elif ch_type == "channel_last":
            new_shape = paddle.concat([bs, img_size, ch])
        else:
            raise KeyError("Wrong ch_type %s" % ch_type)
        new_shape.stop_gradient = True
        return new_shape

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn,
                                 scale_factor):
        """
        Args:
            locations (Variables): anchor points for current layer, [H*W, 2]
            box_cls   (Variables): categories prediction, [N, C, H, W],  C is the number of classes 
            box_reg   (Variables): bounding box prediction, [N, 4, H, W]
            box_ctn   (Variables): centerness prediction, [N, 1, H, W]
            scale_factor   (Variables): [h_scale, w_scale] for input images
        Return:
            box_cls_ch_last  (Variables): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Variables): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        act_shape_cls = self._merge_hw(box_cls)
        box_cls_ch_last = paddle.reshape(x=box_cls, shape=act_shape_cls)
        box_cls_ch_last = F.sigmoid(box_cls_ch_last)

        act_shape_reg = self._merge_hw(box_reg)
        box_reg_ch_last = paddle.reshape(x=box_reg, shape=act_shape_reg)
        box_reg_ch_last = paddle.transpose(box_reg_ch_last, perm=[0, 2, 1])
        box_reg_decoding = paddle.stack(
            [
                locations[:, 0] - box_reg_ch_last[:, :, 0],
                locations[:, 1] - box_reg_ch_last[:, :, 1],
                locations[:, 0] + box_reg_ch_last[:, :, 2],
                locations[:, 1] + box_reg_ch_last[:, :, 3]
            ],
            axis=1)
        box_reg_decoding = paddle.transpose(box_reg_decoding, perm=[0, 2, 1])

        act_shape_ctn = self._merge_hw(box_ctn)
        box_ctn_ch_last = paddle.reshape(x=box_ctn, shape=act_shape_ctn)
        box_ctn_ch_last = F.sigmoid(box_ctn_ch_last)

        # recover the location to original image
        im_scale = paddle.concat([scale_factor, scale_factor], axis=1)
        box_reg_decoding = box_reg_decoding / im_scale
        box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last
        return box_cls_ch_last, box_reg_decoding

    def __call__(self, locations, cls_logits, bboxes_reg, centerness,
                 scale_factor):
        pred_boxes_ = []
        pred_scores_ = []
        for pts, cls, box, ctn in zip(locations, cls_logits, bboxes_reg,
                                      centerness):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, scale_factor)
            pred_boxes_.append(pred_boxes_lvl)
            pred_scores_.append(pred_scores_lvl)
        pred_boxes = paddle.concat(pred_boxes_, axis=1)
        pred_scores = paddle.concat(pred_scores_, axis=2)
        return pred_boxes, pred_scores


@register
class TTFBox(object):
    __shared__ = ['down_ratio']

    def __init__(self, max_per_img=100, score_thresh=0.01, down_ratio=4):
        super(TTFBox, self).__init__()
        self.max_per_img = max_per_img
        self.score_thresh = score_thresh
        self.down_ratio = down_ratio

    def _simple_nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = paddle.cast(hmax == heat, 'float32')
        return heat * keep

    def _topk(self, scores):
        k = self.max_per_img
        shape_fm = paddle.shape(scores)
        shape_fm.stop_gradient = True
        cat, height, width = shape_fm[1], shape_fm[2], shape_fm[3]
        # batch size is 1
        scores_r = paddle.reshape(scores, [cat, -1])
        topk_scores, topk_inds = paddle.topk(scores_r, k)
        topk_scores, topk_inds = paddle.topk(scores_r, k)
        topk_ys = topk_inds // width
        topk_xs = topk_inds % width

        topk_score_r = paddle.reshape(topk_scores, [-1])
        topk_score, topk_ind = paddle.topk(topk_score_r, k)
        k_t = paddle.full(paddle.shape(topk_ind), k, dtype='int64')
        topk_clses = paddle.cast(paddle.floor_divide(topk_ind, k_t), 'float32')

        topk_inds = paddle.reshape(topk_inds, [-1])
        topk_ys = paddle.reshape(topk_ys, [-1, 1])
        topk_xs = paddle.reshape(topk_xs, [-1, 1])
        topk_inds = paddle.gather(topk_inds, topk_ind)
        topk_ys = paddle.gather(topk_ys, topk_ind)
        topk_xs = paddle.gather(topk_xs, topk_ind)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def __call__(self, hm, wh, im_shape, scale_factor):
        heatmap = F.sigmoid(hm)
        heat = self._simple_nms(heatmap)
        scores, inds, clses, ys, xs = self._topk(heat)
        ys = paddle.cast(ys, 'float32') * self.down_ratio
        xs = paddle.cast(xs, 'float32') * self.down_ratio
        scores = paddle.tensor.unsqueeze(scores, [1])
        clses = paddle.tensor.unsqueeze(clses, [1])

        wh_t = paddle.transpose(wh, [0, 2, 3, 1])
        wh = paddle.reshape(wh_t, [-1, paddle.shape(wh_t)[-1]])
        wh = paddle.gather(wh, inds)

        x1 = xs - wh[:, 0:1]
        y1 = ys - wh[:, 1:2]
        x2 = xs + wh[:, 2:3]
        y2 = ys + wh[:, 3:4]

        bboxes = paddle.concat([x1, y1, x2, y2], axis=1)

        scale_y = scale_factor[:, 0:1]
        scale_x = scale_factor[:, 1:2]
        scale_expand = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y], axis=1)
        boxes_shape = paddle.shape(bboxes)
        boxes_shape.stop_gradient = True
        scale_expand = paddle.expand(scale_expand, shape=boxes_shape)
        bboxes = paddle.divide(bboxes, scale_expand)
        results = paddle.concat([clses, scores, bboxes], axis=1)
        # hack: append result with cls=-1 and score=1. to avoid all scores
        # are less than score_thresh which may cause error in gather.
        fill_r = paddle.to_tensor(np.array([[-1, 1, 0, 0, 0, 0]]))
        fill_r = paddle.cast(fill_r, results.dtype)
        results = paddle.concat([results, fill_r])
        scores = results[:, 1]
        valid_ind = paddle.nonzero(scores > self.score_thresh)
        results = paddle.gather(results, valid_ind)
        return results, paddle.shape(results)[0:1]


@register
@serializable
class MaskMatrixNMS(object):
    """
    Matrix NMS for multi-class masks.
    Args:
        update_threshold (float): Updated threshold of categroy score in second time.
        pre_nms_top_n (int): Number of total instance to be kept per image before NMS
        post_nms_top_n (int): Number of total instance to be kept per image after NMS.
        kernel (str):  'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
    Input:
        seg_preds (Variable): shape (n, h, w), segmentation feature maps
        seg_masks (Variable): shape (n, h, w), segmentation feature maps
        cate_labels (Variable): shape (n), mask labels in descending order
        cate_scores (Variable): shape (n), mask scores in descending order
        sum_masks (Variable): a float tensor of the sum of seg_masks
    Returns:
        Variable: cate_scores, tensors of shape (n)
    """

    def __init__(self,
                 update_threshold=0.05,
                 pre_nms_top_n=500,
                 post_nms_top_n=100,
                 kernel='gaussian',
                 sigma=2.0):
        super(MaskMatrixNMS, self).__init__()
        self.update_threshold = update_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.kernel = kernel
        self.sigma = sigma

    def _sort_score(self, scores, top_num):
        if paddle.shape(scores)[0] > top_num:
            return paddle.topk(scores, top_num)[1]
        else:
            return paddle.argsort(scores, descending=True)

    def __call__(self,
                 seg_preds,
                 seg_masks,
                 cate_labels,
                 cate_scores,
                 sum_masks=None):
        # sort and keep top nms_pre
        sort_inds = self._sort_score(cate_scores, self.pre_nms_top_n)
        seg_masks = paddle.gather(seg_masks, index=sort_inds)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        sum_masks = paddle.gather(sum_masks, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)

        seg_masks = paddle.flatten(seg_masks, start_axis=1, stop_axis=-1)
        # inter.
        inter_matrix = paddle.mm(seg_masks, paddle.transpose(seg_masks, [1, 0]))
        n_samples = paddle.shape(cate_labels)
        # union.
        sum_masks_x = paddle.expand(sum_masks, shape=[n_samples, n_samples])
        # iou.
        iou_matrix = (inter_matrix / (
            sum_masks_x + paddle.transpose(sum_masks_x, [1, 0]) - inter_matrix))
        iou_matrix = paddle.triu(iou_matrix, diagonal=1)
        # label_specific matrix.
        cate_labels_x = paddle.expand(cate_labels, shape=[n_samples, n_samples])
        label_matrix = paddle.cast(
            (cate_labels_x == paddle.transpose(cate_labels_x, [1, 0])),
            'float32')
        label_matrix = paddle.triu(label_matrix, diagonal=1)

        # IoU compensation
        compensate_iou = paddle.max((iou_matrix * label_matrix), axis=0)
        compensate_iou = paddle.expand(
            compensate_iou, shape=[n_samples, n_samples])
        compensate_iou = paddle.transpose(compensate_iou, [1, 0])

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'gaussian':
            decay_matrix = paddle.exp(-1 * self.sigma * (decay_iou**2))
            compensate_matrix = paddle.exp(-1 * self.sigma *
                                           (compensate_iou**2))
            decay_coefficient = paddle.min(decay_matrix / compensate_matrix,
                                           axis=0)
        elif self.kernel == 'linear':
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient = paddle.min(decay_matrix, axis=0)
        else:
            raise NotImplementedError

        # update the score.
        cate_scores = cate_scores * decay_coefficient
        y = paddle.zeros(shape=paddle.shape(cate_scores), dtype='float32')
        keep = paddle.where(cate_scores >= self.update_threshold, cate_scores,
                            y)
        keep = paddle.nonzero(keep)
        keep = paddle.squeeze(keep, axis=[1])
        # Prevent empty and increase fake data
        keep = paddle.concat(
            [keep, paddle.cast(paddle.shape(cate_scores)[0] - 1, 'int64')])

        seg_preds = paddle.gather(seg_preds, index=keep)
        cate_scores = paddle.gather(cate_scores, index=keep)
        cate_labels = paddle.gather(cate_labels, index=keep)

        # sort and keep top_k
        sort_inds = self._sort_score(cate_scores, self.post_nms_top_n)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)
        return seg_preds, cate_scores, cate_labels
