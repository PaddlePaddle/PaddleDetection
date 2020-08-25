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

import numpy as np

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.modeling.ops import MultiClassNMS, MultiClassSoftNMS, MatrixNMS
from ppdet.modeling.losses.yolo_loss import YOLOv3Loss
from ppdet.core.workspace import register
from ppdet.modeling.ops import DropBlock
from .iou_aware import get_iou_aware_score
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from ppdet.utils.check import check_version

__all__ = ['YOLOv3Head', 'YOLOv4Head']


@register
class YOLOv3Head(object):
    """
    Head block for YOLOv3 network

    Args:
        conv_block_num (int): number of conv block in each detection block
        norm_decay (float): weight decay for normalization layer weights
        num_classes (int): number of output classes
        anchors (list): anchors
        anchor_masks (list): anchor masks
        nms (object): an instance of `MultiClassNMS`
    """
    __inject__ = ['yolo_loss', 'nms']
    __shared__ = ['num_classes', 'weight_prefix_name']

    def __init__(self,
                 conv_block_num=2,
                 norm_decay=0.,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 drop_block=False,
                 coord_conv=False,
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 block_size=3,
                 keep_prob=0.9,
                 yolo_loss="YOLOv3Loss",
                 spp=False,
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__,
                 weight_prefix_name='',
                 downsample=[32, 16, 8],
                 scale_x_y=1.0,
                 clip_bbox=True):
        check_version("1.8.4")
        self.conv_block_num = conv_block_num
        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.yolo_loss = yolo_loss
        self.nms = nms
        self.prefix_name = weight_prefix_name
        self.drop_block = drop_block
        self.iou_aware = iou_aware
        self.coord_conv = coord_conv
        self.iou_aware_factor = iou_aware_factor
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.use_spp = spp
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.clip_bbox = clip_bbox

    def _create_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_global_var(
            shape=numpy_array.shape, value=0., dtype=numpy_array.dtype)
        fluid.layers.assign(numpy_array, paddle_array)
        return paddle_array

    def _add_coord(self, input, is_test=True):
        if not self.coord_conv:
            return input

        # NOTE: here is used for exporting model for TensorRT inference,
        #       only support batch_size=1 for input shape should be fixed,
        #       and we create tensor with fixed shape from numpy array
        if is_test and input.shape[2] > 0 and input.shape[3] > 0:
            batch_size = 1
            grid_x = int(input.shape[3])
            grid_y = int(input.shape[2])
            idx_i = np.array(
                [[i / (grid_x - 1) * 2.0 - 1 for i in range(grid_x)]],
                dtype='float32')
            gi_np = np.repeat(idx_i, grid_y, axis=0)
            gi_np = np.reshape(gi_np, newshape=[1, 1, grid_y, grid_x])
            gi_np = np.tile(gi_np, reps=[batch_size, 1, 1, 1])

            x_range = self._create_tensor_from_numpy(gi_np.astype(np.float32))
            x_range.stop_gradient = True
            y_range = self._create_tensor_from_numpy(
                gi_np.transpose([0, 1, 3, 2]).astype(np.float32))
            y_range.stop_gradient = True

        # NOTE: in training mode, H and W is variable for random shape,
        #       implement add_coord with shape as Variable
        else:
            input_shape = fluid.layers.shape(input)
            b = input_shape[0]
            h = input_shape[2]
            w = input_shape[3]

            x_range = fluid.layers.range(0, w, 1, 'float32') / ((w - 1.) / 2.)
            x_range = x_range - 1.
            x_range = fluid.layers.unsqueeze(x_range, [0, 1, 2])
            x_range = fluid.layers.expand(x_range, [b, 1, h, 1])
            x_range.stop_gradient = True
            y_range = fluid.layers.transpose(x_range, [0, 1, 3, 2])
            y_range.stop_gradient = True

        return fluid.layers.concat([input, x_range, y_range], axis=1)

    def _conv_bn(self,
                 input,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='leaky',
                 name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.offset')
        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _spp_module(self, input, name=""):
        output1 = input
        output2 = fluid.layers.pool2d(
            input=output1,
            pool_size=5,
            pool_stride=1,
            pool_padding=2,
            ceil_mode=False,
            pool_type='max')
        output3 = fluid.layers.pool2d(
            input=output1,
            pool_size=9,
            pool_stride=1,
            pool_padding=4,
            ceil_mode=False,
            pool_type='max')
        output4 = fluid.layers.pool2d(
            input=output1,
            pool_size=13,
            pool_stride=1,
            pool_padding=6,
            ceil_mode=False,
            pool_type='max')
        output = fluid.layers.concat(
            input=[output1, output2, output3, output4], axis=1)
        return output

    def _detection_block(self,
                         input,
                         channel,
                         conv_block_num=2,
                         is_first=False,
                         is_test=True,
                         name=None):
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2 in detection block {}" \
            .format(channel, name)

        conv = input
        for j in range(conv_block_num):
            conv = self._add_coord(conv, is_test=is_test)
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                name='{}.{}.0'.format(name, j))
            if self.use_spp and is_first and j == 1:
                conv = self._spp_module(conv, name="spp")
                conv = self._conv_bn(
                    conv,
                    512,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    name='{}.{}.spp.conv'.format(name, j))
            conv = self._conv_bn(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                name='{}.{}.1'.format(name, j))
            if self.drop_block and j == 0 and not is_first:
                conv = DropBlock(
                    conv,
                    block_size=self.block_size,
                    keep_prob=self.keep_prob,
                    is_test=is_test)

        if self.drop_block and is_first:
            conv = DropBlock(
                conv,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
        conv = self._add_coord(conv, is_test=is_test)
        route = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            name='{}.2'.format(name))
        new_route = self._add_coord(route, is_test=is_test)
        tip = self._conv_bn(
            new_route,
            channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            name='{}.tip'.format(name))
        return route, tip

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out

    def _parse_anchors(self, anchors):
        """
        Check ANCHORS/ANCHOR_MASKS in config and parse mask_anchors

        """
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _get_outputs(self, input, is_train=True):
        """
        Get YOLOv3 head output

        Args:
            input (list): List of Variables, output of backbone stages
            is_train (bool): whether in train or test mode

        Returns:
            outputs (list): Variables of each output layer
        """

        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = input[-1:-out_layer_num - 1:-1]

        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self._detection_block(
                block,
                channel=64 * (2**out_layer_num) // (2**i),
                is_first=i == 0,
                is_test=(not is_train),
                conv_block_num=self.conv_block_num,
                name=self.prefix_name + "yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            with fluid.name_scope('yolo_output'):
                block_out = fluid.layers.conv2d(
                    input=tip,
                    num_filters=num_filters,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        name=self.prefix_name +
                        "yolo_output.{}.conv.weights".format(i)),
                    bias_attr=ParamAttr(
                        regularizer=L2Decay(0.),
                        name=self.prefix_name +
                        "yolo_output.{}.conv.bias".format(i)))
                outputs.append(block_out)

            if i < len(blocks) - 1:
                # do not perform upsample in the last detection_block
                route = self._conv_bn(
                    input=route,
                    ch_out=256 // (2**i),
                    filter_size=1,
                    stride=1,
                    padding=0,
                    name=self.prefix_name + "yolo_transition.{}".format(i))
                # upsample
                route = self._upsample(route)

        return outputs

    def get_loss(self, input, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of YOLOv3.

        Args:
            input (list): List of Variables, output of backbone stages
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of YOLOv3 network.

        """
        outputs = self._get_outputs(input, is_train=True)

        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes,
                              self.prefix_name)

    def get_prediction(self, input, im_size, exclude_nms=False):
        """
        Get prediction result of YOLOv3 network

        Args:
            input (list): List of Variables, output of backbone stages
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): The prediction result after non-max suppress.

        """

        outputs = self._get_outputs(input, is_train=False)

        boxes = []
        scores = []
        for i, output in enumerate(outputs):
            if self.iou_aware:
                output = get_iou_aware_score(output,
                                             len(self.anchor_masks[i]),
                                             self.num_classes,
                                             self.iou_aware_factor)
            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms.score_threshold,
                downsample_ratio=self.downsample[i],
                name=self.prefix_name + "yolo_box" + str(i),
                clip_bbox=self.clip_bbox,
                scale_x_y=scale_x_y)
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))

        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)

        # Only for benchmark, postprocess(NMS) is not needed
        if exclude_nms:
            return {'bbox': yolo_scores}

        if type(self.nms) is MultiClassSoftNMS:
            yolo_scores = fluid.layers.transpose(yolo_scores, perm=[0, 2, 1])
        pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        return {'bbox': pred}


@register
class YOLOv4Head(YOLOv3Head):
    """
    Head block for YOLOv4 network

    Args:
        anchors (list): anchors
        anchor_masks (list): anchor masks
        nms (object): an instance of `MultiClassNMS`
        spp_stage (int): apply spp on which stage.
        num_classes (int): number of output classes
        downsample (list): downsample ratio for each yolo_head
        scale_x_y (list): scale the center point of bbox at each stage
    """
    __inject__ = ['nms', 'yolo_loss']
    __shared__ = ['num_classes', 'weight_prefix_name']

    def __init__(self,
                 anchors=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                          [72, 146], [142, 110], [192, 243], [459, 401]],
                 anchor_masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=-1,
                     keep_top_k=-1,
                     nms_threshold=0.45,
                     background_label=-1).__dict__,
                 spp_stage=5,
                 num_classes=80,
                 weight_prefix_name='',
                 downsample=[8, 16, 32],
                 scale_x_y=1.0,
                 yolo_loss="YOLOv3Loss",
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 clip_bbox=False):
        super(YOLOv4Head, self).__init__(
            anchors=anchors,
            anchor_masks=anchor_masks,
            nms=nms,
            num_classes=num_classes,
            weight_prefix_name=weight_prefix_name,
            downsample=downsample,
            scale_x_y=scale_x_y,
            yolo_loss=yolo_loss,
            iou_aware=iou_aware,
            iou_aware_factor=iou_aware_factor,
            clip_bbox=clip_bbox)
        self.spp_stage = spp_stage

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out

    def max_pool(self, input, size):
        pad = [(size - 1) // 2] * 2
        return fluid.layers.pool2d(input, size, 'max', pool_padding=pad)

    def spp(self, input):
        branch_a = self.max_pool(input, 13)
        branch_b = self.max_pool(input, 9)
        branch_c = self.max_pool(input, 5)
        out = fluid.layers.concat([branch_a, branch_b, branch_c, input], axis=1)
        return out

    def stack_conv(self,
                   input,
                   ch_list=[512, 1024, 512],
                   filter_list=[1, 3, 1],
                   stride=1,
                   name=None):
        conv = input
        for i, (ch_out, f_size) in enumerate(zip(ch_list, filter_list)):
            padding = 1 if f_size == 3 else 0
            conv = self._conv_bn(
                conv,
                ch_out=ch_out,
                filter_size=f_size,
                stride=stride,
                padding=padding,
                name='{}.{}'.format(name, i))
        return conv

    def spp_module(self, input, name=None):
        conv = self.stack_conv(input, name=name + '.stack_conv.0')
        spp_out = self.spp(conv)
        conv = self.stack_conv(spp_out, name=name + '.stack_conv.1')
        return conv

    def pan_module(self, input, filter_list, name=None):
        for i in range(1, len(input)):
            ch_out = input[i].shape[1] // 2
            conv_left = self._conv_bn(
                input[i],
                ch_out=ch_out,
                filter_size=1,
                stride=1,
                padding=0,
                name=name + '.{}.left'.format(i))
            ch_out = input[i - 1].shape[1] // 2
            conv_right = self._conv_bn(
                input[i - 1],
                ch_out=ch_out,
                filter_size=1,
                stride=1,
                padding=0,
                name=name + '.{}.right'.format(i))
            conv_right = self._upsample(conv_right)
            pan_out = fluid.layers.concat([conv_left, conv_right], axis=1)
            ch_list = [pan_out.shape[1] // 2 * k for k in [1, 2, 1, 2, 1]]
            input[i] = self.stack_conv(
                pan_out,
                ch_list=ch_list,
                filter_list=filter_list,
                name=name + '.stack_conv.{}'.format(i))
        return input

    def _get_outputs(self, input, is_train=True):
        outputs = []
        filter_list = [1, 3, 1, 3, 1]
        spp_stage = len(input) - self.spp_stage
        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = input[-1:-out_layer_num - 1:-1]
        blocks[spp_stage] = self.spp_module(
            blocks[spp_stage], name=self.prefix_name + "spp_module")
        blocks = self.pan_module(
            blocks,
            filter_list=filter_list,
            name=self.prefix_name + 'pan_module')

        # reverse order back to input
        blocks = blocks[::-1]

        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                route = self._conv_bn(
                    route,
                    ch_out=route.shape[1] * 2,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    name=self.prefix_name + 'yolo_block.route.{}'.format(i))
                block = fluid.layers.concat(input=[route, block], axis=1)
                ch_list = [block.shape[1] // 2 * k for k in [1, 2, 1, 2, 1]]
                block = self.stack_conv(
                    block,
                    ch_list=ch_list,
                    filter_list=filter_list,
                    name=self.prefix_name +
                    'yolo_block.stack_conv.{}'.format(i))
            route = block

            block_out = self._conv_bn(
                block,
                ch_out=block.shape[1] * 2,
                filter_size=3,
                stride=1,
                padding=1,
                name=self.prefix_name + 'yolo_output.{}.conv.0'.format(i))

            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            block_out = fluid.layers.conv2d(
                input=block_out,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(name=self.prefix_name +
                                     "yolo_output.{}.conv.1.weights".format(i)),
                bias_attr=ParamAttr(
                    regularizer=L2Decay(0.),
                    name=self.prefix_name +
                    "yolo_output.{}.conv.1.bias".format(i)))
            outputs.append(block_out)

        return outputs
