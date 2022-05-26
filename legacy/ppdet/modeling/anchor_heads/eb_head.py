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
from paddle.fluid.regularizer import L2Decay

from ppdet.modeling.ops import MultiClassNMS
from ppdet.modeling.losses.yolo_loss import YOLOv3Loss
from ppdet.core.workspace import register

__all__ = ['EBHead']


@register
class EBHead(object):
    """
    Head block for pp-yolo-eb, ppyolo for EdgeBoard : https://ai.baidu.com/ai-doc/HWCE/Yk3b86gvp

    Args:
        norm_decay (float): weight decay for normalization layer weights
        num_classes (int): number of output classes
        anchors (list): anchors
        anchor_masks (list): anchor masks
        nms (object): an instance of `MultiClassNMS`
    """
    __inject__ = ['yolo_loss', 'nms']
    __shared__ = ['num_classes', 'weight_prefix_name']

    def __init__(self,
                 norm_decay=0.,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 yolo_loss="YOLOv3Loss",
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__,
                 weight_prefix_name=''):
        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.yolo_loss = yolo_loss
        self.nms = nms
        self.prefix_name = weight_prefix_name
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)

    def _conv_bn(self,
                 input,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='leaky',
                 is_test=True,
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
            is_test=is_test,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _detection_block(self, input, channel, is_test=True, name=None):
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2 in detection block {}" \
                .format(channel, name)

        conv = input
        conv = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
            name='{}.0'.format(name))
        for j in range(4):
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=3,
                stride=1,
                padding=1,
                is_test=is_test,
                name='{}.{}.1'.format(name, j))
            if j == 1:
                route = conv
        return route, conv

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out

    def _pool_concat(self, input):
        pool1 = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        pool2 = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='avg')
        out = fluid.layers.concat(input=[pool1, pool2], axis=1)

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
        Get ppyolo_eb head output

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

        filters_num1 = blocks[1].shape[1] // 2
        blk0 = self._pool_concat(blocks[2])
        blk0 = self._conv_bn(
            blk0,
            filters_num1,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=False,
            name='channel_fusion_1')
        blk1 = fluid.layers.concat(input=[blk0, blocks[1]], axis=1)

        filters_num2 = blocks[0].shape[1] // 2
        blk = self._conv_bn(
            blk1,
            filters_num2,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=False,
            name='channel_fusion_2')
        blk2 = self._conv_bn(
            blk,
            filters_num2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=False,
            name='feature_fusion')
        blk2 = self._pool_concat(blk2)
        blk2 = self._conv_bn(
            blk2,
            filters_num2,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=False,
            name='channel_fusion_3')
        blk3 = fluid.layers.concat(input=[blk2, blocks[0]], axis=1)

        blocks = [blk3, blk1, blocks[2]]

        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self._detection_block(
                block,
                channel=512 // (2**i),
                is_test=(not is_train),
                name=self.prefix_name + "yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
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
                    is_test=(not is_train),
                    name=self.prefix_name + "yolo_transition.{}".format(i))
                # upsample
                route = self._upsample(route)

        return outputs

    def get_loss(self, input, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of ppyolo_eb.

        Args:
            input (list): List of Variables, output of backbone stages
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of ppyolo_eb network.

        """
        outputs = self._get_outputs(input, is_train=True)

        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes,
                              self.prefix_name)

    def get_prediction(self, input, im_size, exclude_nms=False):
        """
        Get prediction result of ppyolo_eb network

        Args:
            input (list): List of Variables, output of backbone stages
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): The prediction result after non-max suppress.

        """

        outputs = self._get_outputs(input, is_train=False)

        boxes = []
        scores = []
        downsample = 32
        for i, output in enumerate(outputs):
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms.score_threshold,
                downsample_ratio=downsample,
                name=self.prefix_name + "yolo_box" + str(i))
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
            downsample //= 2

        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)

        # Only for benchmark, postprocess(NMS) is not needed
        if exclude_nms:
            return {'bbox': yolo_boxes, 'score': yolo_scores}

        pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        return {'bbox': pred}
