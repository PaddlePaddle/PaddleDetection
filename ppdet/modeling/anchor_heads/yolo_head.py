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
from ppdet.core.workspace import register

__all__ = ['YOLOv3Head']


@register
class YOLOv3Head(object):
    """
    Head block for YOLOv3 network

    Args:
        norm_decay (float): weight decay for normalization layer weights
        num_classes (int): number of output classes
        ignore_thresh (float): threshold to ignore confidence loss
        label_smooth (bool): whether to use label smoothing
        anchors (list): anchors
        anchor_masks (list): anchor masks
        nms (object): an instance of `MultiClassNMS`
    """
    __inject__ = ['nms']
    __shared__ = ['num_classes', 'weight_prefix_name']

    def __init__(self,
                 norm_decay=0.,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=True,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__,
                 use_splited_loss=False,
                 batch_size=8,
                 weight_prefix_name=''):
        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.nms = nms
        self._use_splited_loss = use_splited_loss
        self._batch_size = batch_size
        self.prefix_name = weight_prefix_name
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
        for j in range(2):
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                is_test=is_test,
                name='{}.{}.0'.format(name, j))
            conv = self._conv_bn(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                is_test=is_test,
                name='{}.{}.1'.format(name, j))
        route = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
            name='{}.2'.format(name))
        tip = self._conv_bn(
            route,
            channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
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
                channel=512 // (2**i),
                is_test=(not is_train),
                name=self.prefix_name + "yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(name=self.prefix_name +
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

        if self._use_splited_loss:
            return self.get_splited_loss(outputs, targets, gt_box)
        
        losses = []
        downsample = 32
        for i, output in enumerate(outputs):
            anchor_mask = self.anchor_masks[i]
            loss = fluid.layers.yolov3_loss(
                x=output,
                gt_box=gt_box,
                gt_label=gt_label,
                gt_score=gt_score,
                anchors=self.anchors,
                anchor_mask=anchor_mask,
                class_num=self.num_classes,
                ignore_thresh=self.ignore_thresh,
                downsample_ratio=downsample,
                use_label_smooth=self.label_smooth,
                name=self.prefix_name + "yolo_loss" + str(i))
            losses.append(fluid.layers.reduce_mean(loss))
            downsample //= 2

        return {'loss': sum(losses)}

    def get_splited_loss(self, outputs, targets, gt_box):
        """
        Calculate splited YOLOv3 loss

        Args:
            outputs ([Variables]): List of Variables, output of backbone stages
            targets ([Variables]): List of Variables, The targets for yolo
                                   loss calculatation.
            gt_box (Variable): The ground-truth boudding boxes.

        Returns:
            Type: dict
                xy_loss (Variable): YOLOv3 (x, y) coordinates loss
                wh_loss (Variable): YOLOv3 (w, h) coordinates loss
                obj_loss (Variable): YOLOv3 objectness score loss
                cls_loss (Variable): YOLOv3 classification loss

        """

        assert len(outputs) == len(targets), \
            "YOLOv3 output layer number not equal target number"


        downsample = 32
        loss_xys, loss_whs, loss_obj_poss, loss_obj_negs, loss_clss = [], [], [], [], []
        for i, (output, target, mask) in enumerate(zip(outputs, targets, self.anchor_masks)):
            print("output", i, output)
            # split x, y, w, h, obj, cls
            x = fluid.layers.strided_slice(output, axes=[1], starts=[0],
                        ends=[output.shape[1]], strides=[5 + self.num_classes])
            y = fluid.layers.strided_slice(output, axes=[1], starts=[1],
                        ends=[output.shape[1]], strides=[5 + self.num_classes])
            w = fluid.layers.strided_slice(output, axes=[1], starts=[2],
                        ends=[output.shape[1]], strides=[5 + self.num_classes])
            h = fluid.layers.strided_slice(output, axes=[1], starts=[3],
                        ends=[output.shape[1]], strides=[5 + self.num_classes])
            obj = fluid.layers.strided_slice(output, axes=[1], starts=[4],
                        ends=[output.shape[1]], strides=[5 + self.num_classes])
            clss = []
            stride = output.shape[1] // len(mask)
            for m in range(len(mask)):
                clss.append(fluid.layers.slice(output, axes=[1], starts=[stride*m+5],
                                                ends=[stride*m+5+self.num_classes]))
            cls = fluid.layers.transpose(fluid.layers.stack(clss, axis=1), perm=[0, 1, 3, 4, 2])

            # split tx, ty, tw, th, tobj, tcls
            tx = target[:, :, 0, :, :]
            ty = target[:, :, 1, :, :]
            tw = target[:, :, 2, :, :]
            th = target[:, :, 3, :, :]
            tscale = target[:, :, 4, :, :]
            tobj = target[:, :, 5, :, :]
            tcls = fluid.layers.transpose(target[:, :, 6:, :, :], perm=[0, 1, 3, 4, 2])
            tcls.stop_gradient = True

            # NOTE: tobj holds gt_score, obj_mask holds object existence mask
            obj_mask = fluid.layers.cast(tobj > 0., dtype="float32")

            loss_x = fluid.layers.sigmoid_cross_entropy_with_logits(x, tx) * tscale * tobj
            loss_x = fluid.layers.reduce_sum(loss_x * obj_mask, dim=[1, 2, 3])
            loss_y = fluid.layers.sigmoid_cross_entropy_with_logits(y, ty) * tscale * tobj
            loss_y = fluid.layers.reduce_sum(loss_y * obj_mask, dim=[1, 2, 3])
            loss_w = fluid.layers.abs(w - tw) * tscale * tobj
            loss_w = fluid.layers.reduce_sum(loss_w * obj_mask, dim=[1, 2, 3])
            loss_h = fluid.layers.abs(h - th) * tscale * tobj
            loss_h = fluid.layers.reduce_sum(loss_h * obj_mask, dim=[1, 2, 3])

            # pred bbox overlap any gt_bbox over ignore_thresh, 
            # objectness loss will be ignored
            # 1. get pred bbox, note yolo_box api is will cut bbox by image boundary,
            #    which may incur diff, further test required
            bbox, _ = fluid.layers.yolo_box(
                x=output,
                img_size=fluid.layers.ones(shape=[self._batch_size, 2], dtype="int32"),
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=0.,
                downsample_ratio=downsample,
                name=self.prefix_name + "yolo_box" + str(i))
            # 2. split pred bbox and gt bbox by sample, calc IoU between pred bbox
            #    and gt bbox in each sample
            if self._batch_size > 1:
                preds = fluid.layers.split(bbox, self._batch_size, dim=0)
                gts = fluid.layers.split(gt_box, self._batch_size, dim=0)
            else:
                preds = [bbox]
                gts = [gt_box]
            ious = []
            for pred, gt in zip(preds, gts):
                def box_xywh2xyxy(box):
                    x = box[:, 0]
                    y = box[:, 1]
                    w = box[:, 2]
                    h = box[:, 3]
                    return fluid.layers.stack([
                        x - w / 2.,
                        y - h / 2.,
                        x + w / 2.,
                        y + h / 2.,
                        ], axis=1)
                pred = fluid.layers.squeeze(pred, axes=[0])
                gt = box_xywh2xyxy(fluid.layers.squeeze(gt, axes=[0]))
                ious.append(fluid.layers.iou_similarity(pred, gt))
            iou = fluid.layers.stack(ious, axis=0)
            # 3. max_iou <= ignore_threshold as iou_mask for objectness loss calc
            max_iou = fluid.layers.reduce_max(iou, dim=-1)
            fluid.layers.Print(max_iou, message="max_iou{}".format(i), summarize=-1)
            iou_mask = fluid.layers.cast(max_iou <= self.ignore_thresh, dtype="float32")
            output_shape = fluid.layers.shape(output)
            iou_mask = fluid.layers.reshape(iou_mask, (-1, len(mask), output_shape[2], output_shape[3]))
            iou_mask = fluid.layers.elementwise_max(iou_mask, obj_mask)
            iou_mask.stop_gradient = True

            loss_obj = fluid.layers.sigmoid_cross_entropy_with_logits(obj, obj_mask) * iou_mask
            # loss_obj = fluid.layers.reduce_sum(loss_obj * tobj + loss_obj * (1. - obj_mask), dim=[1, 2, 3])
            loss_obj_pos = fluid.layers.reduce_sum(loss_obj * tobj, dim=[1, 2, 3])
            loss_obj_neg = fluid.layers.reduce_sum(loss_obj * (1.0 - obj_mask) * iou_mask, dim=[1, 2, 3])

            loss_cls = fluid.layers.sigmoid_cross_entropy_with_logits(cls, tcls) * tobj
            loss_cls = fluid.layers.reduce_sum(loss_cls * obj_mask, dim=[1, 2, 3, 4])
            downsample //= 2

            loss_xys.append(fluid.layers.reduce_mean(loss_x + loss_y))
            loss_whs.append(fluid.layers.reduce_mean(loss_w + loss_h))
            loss_obj_poss.append(fluid.layers.reduce_mean(loss_obj_pos))
            loss_obj_negs.append(fluid.layers.reduce_mean(loss_obj_neg))
            loss_clss.append(fluid.layers.reduce_mean(loss_cls))
            # fluid.layers.Print(loss_xys[-1], message="loss_xy{}".format(i))
            # fluid.layers.Print(loss_whs[-1], message="loss_wh{}".format(i))
            # fluid.layers.Print(loss_obj_poss[-1], message="loss_obj_pos{}".format(i))
            # fluid.layers.Print(loss_obj_negs[-1], message="loss_obj_neg{}".format(i))
            # fluid.layers.Print(loss_clss[-1], message="loss_cls{}".format(i))

        return { "loss_xy": fluid.layers.sum(loss_xys),
                 "loss_wh": fluid.layers.sum(loss_whs),
                 "loss_obj_pos": fluid.layers.sum(loss_obj_poss),
                 "loss_obj_neg": fluid.layers.sum(loss_obj_negs),
                 "loss_cls": fluid.layers.sum(loss_clss), }

    def get_prediction(self, input, im_size):
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
        pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        return {'bbox': pred}
