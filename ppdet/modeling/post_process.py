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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling.bbox_utils import nonempty_bbox, rbox2poly
from ppdet.modeling.layers import TTFBox
from .transformers import bbox_cxcywh_to_xyxy
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

__all__ = [
    'BBoxPostProcess', 'MaskPostProcess', 'FCOSPostProcess',
    'S2ANetBBoxPostProcess', 'JDEBBoxPostProcess', 'CenterNetPostProcess',
    'DETRBBoxPostProcess', 'SparsePostProcess'
]


@register
class BBoxPostProcess(object):
    __shared__ = ['num_classes', 'export_onnx']
    __inject__ = ['decode', 'nms']

    def __init__(self, num_classes=80, decode=None, nms=None,
                 export_onnx=False):
        super(BBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.export_onnx = export_onnx

    def __call__(self, head_out, rois, im_shape, scale_factor):
        """
        Decode the bbox and do NMS if needed.

        Args:
            head_out (tuple): bbox_pred and cls_prob of bbox_head output.
            rois (tuple): roi and rois_num of rpn_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
            export_onnx (bool): whether export model to onnx
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
        """
        if self.nms is not None:
            bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
            bbox_pred, bbox_num, _ = self.nms(bboxes, score, self.num_classes)

        else:
            bbox_pred, bbox_num = self.decode(head_out, rois, im_shape,
                                              scale_factor)

        if self.export_onnx:
            # add fake box after postprocess when exporting onnx 
            fake_bboxes = paddle.to_tensor(
                np.array(
                    [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype='float32'))

            bbox_pred = paddle.concat([bbox_pred, fake_bboxes])
            bbox_num = bbox_num + 1

        return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to 
        get final prediction. 

        Notes:
        Currently only support bs = 1.

        Args:
            bboxes (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            pred_result (Tensor): The final prediction results with shape [N, 6]
                including labels, scores and bboxes.
        """
        if not self.export_onnx:
            bboxes_list = []
            bbox_num_list = []
            id_start = 0
            fake_bboxes = paddle.to_tensor(
                np.array(
                    [[0., 0.0, 0.0, 0.0, 1.0, 1.0]], dtype='float32'))
            fake_bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))

            # add fake bbox when output is empty for each batch
            for i in range(bbox_num.shape[0]):
                if bbox_num[i] == 0:
                    bboxes_i = fake_bboxes
                    bbox_num_i = fake_bbox_num
                else:
                    bboxes_i = bboxes[id_start:id_start + bbox_num[i], :]
                    bbox_num_i = bbox_num[i]
                    id_start += bbox_num[i]
                bboxes_list.append(bboxes_i)
                bbox_num_list.append(bbox_num_i)
            bboxes = paddle.concat(bboxes_list)
            bbox_num = paddle.concat(bbox_num_list)

        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

        if not self.export_onnx:
            origin_shape_list = []
            scale_factor_list = []
            # scale_factor: scale_y, scale_x
            for i in range(bbox_num.shape[0]):
                expand_shape = paddle.expand(origin_shape[i:i + 1, :],
                                             [bbox_num[i], 2])
                scale_y, scale_x = scale_factor[i][0], scale_factor[i][1]
                scale = paddle.concat([scale_x, scale_y, scale_x, scale_y])
                expand_scale = paddle.expand(scale, [bbox_num[i], 4])
                origin_shape_list.append(expand_shape)
                scale_factor_list.append(expand_scale)

            self.origin_shape_list = paddle.concat(origin_shape_list)
            scale_factor_list = paddle.concat(scale_factor_list)

        else:
            # simplify the computation for bs=1 when exporting onnx
            scale_y, scale_x = scale_factor[0][0], scale_factor[0][1]
            scale = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y]).unsqueeze(0)
            self.origin_shape_list = paddle.expand(origin_shape,
                                                   [bbox_num[0], 2])
            scale_factor_list = paddle.expand(scale, [bbox_num[0], 4])

        # bboxes: [N, 6], label, score, bbox
        pred_label = bboxes[:, 0:1]
        pred_score = bboxes[:, 1:2]
        pred_bbox = bboxes[:, 2:]
        # rescale bbox to original image
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = self.origin_shape_list[:, 0]
        origin_w = self.origin_shape_list[:, 1]
        zeros = paddle.zeros_like(origin_h)
        # clip bbox to [0, original_size]
        x1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 3], origin_h), zeros)
        pred_bbox = paddle.stack([x1, y1, x2, y2], axis=-1)
        # filter empty bbox
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = paddle.unsqueeze(keep_mask, [1])
        pred_label = paddle.where(keep_mask, pred_label,
                                  paddle.ones_like(pred_label) * -1)
        pred_result = paddle.concat([pred_label, pred_score, pred_bbox], axis=1)
        return bboxes, pred_result, bbox_num

    def get_origin_shape(self, ):
        return self.origin_shape_list


@register
class MaskPostProcess(object):
    __shared__ = ['export_onnx']
    """
    refer to:
    https://github.com/facebookresearch/detectron2/layers/mask_ops.py

    Get Mask output according to the output from model
    """

    def __init__(self, binary_thresh=0.5, export_onnx=False):
        super(MaskPostProcess, self).__init__()
        self.binary_thresh = binary_thresh
        self.export_onnx = export_onnx

    def paste_mask(self, masks, boxes, im_h, im_w):
        """
        Paste the mask prediction to the original image.
        """
        x0_int, y0_int = 0, 0
        x1_int, y1_int = im_w, im_h
        x0, y0, x1, y1 = paddle.split(boxes, 4, axis=1)
        N = masks.shape[0]
        img_y = paddle.arange(y0_int, y1_int) + 0.5
        img_x = paddle.arange(x0_int, x1_int) + 0.5

        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        # img_x, img_y have shapes (N, w), (N, h)

        gx = img_x[:, None, :].expand(
            [N, paddle.shape(img_y)[1], paddle.shape(img_x)[1]])
        gy = img_y[:, :, None].expand(
            [N, paddle.shape(img_y)[1], paddle.shape(img_x)[1]])
        grid = paddle.stack([gx, gy], axis=3)
        img_masks = F.grid_sample(masks, grid, align_corners=False)
        return img_masks[:, 0]

    def __call__(self, mask_out, bboxes, bbox_num, origin_shape):
        """
        Decode the mask_out and paste the mask to the origin image.

        Args:
            mask_out (Tensor): mask_head output with shape [N, 28, 28].
            bbox_pred (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            origin_shape (Tensor): The origin shape of the input image, the tensor
                shape is [N, 2], and each row is [h, w].
        Returns:
            pred_result (Tensor): The final prediction mask results with shape
                [N, h, w] in binary mask style.
        """
        num_mask = mask_out.shape[0]
        origin_shape = paddle.cast(origin_shape, 'int32')

        if self.export_onnx:
            h, w = origin_shape[0][0], origin_shape[0][1]
            mask_onnx = self.paste_mask(mask_out[:, None, :, :], bboxes[:, 2:],
                                        h, w)
            mask_onnx = mask_onnx >= self.binary_thresh
            pred_result = paddle.cast(mask_onnx, 'int32')

        else:
            max_h = paddle.max(origin_shape[:, 0])
            max_w = paddle.max(origin_shape[:, 1])
            pred_result = paddle.zeros(
                [num_mask, max_h, max_w], dtype='int32') - 1

            id_start = 0
            for i in range(paddle.shape(bbox_num)[0]):
                bboxes_i = bboxes[id_start:id_start + bbox_num[i], :]
                mask_out_i = mask_out[id_start:id_start + bbox_num[i], :, :]
                im_h = origin_shape[i, 0]
                im_w = origin_shape[i, 1]
                bbox_num_i = bbox_num[id_start]
                pred_mask = self.paste_mask(mask_out_i[:, None, :, :],
                                            bboxes_i[:, 2:], im_h, im_w)
                pred_mask = paddle.cast(pred_mask >= self.binary_thresh,
                                        'int32')
                pred_result[id_start:id_start + bbox_num[i], :im_h, :
                            im_w] = pred_mask
                id_start += bbox_num[i]

        return pred_result


@register
class FCOSPostProcess(object):
    __inject__ = ['decode', 'nms']

    def __init__(self, decode=None, nms=None):
        super(FCOSPostProcess, self).__init__()
        self.decode = decode
        self.nms = nms

    def __call__(self, fcos_head_outs, scale_factor):
        """
        Decode the bbox and do NMS in FCOS.
        """
        locations, cls_logits, bboxes_reg, centerness = fcos_head_outs
        bboxes, score = self.decode(locations, cls_logits, bboxes_reg,
                                    centerness, scale_factor)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score)
        return bbox_pred, bbox_num


@register
class S2ANetBBoxPostProcess(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['nms']

    def __init__(self, num_classes=15, nms_pre=2000, min_bbox_size=0, nms=None):
        super(S2ANetBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.nms_pre = nms_pre
        self.min_bbox_size = min_bbox_size
        self.nms = nms
        self.origin_shape_list = []
        self.fake_pred_cls_score_bbox = paddle.to_tensor(
            np.array(
                [[-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype='float32'))
        self.fake_bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))

    def forward(self, pred_scores, pred_bboxes):
        """
        pred_scores : [N, M]  score
        pred_bboxes : [N, 5]  xc, yc, w, h, a
        im_shape : [N, 2]  im_shape
        scale_factor : [N, 2]  scale_factor
        """
        pred_ploys0 = rbox2poly(pred_bboxes)
        pred_ploys = paddle.unsqueeze(pred_ploys0, axis=0)

        # pred_scores [NA, 16] --> [16, NA]
        pred_scores0 = paddle.transpose(pred_scores, [1, 0])
        pred_scores = paddle.unsqueeze(pred_scores0, axis=0)

        pred_cls_score_bbox, bbox_num, _ = self.nms(pred_ploys, pred_scores,
                                                    self.num_classes)
        # Prevent empty bbox_pred from decode or NMS.
        # Bboxes and score before NMS may be empty due to the score threshold.
        if pred_cls_score_bbox.shape[0] <= 0 or pred_cls_score_bbox.shape[
                1] <= 1:
            pred_cls_score_bbox = self.fake_pred_cls_score_bbox
            bbox_num = self.fake_bbox_num

        pred_cls_score_bbox = paddle.reshape(pred_cls_score_bbox, [-1, 10])
        return pred_cls_score_bbox, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to
        get final prediction.
        Args:
            bboxes(Tensor): bboxes [N, 10]
            bbox_num(Tensor): bbox_num
            im_shape(Tensor): [1 2]
            scale_factor(Tensor): [1 2]
        Returns:
            bbox_pred(Tensor): The output is the prediction with shape [N, 8]
                               including labels, scores and bboxes. The size of
                               bboxes are corresponding to the original image.
        """
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

        origin_shape_list = []
        scale_factor_list = []
        # scale_factor: scale_y, scale_x
        for i in range(bbox_num.shape[0]):
            expand_shape = paddle.expand(origin_shape[i:i + 1, :],
                                         [bbox_num[i], 2])
            scale_y, scale_x = scale_factor[i][0], scale_factor[i][1]
            scale = paddle.concat([
                scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x,
                scale_y
            ])
            expand_scale = paddle.expand(scale, [bbox_num[i], 8])
            origin_shape_list.append(expand_shape)
            scale_factor_list.append(expand_scale)

        origin_shape_list = paddle.concat(origin_shape_list)
        scale_factor_list = paddle.concat(scale_factor_list)

        # bboxes: [N, 10], label, score, bbox
        pred_label_score = bboxes[:, 0:2]
        pred_bbox = bboxes[:, 2:]

        # rescale bbox to original image
        pred_bbox = pred_bbox.reshape([-1, 8])
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = origin_shape_list[:, 0]
        origin_w = origin_shape_list[:, 1]

        bboxes = scaled_bbox
        zeros = paddle.zeros_like(origin_h)
        x1 = paddle.maximum(paddle.minimum(bboxes[:, 0], origin_w - 1), zeros)
        y1 = paddle.maximum(paddle.minimum(bboxes[:, 1], origin_h - 1), zeros)
        x2 = paddle.maximum(paddle.minimum(bboxes[:, 2], origin_w - 1), zeros)
        y2 = paddle.maximum(paddle.minimum(bboxes[:, 3], origin_h - 1), zeros)
        x3 = paddle.maximum(paddle.minimum(bboxes[:, 4], origin_w - 1), zeros)
        y3 = paddle.maximum(paddle.minimum(bboxes[:, 5], origin_h - 1), zeros)
        x4 = paddle.maximum(paddle.minimum(bboxes[:, 6], origin_w - 1), zeros)
        y4 = paddle.maximum(paddle.minimum(bboxes[:, 7], origin_h - 1), zeros)
        pred_bbox = paddle.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)
        pred_result = paddle.concat([pred_label_score, pred_bbox], axis=1)
        return pred_result


@register
class JDEBBoxPostProcess(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['decode', 'nms']

    def __init__(self, num_classes=1, decode=None, nms=None, return_idx=True):
        super(JDEBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.return_idx = return_idx

        self.fake_bbox_pred = paddle.to_tensor(
            np.array(
                [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
        self.fake_bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))
        self.fake_nms_keep_idx = paddle.to_tensor(
            np.array(
                [[0]], dtype='int32'))

        self.fake_yolo_boxes_out = paddle.to_tensor(
            np.array(
                [[[0.0, 0.0, 0.0, 0.0]]], dtype='float32'))
        self.fake_yolo_scores_out = paddle.to_tensor(
            np.array(
                [[[0.0]]], dtype='float32'))
        self.fake_boxes_idx = paddle.to_tensor(np.array([[0]], dtype='int64'))

    def forward(self, head_out, anchors):
        """
        Decode the bbox and do NMS for JDE model. 

        Args:
            head_out (list): Bbox_pred and cls_prob of bbox_head output.
            anchors (list): Anchors of JDE model.

        Returns:
            boxes_idx (Tensor): The index of kept bboxes after decode 'JDEBox'. 
            bbox_pred (Tensor): The output is the prediction with shape [N, 6]
                including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction of each batch with shape [N].
            nms_keep_idx (Tensor): The index of kept bboxes after NMS. 
        """
        boxes_idx, yolo_boxes_scores = self.decode(head_out, anchors)

        if len(boxes_idx) == 0:
            boxes_idx = self.fake_boxes_idx
            yolo_boxes_out = self.fake_yolo_boxes_out
            yolo_scores_out = self.fake_yolo_scores_out
        else:
            yolo_boxes = paddle.gather_nd(yolo_boxes_scores, boxes_idx)
            # TODO: only support bs=1 now
            yolo_boxes_out = paddle.reshape(
                yolo_boxes[:, :4], shape=[1, len(boxes_idx), 4])
            yolo_scores_out = paddle.reshape(
                yolo_boxes[:, 4:5], shape=[1, 1, len(boxes_idx)])
            boxes_idx = boxes_idx[:, 1:]

        if self.return_idx:
            bbox_pred, bbox_num, nms_keep_idx = self.nms(
                yolo_boxes_out, yolo_scores_out, self.num_classes)
            if bbox_pred.shape[0] == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
                nms_keep_idx = self.fake_nms_keep_idx
            return boxes_idx, bbox_pred, bbox_num, nms_keep_idx
        else:
            bbox_pred, bbox_num, _ = self.nms(yolo_boxes_out, yolo_scores_out,
                                              self.num_classes)
            if bbox_pred.shape[0] == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
            return _, bbox_pred, bbox_num, _


@register
class CenterNetPostProcess(TTFBox):
    """
    Postprocess the model outputs to get final prediction:
        1. Do NMS for heatmap to get top `max_per_img` bboxes.
        2. Decode bboxes using center offset and box size.
        3. Rescale decoded bboxes reference to the origin image shape.

    Args:
        max_per_img(int): the maximum number of predicted objects in a image,
            500 by default.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        regress_ltrb (bool): whether to regress left/top/right/bottom or
            width/height for a box, true by default.
        for_mot (bool): whether return other features used in tracking model.
    """

    __shared__ = ['down_ratio', 'for_mot']

    def __init__(self,
                 max_per_img=500,
                 down_ratio=4,
                 regress_ltrb=True,
                 for_mot=False):
        super(TTFBox, self).__init__()
        self.max_per_img = max_per_img
        self.down_ratio = down_ratio
        self.regress_ltrb = regress_ltrb
        self.for_mot = for_mot

    def __call__(self, hm, wh, reg, im_shape, scale_factor):
        heat = self._simple_nms(hm)
        scores, inds, topk_clses, ys, xs = self._topk(heat)
        scores = scores.unsqueeze(1)
        clses = topk_clses.unsqueeze(1)

        reg_t = paddle.transpose(reg, [0, 2, 3, 1])
        # Like TTFBox, batch size is 1.
        # TODO: support batch size > 1
        reg = paddle.reshape(reg_t, [-1, reg_t.shape[-1]])
        reg = paddle.gather(reg, inds)
        xs = paddle.cast(xs, 'float32')
        ys = paddle.cast(ys, 'float32')
        xs = xs + reg[:, 0:1]
        ys = ys + reg[:, 1:2]

        wh_t = paddle.transpose(wh, [0, 2, 3, 1])
        wh = paddle.reshape(wh_t, [-1, wh_t.shape[-1]])
        wh = paddle.gather(wh, inds)

        if self.regress_ltrb:
            x1 = xs - wh[:, 0:1]
            y1 = ys - wh[:, 1:2]
            x2 = xs + wh[:, 2:3]
            y2 = ys + wh[:, 3:4]
        else:
            x1 = xs - wh[:, 0:1] / 2
            y1 = ys - wh[:, 1:2] / 2
            x2 = xs + wh[:, 0:1] / 2
            y2 = ys + wh[:, 1:2] / 2

        n, c, feat_h, feat_w = paddle.shape(hm)
        padw = (feat_w * self.down_ratio - im_shape[0, 1]) / 2
        padh = (feat_h * self.down_ratio - im_shape[0, 0]) / 2
        x1 = x1 * self.down_ratio
        y1 = y1 * self.down_ratio
        x2 = x2 * self.down_ratio
        y2 = y2 * self.down_ratio

        x1 = x1 - padw
        y1 = y1 - padh
        x2 = x2 - padw
        y2 = y2 - padh

        bboxes = paddle.concat([x1, y1, x2, y2], axis=1)
        scale_y = scale_factor[:, 0:1]
        scale_x = scale_factor[:, 1:2]
        scale_expand = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y], axis=1)
        boxes_shape = bboxes.shape[:]
        scale_expand = paddle.expand(scale_expand, shape=boxes_shape)
        bboxes = paddle.divide(bboxes, scale_expand)
        results = paddle.concat([clses, scores, bboxes], axis=1)
        if self.for_mot:
            return results, inds, topk_clses
        else:
            return results, paddle.shape(results)[0:1], topk_clses


@register
class DETRBBoxPostProcess(object):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = []

    def __init__(self,
                 num_classes=80,
                 num_top_queries=100,
                 use_focal_loss=False):
        super(DETRBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss

    def __call__(self, head_out, im_shape, scale_factor):
        """
        Decode the bbox.

        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [bs], and is N.
        """
        bboxes, logits, masks = head_out

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        img_h, img_w = origin_shape.unbind(1)
        origin_shape = paddle.stack(
            [img_w, img_h, img_w, img_h], axis=-1).unsqueeze(0)
        bbox_pred *= origin_shape

        scores = F.sigmoid(logits) if self.use_focal_loss else F.softmax(
            logits)[:, :, :-1]

        if not self.use_focal_loss:
            scores, labels = scores.max(-1), scores.argmax(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = paddle.topk(
                    scores, self.num_top_queries, axis=-1)
                labels = paddle.stack(
                    [paddle.gather(l, i) for l, i in zip(labels, index)])
                bbox_pred = paddle.stack(
                    [paddle.gather(b, i) for b, i in zip(bbox_pred, index)])
        else:
            scores, index = paddle.topk(
                scores.reshape([logits.shape[0], -1]),
                self.num_top_queries,
                axis=-1)
            labels = index % logits.shape[2]
            index = index // logits.shape[2]
            bbox_pred = paddle.stack(
                [paddle.gather(b, i) for b, i in zip(bbox_pred, index)])

        bbox_pred = paddle.concat(
            [
                labels.unsqueeze(-1).astype('float32'), scores.unsqueeze(-1),
                bbox_pred
            ],
            axis=-1)
        bbox_num = paddle.to_tensor(
            bbox_pred.shape[1], dtype='int32').tile([bbox_pred.shape[0]])
        bbox_pred = bbox_pred.reshape([-1, 6])
        return bbox_pred, bbox_num


@register
class SparsePostProcess(object):
    __shared__ = ['num_classes']

    def __init__(self, num_proposals, num_classes=80):
        super(SparsePostProcess, self).__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals

    def __call__(self, box_cls, box_pred, scale_factor_wh, img_whwh):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            scale_factor_wh (Tensor): tensors of shape [batch_size, 2] the scalor of  per img
            img_whwh (Tensor): tensors of shape [batch_size, 4]
        Returns:
            bbox_pred (Tensor): tensors of shape [num_boxes, 6] Each row has 6 values:
            [label, confidence, xmin, ymin, xmax, ymax]
            bbox_num (Tensor): tensors of shape [batch_size] the number of RoIs in each image.
        """
        assert len(box_cls) == len(scale_factor_wh) == len(img_whwh)

        img_wh = img_whwh[:, :2]

        scores = F.sigmoid(box_cls)
        labels = paddle.arange(0, self.num_classes). \
            unsqueeze(0).tile([self.num_proposals, 1]).flatten(start_axis=0, stop_axis=1)

        classes_all = []
        scores_all = []
        boxes_all = []
        for i, (scores_per_image,
                box_pred_per_image) in enumerate(zip(scores, box_pred)):

            scores_per_image, topk_indices = scores_per_image.flatten(
                0, 1).topk(
                    self.num_proposals, sorted=False)
            labels_per_image = paddle.gather(labels, topk_indices, axis=0)

            box_pred_per_image = box_pred_per_image.reshape([-1, 1, 4]).tile(
                [1, self.num_classes, 1]).reshape([-1, 4])
            box_pred_per_image = paddle.gather(
                box_pred_per_image, topk_indices, axis=0)

            classes_all.append(labels_per_image)
            scores_all.append(scores_per_image)
            boxes_all.append(box_pred_per_image)

        bbox_num = paddle.zeros([len(scale_factor_wh)], dtype="int32")
        boxes_final = []

        for i in range(len(scale_factor_wh)):
            classes = classes_all[i]
            boxes = boxes_all[i]
            scores = scores_all[i]

            boxes[:, 0::2] = paddle.clip(
                boxes[:, 0::2], min=0, max=img_wh[i][0]) / scale_factor_wh[i][0]
            boxes[:, 1::2] = paddle.clip(
                boxes[:, 1::2], min=0, max=img_wh[i][1]) / scale_factor_wh[i][1]
            boxes_w, boxes_h = (boxes[:, 2] - boxes[:, 0]).numpy(), (
                boxes[:, 3] - boxes[:, 1]).numpy()

            keep = (boxes_w > 1.) & (boxes_h > 1.)

            if (keep.sum() == 0):
                bboxes = paddle.zeros([1, 6]).astype("float32")
            else:
                boxes = paddle.to_tensor(boxes.numpy()[keep]).astype("float32")
                classes = paddle.to_tensor(classes.numpy()[keep]).astype(
                    "float32").unsqueeze(-1)
                scores = paddle.to_tensor(scores.numpy()[keep]).astype(
                    "float32").unsqueeze(-1)

                bboxes = paddle.concat([classes, scores, boxes], axis=-1)

            boxes_final.append(bboxes)
            bbox_num[i] = bboxes.shape[0]

        bbox_pred = paddle.concat(boxes_final)
        return bbox_pred, bbox_num


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    # _i, _j
    # sorted indices
    # i, j
    # temp variables for box i's (the box currently under consideration)
    # ix1, iy1, ix2, iy2, iarea

    # variables for computing overlap with box j (lower scoring box)
    # xx1, yy1, xx2, yy2
    # w, h
    # inter, ovr

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets
