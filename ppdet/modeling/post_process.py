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
from ppdet.modeling.bbox_utils import nonempty_bbox, rbox2poly, rbox2poly
from ppdet.modeling.layers import TTFBox
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

__all__ = [
    'BBoxPostProcess',
    'MaskPostProcess',
    'FCOSPostProcess',
    'S2ANetBBoxPostProcess',
    'JDEBBoxPostProcess',
    'CenterNetPostProcess',
]


@register
class BBoxPostProcess(object):
    __shared__ = ['num_classes']
    __inject__ = ['decode', 'nms']

    def __init__(self, num_classes=80, decode=None, nms=None):
        super(BBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms

    def __call__(self, head_out, rois, im_shape, scale_factor):
        """
        Decode the bbox and do NMS if needed. 

        Args:
            head_out (tuple): bbox_pred and cls_prob of bbox_head output.
            rois (tuple): roi and rois_num of rpn_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
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
        return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to 
        get final prediction. 
        
        Notes:
        Currently only support bs = 1.

        Args:
            bbox_pred (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            pred_result (Tensor): The final prediction results with shape [N, 6]
                including labels, scores and bboxes.
        """

        if bboxes.shape[0] == 0:
            bbox_pred = paddle.to_tensor(
                np.array(
                    [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
            bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))

        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

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
        return pred_result

    def get_origin_shape(self, ):
        return self.origin_shape_list


@register
class MaskPostProcess(object):
    def __init__(self, binary_thresh=0.5):
        super(MaskPostProcess, self).__init__()
        self.binary_thresh = binary_thresh

    def paste_mask(self, masks, boxes, im_h, im_w):
        """
        Paste the mask prediction to the original image.
        """
        x0, y0, x1, y1 = paddle.split(boxes, 4, axis=1)
        masks = paddle.unsqueeze(masks, [0, 1])
        img_y = paddle.arange(0, im_h, dtype='float32') + 0.5
        img_x = paddle.arange(0, im_w, dtype='float32') + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        img_x = paddle.unsqueeze(img_x, [1])
        img_y = paddle.unsqueeze(img_y, [2])
        N = boxes.shape[0]

        gx = paddle.expand(img_x, [N, img_y.shape[1], img_x.shape[2]])
        gy = paddle.expand(img_y, [N, img_y.shape[1], img_x.shape[2]])
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
        # TODO: support bs > 1 and mask output dtype is bool
        pred_result = paddle.zeros(
            [num_mask, origin_shape[0][0], origin_shape[0][1]], dtype='int32')
        if bbox_num == 1 and bboxes[0][0] == -1:
            return pred_result

        # TODO: optimize chunk paste
        pred_result = []
        for i in range(bboxes.shape[0]):
            im_h, im_w = origin_shape[i][0], origin_shape[i][1]
            pred_mask = self.paste_mask(mask_out[i], bboxes[i:i + 1, 2:], im_h,
                                        im_w)
            pred_mask = pred_mask >= self.binary_thresh
            pred_mask = paddle.cast(pred_mask, 'int32')
            pred_result.append(pred_mask)
        pred_result = paddle.concat(pred_result)
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
class JDEBBoxPostProcess(BBoxPostProcess):
    def __call__(self, head_out, anchors):
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
        boxes_idx, bboxes, score = self.decode(head_out, anchors)
        bbox_pred, bbox_num, nms_keep_idx = self.nms(bboxes, score,
                                                     self.num_classes)
        if bbox_pred.shape[0] == 0:
            bbox_pred = paddle.to_tensor(
                np.array(
                    [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
            bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))
            nms_keep_idx = paddle.to_tensor(np.array([[0]], dtype='int32'))

        return boxes_idx, bbox_pred, bbox_num, nms_keep_idx


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

    __shared__ = ['down_ratio']

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
        scores, inds, clses, ys, xs = self._topk(heat)
        scores = paddle.tensor.unsqueeze(scores, [1])
        clses = paddle.tensor.unsqueeze(clses, [1])

        reg_t = paddle.transpose(reg, [0, 2, 3, 1])
        # Like TTFBox, batch size is 1.
        # TODO: support batch size > 1
        reg = paddle.reshape(reg_t, [-1, paddle.shape(reg_t)[-1]])
        reg = paddle.gather(reg, inds)
        xs = paddle.cast(xs, 'float32')
        ys = paddle.cast(ys, 'float32')
        xs = xs + reg[:, 0:1]
        ys = ys + reg[:, 1:2]

        wh_t = paddle.transpose(wh, [0, 2, 3, 1])
        wh = paddle.reshape(wh_t, [-1, paddle.shape(wh_t)[-1]])
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
        boxes_shape = paddle.shape(bboxes)
        boxes_shape.stop_gradient = True
        scale_expand = paddle.expand(scale_expand, shape=boxes_shape)
        bboxes = paddle.divide(bboxes, scale_expand)
        if self.for_mot:
            results = paddle.concat([bboxes, scores, clses], axis=1)
            return results, inds
        else:
            results = paddle.concat([clses, scores, bboxes], axis=1)
            return results, paddle.shape(results)[0:1]
