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
from ppdet.modeling.bbox_utils import nonempty_bbox
from . import ops
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

__all__ = ['BBoxPostProcess', 'MaskPostProcess', 'FCOSPostProcess']


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

        # Prevent empty bbox_pred from decode or NMS.
        # Bboxes and score before NMS may be empty due to the score threshold.
        if bbox_pred.shape[0] == 0:
            bbox_pred = paddle.to_tensor(
                np.array(
                    [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
            bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))
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
        # TODO: Because paddle.expand transform error when dygraph
        # to static, use reshape to avoid mistakes.
        gx = paddle.reshape(gx, [N, img_y.shape[1], img_x.shape[2]])
        gy = paddle.reshape(gy, [N, img_y.shape[1], img_x.shape[2]])
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
class S2ANetBBoxPostProcess(object):
    __inject__ = ['nms']

    def __init__(self, nms_pre=2000, min_bbox_size=0, nms=None):
        super(S2ANetBBoxPostProcess, self).__init__()
        self.nms_pre = nms_pre
        self.min_bbox_size = min_bbox_size
        self.nms = nms
        self.origin_shape_list = []

    def rbox2poly(self, rrect, get_best_begin_point=True):
        """
        rrect: [N, 5] [x_ctr,y_ctr,w,h,angle]
        to
        poly:[x0,y0,x1,y1,x2,y2,x3,y3]
        """
        bbox_num = rrect.shape[0]
        x_ctr = rrect[:, 0]
        y_ctr = rrect[:, 1]
        width = rrect[:, 2]
        height = rrect[:, 3]
        angle = rrect[:, 4]

        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        # rect 2x4
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])

        # R:[2,2,M]  rect:[2,4,M]
        #poly = R.dot(rect)
        poly = []
        for i in range(R.shape[2]):
            poly.append(R[:, :, i].dot(rect[:, :, i]))
        # poly:[M, 2, 4]
        poly = np.array(poly)
        coor_x = poly[:, 0, :4] + x_ctr.reshape(bbox_num, 1)
        coor_y = poly[:, 1, :4] + y_ctr.reshape(bbox_num, 1)
        poly = np.stack(
            [
                coor_x[:, 0], coor_y[:, 0], coor_x[:, 1], coor_y[:, 1],
                coor_x[:, 2], coor_y[:, 2], coor_x[:, 3], coor_y[:, 3]
            ],
            axis=1)
        if get_best_begin_point:
            poly_lst = [get_best_begin_point_single(e) for e in poly]
            poly = np.array(poly_lst)
        return poly

    def get_prediction(self, pred_scores, pred_bboxes, im_shape, scale_factor):
        """
        pred_scores : [N, M]  score
        pred_bboxes : [N, 5]  xc, yc, w, h, a
        im_shape : [N, 2]  im_shape
        scale_factor : [N, 2]  scale_factor
        """
        # TODO: support bs>1
        pred_ploys = self.rbox2poly(pred_bboxes.numpy(), False)
        pred_ploys = paddle.to_tensor(pred_ploys)
        pred_ploys = paddle.reshape(
            pred_ploys, [1, pred_ploys.shape[0], pred_ploys.shape[1]])

        pred_scores = paddle.to_tensor(pred_scores)
        # pred_scores [NA, 16] --> [16, NA]
        pred_scores = paddle.transpose(pred_scores, [1, 0])
        pred_scores = paddle.reshape(
            pred_scores, [1, pred_scores.shape[0], pred_scores.shape[1]])
        pred_cls_score_bbox, bbox_num, index = self.nms(pred_ploys, pred_scores)

        # post process scale
        # result [n, 10]
        if bbox_num > 0:
            pred_bbox, bbox_num = self.post_process(pred_cls_score_bbox[:, 2:],
                                                    bbox_num, im_shape[0],
                                                    scale_factor[0])

            pred_cls_score_bbox = paddle.concat(
                [pred_cls_score_bbox[:, 0:2], pred_bbox], axis=1)
        else:
            pred_cls_score_bbox = paddle.to_tensor(
                np.array(
                    [[-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype='float32'))
            bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))
        return pred_cls_score_bbox, bbox_num, index

    def post_process(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to
        get final prediction.

        Args:
            bboxes(Tensor): bboxes [N, 8]
            bbox_num(Tensor): bbox_num
            im_shape(Tensor): [1 2]
            scale_factor(Tensor): [1 2]
        Returns:
            bbox_pred(Tensor): The output is the prediction with shape [N, 8]
                               including labels, scores and bboxes. The size of
                               bboxes are corresponding to the original image.
        """

        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

        origin_h = origin_shape[0]
        origin_w = origin_shape[1]

        bboxes[:, 0::2] = bboxes[:, 0::2] / scale_factor[0]
        bboxes[:, 1::2] = bboxes[:, 1::2] / scale_factor[1]

        zeros = paddle.zeros_like(origin_h)
        x1 = paddle.maximum(paddle.minimum(bboxes[:, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(bboxes[:, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(bboxes[:, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(bboxes[:, 3], origin_h), zeros)
        x3 = paddle.maximum(paddle.minimum(bboxes[:, 4], origin_w), zeros)
        y3 = paddle.maximum(paddle.minimum(bboxes[:, 5], origin_h), zeros)
        x4 = paddle.maximum(paddle.minimum(bboxes[:, 6], origin_w), zeros)
        y4 = paddle.maximum(paddle.minimum(bboxes[:, 7], origin_h), zeros)
        bbox = paddle.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)
        bboxes = (bbox, bbox_num)
        return bboxes
