import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling.bbox_utils import nonempty_bbox
from . import ops


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
        bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score, self.num_classes)
        return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        # bboxes: [N, 6]
        if bboxes.shape[0] == 0:
            return bboxes
        # scale_factor: scale_y, scale_x
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)

        origin_shape_list = []
        scale_factor_list = []
        for i in range(bbox_num.shape[0]):
            expand_shape = paddle.expand(origin_shape[i:i + 1, :],
                                         [bbox_num[i], 2])
            scale_y, scale_x = scale_factor[i]
            scale = paddle.concat([scale_x, scale_y, scale_x, scale_y])
            expand_scale = paddle.expand(scale, [bbox_num[i], 4])
            origin_shape_list.append(expand_shape)
            scale_factor_list.append(expand_scale)

        self.origin_shape_list = paddle.concat(origin_shape_list)
        scale_factor_list = paddle.concat(scale_factor_list)

        pred_label = bboxes[:, 0:1]
        pred_score = bboxes[:, 1:2]
        pred_bbox = bboxes[:, 2:]
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = self.origin_shape_list[:, 0]
        origin_w = self.origin_shape_list[:, 1]
        zeros = paddle.zeros_like(origin_h)
        x1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 3], origin_h), zeros)
        pred_bbox = paddle.stack([x1, y1, x2, y2], axis=-1)

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
        # paste each mask on image
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
        if bboxes.shape == 0:
            return mask_out

        num_mask = mask_out.shape[0]
        # TODO: support bs > 1
        pred_result = paddle.zeros(
            [num_mask, origin_shape[0][0], origin_shape[0][1]], dtype='bool')
        # TODO: optimize chunk paste
        for i in range(bboxes.shape[0]):
            im_h, im_w = origin_shape[i]
            pred_mask = self.paste_mask(mask_out[i], bboxes[i:i + 1, 2:], im_h,
                                        im_w)
            pred_mask = pred_mask >= self.binary_thresh
            pred_result[i] = pred_mask
        return pred_result


@register
class FCOSPostProcess(object):
    __inject__ = ['decode', 'nms']

    def __init__(self, decode=None, nms=None):
        super(FCOSPostProcess, self).__init__()
        self.decode = decode
        self.nms = nms

    def __call__(self, fcos_head_outs, scale_factor):
        locations, cls_logits, bboxes_reg, centerness = fcos_head_outs
        bboxes, score = self.decode(locations, cls_logits, bboxes_reg,
                                    centerness, scale_factor)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score)
        return bbox_pred, bbox_num
