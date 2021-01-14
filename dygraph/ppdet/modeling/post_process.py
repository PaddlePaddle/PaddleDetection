import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.py_op.post_process import mask_post_process
from . import ops


@register
class BBoxPostProcess(object):
    __inject__ = ['decode', 'nms']

    def __init__(self, decode=None, nms=None):
        super(BBoxPostProcess, self).__init__()
        self.decode = decode
        self.nms = nms

    def __call__(self, head_out, rois, im_shape, scale_factor=None):
        bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score)
        return bbox_pred, bbox_num


@register
class MaskPostProcess(object):
    __shared__ = ['mask_resolution']

    def __init__(self, mask_resolution=28, binary_thresh=0.5):
        super(MaskPostProcess, self).__init__()
        self.mask_resolution = mask_resolution
        self.binary_thresh = binary_thresh

    def __call__(self, bboxes, mask_head_out, im_shape, scale_factor=None):
        # TODO: modify related ops for deploying
        bboxes_np = (i.numpy() for i in bboxes)
        mask = mask_post_process(bboxes_np,
                                 mask_head_out.numpy(),
                                 im_shape.numpy(), scale_factor[:, 0].numpy(),
                                 self.mask_resolution, self.binary_thresh)
        mask = {'mask': mask}
        return mask


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
