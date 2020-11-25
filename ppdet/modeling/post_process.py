import numpy as np
import paddle.fluid as fluid
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

    def __call__(self,
                 head_out,
                 rois,
                 im_shape,
                 scale_factor=None,
                 var_weight=1.):
        # TODO: compatible for im_info
        # remove after unify the im_shape. scale_factor
        if im_shape.shape[1] > 2:
            origin_shape = im_shape[:, :2]
            scale_factor = im_shape[:, 2:]
        else:
            origin_shape = im_shape
        bboxes, score = self.decode(head_out, rois, origin_shape, scale_factor,
                                    var_weight)
        bbox_pred, bbox_num = self.nms(bboxes, score)
        return bbox_pred, bbox_num


@register
class MaskPostProcess(object):
    __shared__ = ['mask_resolution']

    def __init__(self, mask_resolution=28, binary_thresh=0.5):
        super(MaskPostProcess, self).__init__()
        self.mask_resolution = mask_resolution
        self.binary_thresh = binary_thresh

    def __call__(self, bboxes, mask_head_out, im_info):
        # TODO: modify related ops for deploying
        bboxes_np = (i.numpy() for i in bboxes)
        mask = mask_post_process(bboxes_np,
                                 mask_head_out.numpy(),
                                 im_info.numpy(), self.mask_resolution,
                                 self.binary_thresh)
        mask = {'mask': mask}
        return mask
