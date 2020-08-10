import numpy as np
import paddle.fluid as fluid
from ppdet.core.workspace import register

# TODO: regitster mask_post_process op 
from ppdet.py_op.post_process import mask_post_process


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


@register
class Mask(object):
    __inject__ = ['mask_target_generator', 'mask_post_process']

    def __init__(self, mask_target_generator, mask_post_process):
        super(Mask, self).__init__()
        self.mask_target_generator = mask_target_generator
        self.mask_post_process = mask_post_process

    def __call__(self, inputs, rois, targets):
        mask_rois, rois_has_mask_int32 = self.generate_mask_target(inputs, rois,
                                                                   targets)
        return mask_rois, rois_has_mask_int32

    def generate_mask_target(self, inputs, rois, targets):
        labels_int32 = targets['labels_int32']
        proposals, proposals_num = rois
        mask_rois, mask_rois_num, self.rois_has_mask_int32, self.mask_int32 = self.mask_target_generator(
            im_info=inputs['im_info'],
            gt_classes=inputs['gt_class'],
            is_crowd=inputs['is_crowd'],
            gt_segms=inputs['gt_mask'],
            rois=proposals,
            rois_num=proposals_num,
            labels_int32=labels_int32)
        self.mask_rois = (mask_rois, mask_rois_num)
        return self.mask_rois, self.rois_has_mask_int32

    def get_targets(self):
        return self.mask_int32

    def post_process(self, bboxes, mask_head_out, im_info):
        mask = self.mask_post_process(bboxes, mask_head_out, im_info)
        return mask
