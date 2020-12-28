import numpy as np
from ppdet.core.workspace import register


@register
class Mask(object):
    __inject__ = ['mask_target_generator']

    def __init__(self, mask_target_generator):
        super(Mask, self).__init__()
        self.mask_target_generator = mask_target_generator

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
            gt_segms=inputs['gt_poly'],
            rois=proposals,
            rois_num=proposals_num,
            labels_int32=labels_int32)
        self.mask_rois = (mask_rois, mask_rois_num)
        return self.mask_rois, self.rois_has_mask_int32

    def get_targets(self):
        return self.mask_int32
