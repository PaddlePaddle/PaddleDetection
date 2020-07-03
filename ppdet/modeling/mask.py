import numpy as np
import paddle.fluid as fluid
from ppdet.core.workspace import register
from ppdet.modeling.ops import MaskTargetGenerator
# TODO: modify here into ppdet.modeling.ops like DecodeClipNms 
from ppdet.py_op.post_process import mask_post_process


@register
class MaskPostProcess(object):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=81):
        super(MaskPostProcess, self).__init__()
        self.num_classes = num_classes

    def __call__(self, inputs):
        # TODO: modify related ops for deploying
        outs = mask_post_process(inputs['predicted_bbox_nums'].numpy(),
                                 inputs['predicted_bbox'].numpy(),
                                 inputs['mask_logits'].numpy(),
                                 inputs['im_info'].numpy())
        outs = {'predicted_mask': outs}
        return outs


@register
class Mask(object):
    __inject__ = ['mask_target_generator', 'mask_post_process']

    def __init__(self,
                 mask_target_generator=MaskTargetGenerator().__dict__,
                 mask_post_process=MaskPostProcess().__dict__):
        super(Mask, self).__init__()
        self.mask_target_generator = mask_target_generator
        self.mask_post_process = mask_post_process
        if isinstance(mask_target_generator, dict):
            self.mask_target_generator = MaskTargetGenerator(
                **mask_target_generator)
        if isinstance(mask_post_process, dict):
            self.mask_post_process = MaskPostProcess(**mask_post_process)

    def __call__(self, inputs):
        outs = {}
        if inputs['mode'] == 'train':
            outs = self.generate_mask_target(inputs)
        return outs

    def generate_mask_target(self, inputs):
        proposal_out = inputs['proposal_' + str(inputs['stage'])]
        outs = self.mask_target_generator(
            im_info=inputs['im_info'],
            gt_classes=inputs['gt_class'],
            is_crowd=inputs['is_crowd'],
            gt_segms=inputs['gt_mask'],
            rois=proposal_out['rois'],
            rois_nums=proposal_out['rois_nums'],
            labels_int32=proposal_out['labels_int32'])
        outs = {
            'mask_rois': outs[0],
            'rois_has_mask_int32': outs[1],
            'mask_int32': outs[2]
        }
        return outs

    def post_process(self, inputs):
        outs = self.mask_post_process(inputs)
        return outs
