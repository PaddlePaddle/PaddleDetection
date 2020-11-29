from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['SSD']


@register
class SSD(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['backbone', 'neck', 'ssd_head', 'post_process']

    def __init__(self, backbone, ssd_head, post_process, neck=None):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.ssd_head = ssd_head
        self.post_process = post_process

    def model_arch(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        if self.neck is not None:
            body_feats, spatial_scale = self.neck(body_feats)

        # SSD Head
        self.ssd_head_outs = self.ssd_head(body_feats)

    def get_loss(self, ):
        loss = self.ssd_head.get_loss(self.inputs, self.ssd_head_outs)
        return loss

    def get_pred(self, ):
        output = {}
        # mask = self.mask_post_process(self.bboxes, self.mask_head_out,
        #                               self.inputs['im_info'])
        # bbox, bbox_num = self.bboxes
        # output = {
        #     'bbox': bbox.numpy(),
        #     'bbox_num': bbox_num.numpy(),
        #     'im_id': self.inputs['im_id'].numpy()
        # }
        # output.update(mask)
        return output
