from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        self.ssd_head_outs, self.anchors = self.ssd_head(body_feats,
                                                         self.inputs['image'])

    def get_loss(self, ):
        loss = self.ssd_head.get_loss(self.ssd_head_outs, self.inputs,
                                      self.anchors)
        return {"loss": loss}

    def get_pred(self):
        bbox, bbox_num = self.post_process(self.ssd_head_outs, self.anchors,
                                           self.inputs['im_shape'],
                                           self.inputs['scale_factor'])
        outs = {
            "bbox": bbox,
            "bbox_num": bbox_num,
        }
        return outs
