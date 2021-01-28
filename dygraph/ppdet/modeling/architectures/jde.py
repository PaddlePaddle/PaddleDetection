from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['JDE']


@register
class JDE(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'backbone',
        'neck',
        'jde_head',
        'post_process',
    ]

    def __init__(self,
                 backbone='DarkNet',
                 neck='YOLOv3FPN',
                 jde_head='YOLOv3Head',
                 post_process='BBoxPostProcess'):
        super(JDE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.jde_head = jde_head
        self.post_process = post_process

    def model_arch(self, ):
        body_feats = self.backbone(self.inputs)
        body_feats = self.neck(body_feats)
        self.jde_head_outs = self.jde_head(body_feats)

    def get_loss(self, ):
        loss = self.jde_head.get_loss(self.jde_head_outs, self.inputs)
        return loss

    def get_pred(self):
        jde_head_outs = self.jde_head.get_outputs(self.jde_head_outs)
        bbox, bbox_num = self.post_process(
            jde_head_outs, self.jde_head.mask_anchors,
            self.inputs['im_shape'], self.inputs['scale_factor'])
        outs = {
            "bbox": bbox,
            "bbox_num": bbox_num,
        }
        return outs
