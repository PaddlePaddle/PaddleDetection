from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['JDE']


@register
class JDE(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DarkNet',
                 neck='JDEFPN',
                 jde_head='JDEHead',
                 post_process='BBoxPostProcess'):
        super(JDE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.jde_head = jde_head
        self.post_process = post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        jde_head = create(cfg['jde_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "jde_head": jde_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        yolo_feats, identify_feats = self.neck(body_feats)
        # self.det_outs, self.ide_outs = self.jde_head(yolo_feats, identify_feats)
        det_outs, ide_outs = self.jde_head(yolo_feats, identify_feats)

        if self.training:
            return det_outs, ide_outs
        else:
            jde_head_outs = self.jde_head.get_outputs(det_outs, ide_outs)
            bbox, bbox_num = self.post_process(
                jde_head_outs, self.jde_head.mask_anchors,
                self.inputs['im_shape'], self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self):
        det_outs, ide_outs = self._forward()
        loss = self.jde_head.get_loss(det_outs, ide_outs, self.inputs)
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        label = bbox_pred[:, 0]
        score = bbox_pred[:, 1]
        bbox = bbox_pred[:, 2:]
        output = {
            'bbox': bbox,
            'score': score,
            'label': label,
            'bbox_num': bbox_num
        }
        return output
