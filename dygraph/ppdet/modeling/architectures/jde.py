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
                 post_process='BBoxPostProcess',
                 test_emb=False):
        super(JDE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.jde_head = jde_head
        self.post_process = post_process
        self.test_emb = test_emb

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

        if self.training:
            jde_losses = self.jde_head(yolo_feats, identify_feats, self.inputs)
            return jde_losses
        else:
            if self.test_emb:
                embs_and_gts = self.jde_head(yolo_feats, identify_feats,
                                             self.inputs, self.test_emb)
                return embs_and_gts
            else:
                yolo_outs = self.jde_head(yolo_feats, identify_feats)
                bbox, bbox_num = self.post_process(
                    yolo_outs, self.jde_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])
                return bbox, bbox_num

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        if self.test_emb:
            emb_and_gt = self._forward()
            return emb_and_gt
        else:
            bbox_pred, bbox_num = self._forward()
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
            return output
