from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle

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
                 test_track=False,
                 test_emb=False):
        super(JDE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.jde_head = jde_head
        self.post_process = post_process
        self.test_emb = test_emb
        self.test_track = test_track

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
                embs_and_gts = self.jde_head(
                    yolo_feats,
                    identify_feats,
                    targets=self.inputs,
                    test_emb=True)
                return embs_and_gts

            elif self.test_track:
                yolo_outs, emb_outs = self.jde_head(
                    yolo_feats, identify_feats, test_track=True)

                bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_outs, self.jde_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])

                nms_keep_idx.stop_gradient = True
                embeding = paddle.gather_nd(emb_outs, nms_keep_idx)

                emb_det_results = {
                    'bbox': bbox,
                    'bbox_num': bbox_num,
                    'img0_shape': self.inputs['img0_shape'],
                    'embeding': embeding
                }
                return emb_det_results

            else:
                yolo_outs = self.jde_head(yolo_feats, identify_feats)
                bbox_pred, bbox_num, _ = self.post_process(
                    yolo_outs, self.jde_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])
                det_results = {'bbox': bbox_pred, 'bbox_num': bbox_num}
                return det_results

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
