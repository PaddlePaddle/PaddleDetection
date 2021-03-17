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
                 detection_head='YOLOv3Head',
                 post_process='JDEBBoxPostProcess',
                 embedding_head='JEDEmbeddingHead',
                 tracker='JDETracker',
                 test_emb=False,
                 test_track=False):
        super(JDE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.detection_head = detection_head
        self.post_process = post_process
        self.embedding_head = embedding_head
        self.tracker = tracker
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
        detection_head = create(cfg['detection_head'], **kwargs)
        embedding_head = create(cfg['embedding_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "detection_head": detection_head,
            "embedding_head": embedding_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        det_feats, emb_feats = self.neck(body_feats)

        if self.training:
            loss_confs, loss_boxes = self.detection_head(det_feats, self.inputs)
            jde_losses = self.embedding_head(emb_feats, self.inputs, loss_confs,
                                             loss_boxes)
            return jde_losses
        else:
            if self.test_emb:
                embs_and_gts = self.embedding_head(
                    emb_feats, self.inputs, test_emb=True)
                return embs_and_gts

            elif self.test_track:
                det_outs = self.detection_head(det_feats, self.inputs)
                emb_outs = self.embedding_head(emb_feats, self.inputs)

                bbox, bbox_num, nms_keep_idx = self.post_process(
                    det_outs, self.detection_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])

                nms_keep_idx.stop_gradient = True
                embeddings = paddle.gather_nd(emb_outs, nms_keep_idx)

                dets_and_embs = {
                    'bbox': bbox,
                    'bbox_num': bbox_num,
                    'img0_shape': self.inputs['img0_shape'],
                    'embedding': embeddings,
                }

                return dets_and_embs

            else:
                det_outs = self.detection_head(det_feats)

                bbox_pred, bbox_num, _ = self.post_process(
                    det_outs, self.detection_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])

                det_results = {'bbox': bbox_pred, 'bbox_num': bbox_num}
                return det_results

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
