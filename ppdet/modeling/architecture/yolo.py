from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['YOLOv3']


@register
class YOLOv3(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'anchor',
        'backbone',
        'yolo_head',
    ]

    def __init__(self, anchor, backbone, yolo_head):
        super(YOLOv3, self).__init__()
        self.anchor = anchor
        self.backbone = backbone
        self.yolo_head = yolo_head

    def model_arch(self, ):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # YOLO Head
        self.yolo_head_out = self.yolo_head(body_feats)

        # Anchor
        self.anchors, self.anchor_masks, self.mask_anchors = self.anchor()

    def loss(self, ):
        yolo_loss = self.yolo_head.loss(self.inputs, self.yolo_head_out,
                                        self.anchors, self.anchor_masks,
                                        self.mask_anchors)
        return yolo_loss

    def infer(self, ):
        bbox, bbox_num = self.anchor.post_process(
            self.inputs['im_size'], self.yolo_head_out, self.mask_anchors)
        outs = {
            "bbox": bbox.numpy(),
            "bbox_num": bbox_num,
            'im_id': self.inputs['im_id'].numpy()
        }
        return outs
