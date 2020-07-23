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

    def __init__(self, anchor, backbone, yolo_head, *args, **kwargs):
        super(YOLOv3, self).__init__(*args, **kwargs)
        self.anchor = anchor
        self.backbone = backbone
        self.yolo_head = yolo_head

    def model_arch(self, ):
        # Backbone
        bb_out = self.backbone(self.gbd)
        self.gbd.update(bb_out)

        # YOLO Head
        yolo_head_out = self.yolo_head(self.gbd)
        self.gbd.update(yolo_head_out)

        # Anchor
        anchor_out = self.anchor(self.gbd)
        self.gbd.update(anchor_out)

        if self.gbd['mode'] == 'infer':
            bbox_out = self.anchor.post_process(self.gbd)
            self.gbd.update(bbox_out)

    def loss(self, ):
        yolo_loss = self.yolo_head.loss(self.gbd)
        out = {'loss': yolo_loss}
        return out

    def infer(self, ):
        outs = {
            "bbox": self.gbd['predicted_bbox'].numpy(),
            "bbox_nums": self.gbd['predicted_bbox_nums'],
            'im_id': self.gbd['im_id'].numpy()
        }
        return outs
