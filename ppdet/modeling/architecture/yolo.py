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
        'backbone',
        'neck',
        'yolo_head',
        'post_process',
    ]

    def __init__(self,
                 backbone='DarkNet',
                 neck='YOLOv3FPN',
                 yolo_head='YOLOv3Head',
                 post_process='BBoxPostProcess'):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process

    def model_arch(self, ):
        # Backbone
        blocks = self.backbone(self.inputs)

        # neck
        feats = self.neck(blocks)

        # YOLO Head
        self.yolo_head_outs = self.yolo_head(feats)

    def loss(self, ):
        yolo_loss = self.yolo_head.loss(self.inputs, self.yolo_head_outs)
        return yolo_loss

    def infer(self, ):
        bbox, bbox_num = self.post_process(
            self.yolo_head_outs, self.yolo_head.mask_anchors, self.inputs['im_size'])
        outs = {
            "bbox": bbox.numpy(),
            "bbox_num": bbox_num.numpy(),
            'im_id': self.inputs['im_id'].numpy()
        }
        return outs
