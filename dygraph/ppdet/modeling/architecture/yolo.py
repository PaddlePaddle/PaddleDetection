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
        body_feats = self.backbone(self.inputs)

        # neck
        body_feats = self.neck(body_feats)

        # YOLO Head
        self.yolo_head_outs = self.yolo_head(body_feats)

    def get_loss(self, ):
        loss = self.yolo_head.get_loss(self.yolo_head_outs, self.inputs)
        return loss

    def get_pred(self):
        bbox, bbox_num = self.post_process(
            self.yolo_head_outs, self.yolo_head.mask_anchors,
            self.inputs['im_shape'], self.inputs['scale_factor'])
        outs = {
            "bbox": bbox,
            "bbox_num": bbox_num,
        }
        return outs
