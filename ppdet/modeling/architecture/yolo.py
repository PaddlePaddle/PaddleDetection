from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid

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

    def __init__(self, anchor, backbone, yolo_head, mode='train'):
        super(YOLOv3, self).__init__()
        self.anchor = anchor
        self.backbone = backbone
        self.yolo_head = yolo_head
        self.mode = mode

    def forward(self, inputs, inputs_keys):
        self.gbd = self.build_inputs(inputs, inputs_keys)
        self.gbd['mode'] = self.mode

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

        # result  
        if self.gbd['mode'] == 'train':
            return self.loss(self.gbd)
        elif self.gbd['mode'] == 'infer':
            return self.infer(self.gbd)
        else:
            raise "Now, only support train or infer mode!"

    def loss(self, inputs):
        yolo_loss = self.yolo_head.loss(inputs)
        out = {'loss': yolo_loss, }
        return out

    def infer(self, inputs):
        outs = {
            "bbox": inputs['predicted_bbox'].numpy(),
            "bbox_nums": inputs['predicted_bbox_nums']
        }
        print(outs['bbox_nums'])
        return outs
