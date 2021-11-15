from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['PAA']


@register
class PAA(BaseArch):
    """
    Args:
        backbone (object): backbone instance
        bbox_head (object): `BBoxHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        neck (object): 'FPN' instance
    """
    __category__ = 'architecture'
    __inject__ = ['bbox_post_process']

    def __init__(self,
                 backbone,
                 bbox_head,
                 bbox_post_process,
                 neck=None):
        super(PAA, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        bbox_head = create(cfg['bbox_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "bbox_head": bbox_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        if self.training:
            bbox_loss = self.bbox_head(body_feats, self.inputs)
            return bbox_loss
        else:
            preds = self.bbox_head(body_feats, None)

            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bbox, bbox_num = self.bbox_post_process(preds,
                                                    im_shape, scale_factor)

            # rescale the prediction back to origin image
            bbox_pred = self.bbox_post_process.get_pred(bbox, bbox_num,
                                                        im_shape, scale_factor)
            return bbox_pred, bbox_num

    def get_loss(self, ):
        bbox_loss = self._forward()
        loss = {}
        loss.update(bbox_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
