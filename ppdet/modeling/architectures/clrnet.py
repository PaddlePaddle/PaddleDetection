from .meta_arch import BaseArch
from ppdet.core.workspace import register, create



__all__ = ['CLRNet']


@register
class CLRNet(BaseArch):
    __category__ = 'architecture'

    def __init__(self,
                 backbone="CLRResNet",
                 neck="CLRFPN",
                 clr_head="CLRHead",
                 post_process=None):
        super(CLRNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = clr_head
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
        clr_head = create(cfg['clr_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            'clr_head': clr_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs['image'])
        # neck
        neck_feats = self.neck(body_feats)
        # CRL Head

        if self.training:
            output = self.heads(neck_feats, self.inputs)
        else:
            output = self.heads(neck_feats)
            output = self.heads.get_lanes(output)
            output = {"lanes": output, "img_path": self.inputs['full_img_path'], "img_name": self.inputs['img_name']}
        
        return output
            
    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()