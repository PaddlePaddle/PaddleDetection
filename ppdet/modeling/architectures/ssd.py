# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle
import paddle.nn.functional as F

__all__ = ['SSD']


@register
class SSD(BaseArch):
    """
    Single Shot MultiBox Detector, see https://arxiv.org/abs/1512.02325

    Args:
        backbone (nn.Layer): backbone instance
        ssd_head (nn.Layer): `SSDHead` instance
        post_process (object): `BBoxPostProcess` instance
    """

    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self, backbone, ssd_head, post_process, r34_backbone=False):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.ssd_head = ssd_head
        self.post_process = post_process
        self.r34_backbone = r34_backbone
        if self.r34_backbone:
            from ppdet.modeling.backbones.resnet import ResNet
            assert isinstance(self.backbone, ResNet) and \
                   self.backbone.depth == 34, \
                "If you set r34_backbone=True, please use ResNet-34 as backbone."
            self.backbone.res_layers[2].blocks[0].branch2a.conv._stride = [1, 1]
            self.backbone.res_layers[2].blocks[0].short.conv._stride = [1, 1]

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # head
        kwargs = {'input_shape': backbone.out_shape}
        ssd_head = create(cfg['ssd_head'], **kwargs)

        return {
            'backbone': backbone,
            "ssd_head": ssd_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # SSD Head
        if self.training:
            return self.ssd_head(body_feats, self.inputs['image'],
                                 self.inputs['gt_bbox'],
                                 self.inputs['gt_class'])
        else:
            preds, anchors = self.ssd_head(body_feats, self.inputs['image'])
            bbox, bbox_num, nms_keep_idx = self.post_process(
                preds, anchors, self.inputs['im_shape'],
                self.inputs['scale_factor'])

            if self.use_extra_data:
                extra_data = {}  # record the bbox output before nms, such like scores and nms_keep_idx
                """extra_data:{
                            'scores': predict scores,
                            'nms_keep_idx': bbox index before nms,
                           }
                           """
                preds_logits = preds[1]  # [[1xNumBBoxNumClass]]
                extra_data['scores'] = F.softmax(paddle.concat(
                    preds_logits, axis=1)).transpose([0, 2, 1])
                extra_data['logits'] = paddle.concat(
                    preds_logits, axis=1).transpose([0, 2, 1])
                extra_data['nms_keep_idx'] = nms_keep_idx  # bbox index before nms
                return bbox, bbox_num, extra_data
            else:
                return bbox, bbox_num

    def get_loss(self, ):
        return {"loss": self._forward()}

    def get_pred(self):
        if self.use_extra_data:
            bbox_pred, bbox_num, extra_data = self._forward()
            output = {
                "bbox": bbox_pred,
                "bbox_num": bbox_num,
                "extra_data": extra_data
            }
        else:
            bbox_pred, bbox_num = self._forward()
            output = {
                "bbox": bbox_pred,
                "bbox_num": bbox_num,
            }
        return output
