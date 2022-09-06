# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['SparseInst']


def _rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.astype(paddle.float32)
    return scores * (
        (masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


@register
class SparseInst(BaseArch):
    """
    SparseInst network, see https://arxiv.org/abs/2003.10152

    Args:
        backbone (object): an backbone instance
        solov2_head (object): an `SOLOv2Head` instance
        mask_head (object): an `SOLOv2MaskHead` instance
        neck (object): neck of network, such as feature pyramid network instance
    """

    __category__ = 'architecture'
    __inject__ = ['criterion']

    def __init__(self,
                 backbone,
                 encoder,
                 decoder,
                 criterion='SparseInstLoss',
                 cls_threshold=0.005,
                 mask_threshold=0.45):
        super(SparseInst, self).__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

        self.cls_threshold = cls_threshold
        self.mask_threshold = mask_threshold

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        encoder = create(cfg['encoder'], **kwargs)
        kwargs = {'input_shape': encoder.out_shape}
        decoder = create(cfg['decoder'], **kwargs)

        return {
            'backbone': backbone,
            'decoder': decoder,
            'encoder': encoder,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        body_feats = self.encoder(body_feats)
        raw_pred_out = self.decoder(body_feats)

        featmap_size = paddle.shape(raw_pred_out["pred_masks"])[-2:]
        input_shape = [featmap_size[0] * 4, featmap_size[1] * 4]

        if self.training:
            return self.criterion(raw_pred_out, self.inputs, input_shape)
        else:
            pred_masks = F.sigmoid(raw_pred_out["pred_masks"])
            pred_objectness = F.sigmoid(raw_pred_out["pred_scores"])
            pred_scores = F.sigmoid(raw_pred_out["pred_logits"])
            pred_scores = paddle.sqrt(pred_scores * pred_objectness)

            # currently, only batch_size=1 is supported in inference
            for idx in range(1):
                scores_per_image = pred_scores[idx]
                mask_pred_per_image = pred_masks[idx]
                im_shape = self.inputs['im_shape'][idx]
                scale_factor = self.inputs['scale_factor'][idx]

                origin_shape = paddle.round(im_shape /
                                            scale_factor).astype(paddle.int32)
                # max/argmax
                scores = scores_per_image.max(axis=-1)
                labels = scores_per_image.argmax(axis=-1)

                # cls threshold filter, adaptation for converting to static model
                keep = paddle.nonzero(
                    paddle.where(scores > self.cls_threshold, scores,
                                 paddle.zeros_like(scores))).squeeze(1)
                scores = paddle.gather(scores, keep)
                labels = paddle.gather(labels, keep)
                mask_pred_per_image = paddle.gather(mask_pred_per_image, keep)

                if scores.shape[0] == 0:
                    continue
                h = paddle.cast(im_shape[0], 'int32')[0]
                w = paddle.cast(im_shape[1], 'int32')[0]
                # rescoring mask using maskness
                scores = _rescoring_mask(
                    scores, mask_pred_per_image > self.mask_threshold,
                    mask_pred_per_image)

                mask_pred_per_image = F.interpolate(
                    mask_pred_per_image.unsqueeze(1),
                    size=input_shape,
                    mode="bilinear",
                    align_corners=False)
                mask_pred_per_image = paddle.slice(
                    mask_pred_per_image,
                    axes=[2, 3],
                    starts=[0, 0],
                    ends=[h, w])
                mask_pred_per_image = F.interpolate(
                    mask_pred_per_image,
                    size=origin_shape,
                    mode='bilinear',
                    align_corners=False).squeeze(1)

                bbox_num = paddle.shape(labels)[0]
                mask_pred = paddle.cast(
                    mask_pred_per_image > self.mask_threshold, 'uint8')
            return mask_pred, bbox_num, labels, scores

    def get_loss(self):
        loss = self._forward()
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        mask_pred, bbox_num, labels, scores = self._forward()
        result = {
            "segm": mask_pred,
            "bbox_num": bbox_num,
            "cate_label": labels,
            "cate_score": scores
        }

        return result
