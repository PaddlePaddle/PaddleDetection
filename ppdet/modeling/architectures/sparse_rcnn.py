# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ["SparseRCNN"]


@register
class SparseRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = ["postprocess"]

    def __init__(self,
                 backbone,
                 neck,
                 head="SparsercnnHead",
                 postprocess="SparsePostProcess"):
        super(SparseRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.postprocess = postprocess

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'roi_input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        head_outs = self.head(fpn_feats, self.inputs["img_whwh"])

        if not self.training:
            bbox_pred, bbox_num = self.postprocess(
                head_outs["pred_logits"], head_outs["pred_boxes"],
                self.inputs["scale_factor_whwh"], self.inputs["ori_shape"])
            return bbox_pred, bbox_num
        else:
            return head_outs

    def get_loss(self):
        batch_gt_class = self.inputs["gt_class"]
        batch_gt_box = self.inputs["gt_bbox"]
        batch_whwh = self.inputs["img_whwh"]
        targets = []

        for i in range(len(batch_gt_class)):
            boxes = batch_gt_box[i]
            labels = batch_gt_class[i].squeeze(-1)
            img_whwh = batch_whwh[i]
            img_whwh_tgt = img_whwh.unsqueeze(0).tile([int(boxes.shape[0]), 1])
            targets.append({
                "boxes": boxes,
                "labels": labels,
                "img_whwh": img_whwh,
                "img_whwh_tgt": img_whwh_tgt
            })

        outputs = self._forward()
        loss_dict = self.head.get_loss(outputs, targets)
        acc = loss_dict["acc"]
        loss_dict.pop("acc")
        total_loss = sum(loss_dict.values())
        loss_dict.update({"loss": total_loss, "acc": acc})
        return loss_dict

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
