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

import paddle
import paddle.nn as nn

from .meta_arch import BaseArch
from ppdet.core.workspace import register, create
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

from ..embedder.clip_utils import build_text_embedding_coco, read_clip_feat

__all__ = ['DETR', 'OVDETR']


@register
class DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()


@register
class OVDETR(DETR):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['exclude_post_process']

    def __init__(
            self,
            backbone,
            neck,
            transformer,
            detr_head,
            max_len=15,
            prob=0.5,
            with_box_refine=True,
            two_stage=True,
            bpe_path='ppdet://v1/paddledet/models/clip/bpe_simple_vocab_16e6.txt.gz',
            clip_path='',
            clip_feat_path='ppdet://v1/paddledet/data/coco/clip_feat_coco_pickle_label.pkl',
            post_process='OVDETRBBoxPostProcess',
            exclude_post_process=False):
        super(OVDETR, self).__init__(
            backbone=backbone,
            transformer=transformer,
            detr_head=detr_head,
            post_process=post_process,
            exclude_post_process=exclude_post_process)
        if neck is not None:
            self.neck = neck

        self.zeroshot_w = build_text_embedding_coco(bpe_path, clip_path).t()
        self.patch2query = nn.Linear(512, 256)
        self.patch2query_img = nn.Linear(512, 256)
        # mark 源码此处for layer in [self.patch2query]:
        xavier_uniform_(self.patch2query.weight)
        constant_(self.patch2query.bias, 0)

        self.all_ids = paddle.to_tensor(list(range(self.zeroshot_w.shape[-1])))
        self.max_len = max_len
        self.max_pad_len = self.max_len - 3

        # with open(clip_feat_path, 'rb') as f:
        #     self.clip_feat = pickle.load(f)
        self.clip_feat = read_clip_feat(clip_feat_path)
        self.prob = prob
        self.two_stage = two_stage
        if with_box_refine:
            self.transformer.decoder.bbox_head = self.detr_head.bbox_head
        else:
            self.transformer.decoder.bbox_head = None

        if two_stage:
            self.transformer.decoder.score_head = self.detr_head.score_head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)
        # transformer
        transformer = create(cfg['transformer'])

        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape,
            'two_stage': cfg['two_stage'],
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            'transformer': transformer,
            "detr_head": detr_head,
        }

    def _forward(self):
        if self.training:
            return self.forward_train()
        else:
            return self.forward_test()

    def forward_train(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        body_feats = self.neck(body_feats)

        pad_mask = self.inputs['pad_mask'] if self.training else None

        # clip_query
        if sum(len(a) for a in self.inputs["gt_class"]) > 0:
            uniq_labels = paddle.concat(self.inputs["gt_class"])
            uniq_labels = paddle.unique(uniq_labels)
            uniq_labels = uniq_labels[paddle.to_tensor(
                list(range(len(uniq_labels))))][:self.max_len]
            # uniq_labels = uniq_labels[paddle.randperm(len(uniq_labels))][: self.max_len]
        else:
            uniq_labels = paddle.to_tensor([])
        select_id = uniq_labels.tolist()

        if len(select_id) < self.max_pad_len:
            pad_len = self.max_pad_len - len(uniq_labels)
            extra_list = [i for i in self.all_ids if i not in uniq_labels]
            extra_list = paddle.to_tensor(extra_list)
            # extra_labels = extra_list[paddle.randperm(len(extra_list))][:pad_len].squeeze(1)
            extra_labels = extra_list[paddle.to_tensor(
                list(range(len(extra_list))))][:pad_len].squeeze(1)

            select_id += extra_labels.tolist()
        select_id_tensor = paddle.to_tensor(select_id)
        text_query = paddle.index_select(
            self.zeroshot_w, select_id_tensor, axis=1).t()

        img_query = []
        for cat_id in select_id:
            # index = paddle.randperm(len(self.clip_feat[cat_id]))[0:1]
            index = paddle.to_tensor(list(range(len(self.clip_feat[cat_id]))))[
                0:1]
            img_query.append(
                paddle.to_tensor(self.clip_feat[cat_id]).index_select(index))
        img_query = paddle.concat(img_query)
        img_query = img_query / paddle.linalg.norm(
            img_query, axis=1, keepdim=True)

        mask = (paddle.rand([len(text_query)]) < self.prob
                ).astype('float16').unsqueeze(1)
        clip_query_ori = (text_query * mask + img_query * (1 - mask)).detach()

        dtype = self.patch2query.weight.dtype
        text_query = self.patch2query(text_query.astype(dtype))
        img_query = self.patch2query_img(img_query.astype(dtype))
        clip_query = text_query * mask + img_query * (1 - mask)

        # Transformer
        head_inputs_dict = self.transformer(
            body_feats, pad_mask, text_query=clip_query)
        head_inputs_dict.update(
            dict(
                select_id=select_id,
                clip_query_ori=clip_query_ori, ))

        # DETR Head
        loss = self.detr_head(head_inputs_dict, body_feats, self.inputs)

        # paddle.device.cuda.empty_cache()

        return loss

    def forward_test(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        body_feats = self.neck(body_feats)

        pad_mask = self.inputs['pad_mask'] if self.training else None
        # out_transformer, clip_id, memory_feature = self.transformer(body_feats, pad_mask, self.inputs)

        # clip_query
        select_id = list(range(self.zeroshot_w.shape[-1]))
        num_patch = 15
        dtype = self.patch2query.weight.dtype
        preds_list = []
        bboxes_list = []
        logits_list = []

        for c in range(len(select_id) // num_patch + 1):
            clip_query = self.zeroshot_w[:, c * num_patch:(c + 1) *
                                         num_patch].t()
            clip_query_ori = clip_query
            clip_query = self.patch2query(clip_query.astype(dtype))

            # Transformer
            head_inputs_dict = self.transformer(
                body_feats, pad_mask, text_query=clip_query)
            head_inputs_dict.update(
                dict(
                    select_id=select_id,
                    clip_query_ori=clip_query_ori, ))

            # DETR Head
            preds = self.detr_head(head_inputs_dict, body_feats)
            bboxes, logits, masks = preds

            bboxes_list.append(bboxes)
            logits_list.append(logits)
            preds_list.append(preds)
        bboxes = paddle.concat(bboxes_list, -2).unsqueeze(0)
        logits = paddle.concat(logits_list, -2)

        if self.exclude_post_process:
            # bboxes, logits, masks = preds
            return bboxes, logits
        else:
            bbox, bbox_num = self.post_process(bboxes, logits, select_id,
                                               self.inputs['im_shape'],
                                               self.inputs['scale_factor'])
            # print(bbox)
            # exit()
            return bbox, bbox_num

    def get_loss(self):
        losses = self._forward()
        losses.update({
            'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
        })
        return losses

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output


def masked_fill(tensor, mask, value):
    cover = paddle.full_like(tensor, value)
    out = paddle.where(mask, cover, tensor)
    return out


def get_valid_ratio(mask):
    _, H, W = paddle.shape(mask)
    valid_ratio_h = paddle.sum(mask[:, :, 0], 1) / H
    valid_ratio_w = paddle.sum(mask[:, 0, :], 1) / W
    # [b, 2]
    return paddle.stack([valid_ratio_w, valid_ratio_h], -1)
