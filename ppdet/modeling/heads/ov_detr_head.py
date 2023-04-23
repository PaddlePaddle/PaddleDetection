# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling.transformers import _get_clones
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_
from ppdet.modeling.transformers.utils import inverse_sigmoid
import math

__all__ = ['OVDETRHead']


class MLP(paddle.nn.Layer):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = paddle.nn.LayerList(sublayers=(paddle.nn.Linear(
            in_features=n, out_features=k) for n, k in zip([input_dim] + h,
                                                           h + [output_dim])))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = paddle.nn.functional.relu(
                x=layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@register
class OVDETRHead(nn.Layer):
    __inject__ = ['transformer', 'loss']

    def __init__(self,
                 transformer=None,
                 num_classes=80,
                 num_queries=300,
                 num_layers=6,
                 aux_loss=True,
                 with_box_refine=False,
                 two_stage=False,
                 cls_out_channels=91,
                 dataset_file="coco",
                 max_len=15,
                 clip_feat_path=None,
                 prob=0.5,
                 loss=''):
        super(OVDETRHead, self).__init__()

        self.num_queries = num_queries
        self.two_stage = two_stage
        self.transformer = transformer
        hidden_dim = self.transformer.hidden_dim
        self.class_embed = paddle.nn.Linear(
            in_features=hidden_dim, out_features=cls_out_channels)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.patch2query = paddle.nn.Linear(in_features=512, out_features=256)
        self.patch2query_img = paddle.nn.Linear(
            in_features=512, out_features=256)
        xavier_uniform_(self.patch2query.weight)
        constant_(self.patch2query.bias, 0)

        self.feature_align = paddle.nn.Linear(in_features=256, out_features=512)
        xavier_uniform_(self.feature_align.weight)
        constant_(self.feature_align.bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.feature_align = _get_clones(self.feature_align, num_pred)
        else:
            self.feature_align = paddle.nn.LayerList(
                sublayers=[self.feature_align for _ in range(num_pred)])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        constant_(self.class_embed.bias, bias_value)
        constant_(self.bbox_embed.layers[-1].weight, 0)
        constant_(self.bbox_embed.layers[-1].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1
                    if two_stage else transformer.decoder.num_layers)
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            constant_(self.bbox_embed[0].layers[-1].bias[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            constant_(self.bbox_embed.layers[-1].bias[2:], -2.0)
            self.class_embed = paddle.nn.LayerList(
                sublayers=[self.class_embed for _ in range(num_pred)])
            self.bbox_embed = paddle.nn.LayerList(
                sublayers=[self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                constant_(box_embed.layers[-1].bias[2:], 0.0)

        self.max_len = max_len
        self.max_pad_len = max_len - 3
        self.aux_loss = aux_loss

        if clip_feat_path is not None:
            self.clip_feat = paddle.load(clip_feat_path)
        self.prob = prob
        self.loss = loss

    def get_outputs_class(self, layer, data):
        return layer(data)

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{
            "pred_logits": a,
            "pred_boxes": b
        } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(self, feat, mask, zeroshot_w, gt_class=None, gt_bbox=None):
        if self.training:
            return self.forward_train(gt_class, gt_bbox, feat, mask, zeroshot_w)
        else:
            return self.forward_test(feat, mask, zeroshot_w)

    def forward_train(self, gt_class, gt_bbox, feat, mask, zeroshot_w):

        zeroshot_w = zeroshot_w.t()
        uniq_labels = paddle.concat([c.t() for c in gt_class], axis=1)
        uniq_labels = paddle.unique(uniq_labels)
        index = paddle.randperm(uniq_labels.shape[-1])
        uniq_labels = uniq_labels[index][:self.max_len]
        select_id = uniq_labels.tolist()
        all_ids = paddle.arange(end=zeroshot_w.shape[-1])

        if len(select_id) < self.max_pad_len:
            pad_len = self.max_pad_len - len(uniq_labels)
            extra_list = paddle.to_tensor(
                [i for i in all_ids if i not in uniq_labels])
            extra_labels = extra_list[paddle.randperm(len(extra_list))][:
                                                                        pad_len]
            select_id += extra_labels.squeeze(axis=1).tolist()

        text_query = paddle.index_select(
            zeroshot_w, paddle.to_tensor(select_id), axis=1).t()
        img_query = []

        for cat_id in select_id:
            index = paddle.randperm(len(self.clip_feat[cat_id]))[0:1]
            img_query.append(self.clip_feat[cat_id][index])

        img_query = paddle.stack(img_query).astype(text_query.dtype)
        img_query = img_query / paddle.linalg.norm(
            img_query, axis=-1, keepdim=True)

        query_mask = paddle.rand(shape=[len(text_query)]) < self.prob
        query_mask = query_mask.astype('float32').unsqueeze(axis=1)
        clip_query_ori = (text_query * query_mask + img_query *
                          (1 - query_mask)).detach()

        text_query = self.patch2query(text_query)
        img_query = self.patch2query_img(img_query)
        clip_query = text_query * query_mask + img_query * (1 - query_mask)

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact, ), memory_features = self.transformer(
                feat, mask, text_query=clip_query)

        outputs_classes = []
        outputs_coords = []
        outputs_embeds = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.get_outputs_class(self.class_embed[lvl],
                                                   hs[lvl])

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = F.sigmoid(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_embeds.append(self.feature_align[lvl](hs[lvl]))

        outputs_class = paddle.stack(x=outputs_classes)
        outputs_coord = paddle.stack(x=outputs_coords)
        outputs_embed = paddle.stack(x=outputs_embeds)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_embed": outputs_embed[-1],
            "select_id": select_id,
            "clip_query": clip_query_ori,
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class,
                                                    outputs_coord)

            for temp, embed in zip(out["aux_outputs"], outputs_embed[:-1]):
                temp["select_id"] = select_id
                temp["pred_embed"] = embed
                temp["clip_query"] = clip_query_ori

        if self.two_stage:
            enc_outputs_coord = F.sigmoid(enc_outputs_coord_unact)
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "select_id": select_id,
            }

        return self.loss(out, gt_bbox, gt_class, masks=mask)

    def forward_test(self, feat, mask, zeroshot_w):

        zeroshot_w = zeroshot_w.t()
        select_id = list(range(zeroshot_w.shape[-1]))
        num_patch = 15

        outputs_class_list = []
        outputs_coord_list = []
        for c in range(len(select_id) // num_patch + 1):
            clip_query = zeroshot_w[:, c * num_patch:(c + 1) * num_patch].t()
            clip_query = self.patch2query(clip_query)
            (
                hs,
                init_reference,
                inter_references,
                enc_outputs_class,
                enc_outputs_coord_unact, ), memory_features = self.transformer(
                    feat, mask, text_query=clip_query)

            outputs_classes = []
            outputs_coords = []

            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)

                outputs_class = self.get_outputs_class(self.class_embed[lvl],
                                                       hs[lvl])

                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference

                outputs_coord = F.sigmoid(tmp)
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = paddle.stack(outputs_classes)
            outputs_coord = paddle.stack(outputs_coords)
            outputs_class_list.append(outputs_class)
            outputs_coord_list.append(outputs_coord)
        outputs_class = paddle.concat(x=outputs_class_list, axis=-2)
        outputs_coord = paddle.concat(x=outputs_coord_list, axis=-2)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1]
        }
        out["select_id"] = select_id

        return out
