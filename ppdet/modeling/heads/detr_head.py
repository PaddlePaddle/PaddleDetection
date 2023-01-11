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
import paddle.nn.functional as F
from ppdet.core.workspace import register
import pycocotools.mask as mask_util
from ..initializer import linear_init_, constant_
from ..transformers.utils import inverse_sigmoid

__all__ = ['DETRHead', 'DeformableDETRHead', 'DINOHead']


class MLP(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiHeadAttentionMap(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/segmentation.py

        This is a 2D attention module, which only returns the attention softmax (no multiplication by value)
    """

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0,
                 bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Constant()) if bias else False

        self.q_proj = nn.Linear(query_dim, hidden_dim, weight_attr, bias_attr)
        self.k_proj = nn.Conv2D(
            query_dim,
            hidden_dim,
            1,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

        self.normalize_fact = float(hidden_dim / self.num_heads)**-0.5

    def forward(self, q, k, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        bs, num_queries, n, c, h, w = q.shape[0], q.shape[1], self.num_heads,\
                                      self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1]
        qh = q.reshape([bs, num_queries, n, c])
        kh = k.reshape([bs, n, c, h, w])
        # weights = paddle.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
        qh = qh.transpose([0, 2, 1, 3]).reshape([-1, num_queries, c])
        kh = kh.reshape([-1, c, h * w])
        weights = paddle.bmm(qh * self.normalize_fact, kh).reshape(
            [bs, n, num_queries, h, w]).transpose([0, 2, 1, 3, 4])

        if mask is not None:
            weights += mask
        # fix a potenial bug: https://github.com/facebookresearch/detr/issues/247
        weights = F.softmax(weights.flatten(3), axis=-1).reshape(weights.shape)
        weights = self.dropout(weights)
        return weights


class MaskHeadFPNConv(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/segmentation.py

        Simple convolutional head, using group norm.
        Upsampling is done using a FPN approach
    """

    def __init__(self, input_dim, fpn_dims, context_dim, num_groups=8):
        super().__init__()

        inter_dims = [input_dim,
                      ] + [context_dim // (2**i) for i in range(1, 5)]
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Constant())

        self.conv0 = self._make_layers(input_dim, input_dim, 3, num_groups,
                                       weight_attr, bias_attr)
        self.conv_inter = nn.LayerList()
        for in_dims, out_dims in zip(inter_dims[:-1], inter_dims[1:]):
            self.conv_inter.append(
                self._make_layers(in_dims, out_dims, 3, num_groups, weight_attr,
                                  bias_attr))

        self.conv_out = nn.Conv2D(
            inter_dims[-1],
            1,
            3,
            padding=1,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

        self.adapter = nn.LayerList()
        for i in range(len(fpn_dims)):
            self.adapter.append(
                nn.Conv2D(
                    fpn_dims[i],
                    inter_dims[i + 1],
                    1,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr))

    def _make_layers(self,
                     in_dims,
                     out_dims,
                     kernel_size,
                     num_groups,
                     weight_attr=None,
                     bias_attr=None):
        return nn.Sequential(
            nn.Conv2D(
                in_dims,
                out_dims,
                kernel_size,
                padding=kernel_size // 2,
                weight_attr=weight_attr,
                bias_attr=bias_attr),
            nn.GroupNorm(num_groups, out_dims),
            nn.ReLU())

    def forward(self, x, bbox_attention_map, fpns):
        x = paddle.concat([
            x.tile([bbox_attention_map.shape[1], 1, 1, 1]),
            bbox_attention_map.flatten(0, 1)
        ], 1)
        x = self.conv0(x)
        for inter_layer, adapter_layer, feat in zip(self.conv_inter[:-1],
                                                    self.adapter, fpns):
            feat = adapter_layer(feat).tile(
                [bbox_attention_map.shape[1], 1, 1, 1])
            x = inter_layer(x)
            x = feat + F.interpolate(x, size=feat.shape[-2:])

        x = self.conv_inter[-1](x)
        x = self.conv_out(x)
        return x


@register
class DETRHead(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'use_focal_loss']
    __inject__ = ['loss']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 nhead=8,
                 num_mlp_layers=3,
                 loss='DETRLoss',
                 fpn_dims=[1024, 512, 256],
                 with_mask_head=False,
                 use_focal_loss=False):
        super(DETRHead, self).__init__()
        # add background class
        self.num_classes = num_classes if use_focal_loss else num_classes + 1
        self.hidden_dim = hidden_dim
        self.loss = loss
        self.with_mask_head = with_mask_head
        self.use_focal_loss = use_focal_loss

        self.score_head = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_head = MLP(hidden_dim,
                             hidden_dim,
                             output_dim=4,
                             num_layers=num_mlp_layers)
        if self.with_mask_head:
            self.bbox_attention = MultiHeadAttentionMap(hidden_dim, hidden_dim,
                                                        nhead)
            self.mask_head = MaskHeadFPNConv(hidden_dim + nhead, fpn_dims,
                                             hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.score_head)

    @classmethod
    def from_config(cls, cfg, hidden_dim, nhead, input_shape):

        return {
            'hidden_dim': hidden_dim,
            'nhead': nhead,
            'fpn_dims': [i.channels for i in input_shape[::-1]][1:]
        }

    @staticmethod
    def get_gt_mask_from_polygons(gt_poly, pad_mask):
        out_gt_mask = []
        for polygons, padding in zip(gt_poly, pad_mask):
            height, width = int(padding[:, 0].sum()), int(padding[0, :].sum())
            masks = []
            for obj_poly in polygons:
                rles = mask_util.frPyObjects(obj_poly, height, width)
                rle = mask_util.merge(rles)
                masks.append(
                    paddle.to_tensor(mask_util.decode(rle)).astype('float32'))
            masks = paddle.stack(masks)
            masks_pad = paddle.zeros(
                [masks.shape[0], pad_mask.shape[1], pad_mask.shape[2]])
            masks_pad[:, :height, :width] = masks
            out_gt_mask.append(masks_pad)
        return out_gt_mask

    def forward(self, out_transformer, body_feats, inputs=None):
        r"""
        Args:
            out_transformer (Tuple): (feats: [num_levels, batch_size,
                                                num_queries, hidden_dim],
                            memory: [batch_size, hidden_dim, h, w],
                            src_proj: [batch_size, h*w, hidden_dim],
                            src_mask: [batch_size, 1, 1, h, w])
            body_feats (List(Tensor)): list[[B, C, H, W]]
            inputs (dict): dict(inputs)
        """
        feats, memory, src_proj, src_mask = out_transformer
        outputs_logit = self.score_head(feats)
        outputs_bbox = F.sigmoid(self.bbox_head(feats))
        outputs_seg = None
        if self.with_mask_head:
            bbox_attention_map = self.bbox_attention(feats[-1], memory,
                                                     src_mask)
            fpn_feats = [a for a in body_feats[::-1]][1:]
            outputs_seg = self.mask_head(src_proj, bbox_attention_map,
                                         fpn_feats)
            outputs_seg = outputs_seg.reshape([
                feats.shape[1], feats.shape[2], outputs_seg.shape[-2],
                outputs_seg.shape[-1]
            ])

        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
            gt_mask = self.get_gt_mask_from_polygons(
                inputs['gt_poly'],
                inputs['pad_mask']) if 'gt_poly' in inputs else None
            return self.loss(
                outputs_bbox,
                outputs_logit,
                inputs['gt_bbox'],
                inputs['gt_class'],
                masks=outputs_seg,
                gt_mask=gt_mask)
        else:
            return (outputs_bbox[-1], outputs_logit[-1], outputs_seg)


@register
class DeformableDETRHead(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim']
    __inject__ = ['loss']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=512,
                 nhead=8,
                 num_mlp_layers=3,
                 loss='DETRLoss'):
        super(DeformableDETRHead, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.loss = loss

        self.score_head = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_head = MLP(hidden_dim,
                             hidden_dim,
                             output_dim=4,
                             num_layers=num_mlp_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.score_head)
        constant_(self.score_head.bias, -4.595)
        constant_(self.bbox_head.layers[-1].weight)

        with paddle.no_grad():
            bias = paddle.zeros_like(self.bbox_head.layers[-1].bias)
            bias[2:] = -2.0
            self.bbox_head.layers[-1].bias.set_value(bias)

    @classmethod
    def from_config(cls, cfg, hidden_dim, nhead, input_shape):
        return {'hidden_dim': hidden_dim, 'nhead': nhead}

    def forward(self, out_transformer, body_feats, inputs=None):
        r"""
        Args:
            out_transformer (Tuple): (feats: [num_levels, batch_size,
                                                num_queries, hidden_dim],
                            memory: [batch_size,
                                \sum_{l=0}^{L-1} H_l \cdot W_l, hidden_dim],
                            reference_points: [batch_size, num_queries, 2])
            body_feats (List(Tensor)): list[[B, C, H, W]]
            inputs (dict): dict(inputs)
        """
        feats, memory, reference_points = out_transformer
        reference_points = inverse_sigmoid(reference_points.unsqueeze(0))
        outputs_bbox = self.bbox_head(feats)

        # It's equivalent to "outputs_bbox[:, :, :, :2] += reference_points",
        # but the gradient is wrong in paddle.
        outputs_bbox = paddle.concat(
            [
                outputs_bbox[:, :, :, :2] + reference_points,
                outputs_bbox[:, :, :, 2:]
            ],
            axis=-1)

        outputs_bbox = F.sigmoid(outputs_bbox)
        outputs_logit = self.score_head(feats)

        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs

            return self.loss(outputs_bbox, outputs_logit, inputs['gt_bbox'],
                             inputs['gt_class'])
        else:
            return (outputs_bbox[-1], outputs_logit[-1], None)


@register
class DINOHead(nn.Layer):
    __inject__ = ['loss']

    def __init__(self, loss='DINOLoss'):
        super(DINOHead, self).__init__()
        self.loss = loss

    def forward(self, out_transformer, body_feats, inputs=None):
        (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
         dn_meta) = out_transformer
        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs

            if dn_meta is not None:
                dn_out_bboxes, dec_out_bboxes = paddle.split(
                    dec_out_bboxes, dn_meta['dn_num_split'], axis=2)
                dn_out_logits, dec_out_logits = paddle.split(
                    dec_out_logits, dn_meta['dn_num_split'], axis=2)
            else:
                dn_out_bboxes, dn_out_logits = None, None

            out_bboxes = paddle.concat(
                [enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
            out_logits = paddle.concat(
                [enc_topk_logits.unsqueeze(0), dec_out_logits])

            return self.loss(
                out_bboxes,
                out_logits,
                inputs['gt_bbox'],
                inputs['gt_class'],
                dn_out_bboxes=dn_out_bboxes,
                dn_out_logits=dn_out_logits,
                dn_meta=dn_meta)
        else:
            return (dec_out_bboxes[-1], dec_out_logits[-1], None)
