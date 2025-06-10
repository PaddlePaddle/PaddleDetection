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
from ppdet.core.workspace import register

__all__ = ['DFINEHead']


@register
class DFINEHead(nn.Layer):
    __inject__ = ['loss']

    def __init__(self, loss='DFINELoss', eval_idx=-1):
        super(DFINEHead, self).__init__()
        self.loss = loss
        self.eval_idx = eval_idx

    def forward(self, out_transformer, body_feats, inputs=None):
        (dec_out_bboxes, dec_out_logits,
         dec_out_corners, dec_out_refs, dec_pre_bboxes, dec_pre_logits,
         enc_topk_bboxes, enc_topk_logits,
         dn_meta) = out_transformer
        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs

            if dn_meta is not None:
                if isinstance(dn_meta, list):
                    dual_groups = len(dn_meta) - 1
                    dec_out_bboxes = paddle.split(
                        dec_out_bboxes, dual_groups + 1, axis=2)
                    dec_out_logits = paddle.split(
                        dec_out_logits, dual_groups + 1, axis=2)
                    enc_topk_bboxes = paddle.split(
                        enc_topk_bboxes, dual_groups + 1, axis=1)
                    enc_topk_logits = paddle.split(
                        enc_topk_logits, dual_groups + 1, axis=1)
                    dec_out_corners = paddle.split(
                        dec_out_corners, dual_groups + 1, axis=2)
                    dec_out_refs = paddle.split(
                        dec_out_refs, dual_groups + 1, axis=2)
                    dec_pre_bboxes = paddle.split(
                        dec_pre_bboxes, dual_groups + 1, axis=1)
                    dec_pre_logits = paddle.split(
                        dec_pre_logits, dual_groups + 1, axis=1)

                    loss = {}
                    for g_id in range(dual_groups + 1):
                        if dn_meta[g_id] is not None:
                            dn_out_bboxes_gid, dec_out_bboxes_gid = paddle.split(
                                dec_out_bboxes[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                            dn_out_logits_gid, dec_out_logits_gid = paddle.split(
                                dec_out_logits[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                            dn_out_corners_gid, dec_out_corners_gid = paddle.split(
                                dec_out_corners[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                            dn_out_refs_gid, dec_out_refs_gid = paddle.split(
                                dec_out_refs[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                            dn_pre_bboxes_gid, dec_pre_bboxes_gid = paddle.split(
                                dec_pre_bboxes[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=1)
                            dn_pre_logits_gid, dec_pre_logits_gid = paddle.split(
                                dec_pre_logits[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=1)
                        else:
                            dn_out_bboxes_gid, dn_out_logits_gid = None, None
                            dec_out_bboxes_gid = dec_out_bboxes[g_id]
                            dec_out_logits_gid = dec_out_logits[g_id]
                            dn_out_corners_gid, dn_out_refs_gid = None, None
                            dn_pre_bboxes_gid, dn_pre_logits_gid = None, None
                            dec_out_corners_gid = dec_out_corners[g_id]
                            dec_out_refs_gid = dec_out_refs[g_id]
                            dec_pre_bboxes_gid = dec_pre_bboxes[g_id]
                            dec_pre_logits_gid = dec_pre_logits[g_id]
                        out_bboxes_gid = paddle.concat([
                            enc_topk_bboxes[g_id].unsqueeze(0),
                            dec_out_bboxes_gid
                        ])
                        out_logits_gid = paddle.concat([
                            enc_topk_logits[g_id].unsqueeze(0),
                            dec_out_logits_gid
                        ])
                        loss_gid = self.loss(
                            out_bboxes_gid,
                            out_logits_gid,
                            dec_out_corners_gid,
                            dec_out_refs_gid,
                            dec_pre_bboxes_gid,
                            dec_pre_logits_gid,
                            inputs['gt_bbox'],
                            inputs['gt_class'],
                            dn_out_bboxes=dn_out_bboxes_gid,
                            dn_out_logits=dn_out_logits_gid,
                            dn_out_corners=dn_out_corners_gid,
                            dn_out_refs=dn_out_refs_gid,
                            dn_pre_bboxes=dn_pre_bboxes_gid,
                            dn_pre_logits=dn_pre_logits_gid,
                            dn_meta=dn_meta[g_id])
                        # sum loss
                        for key, value in loss_gid.items():
                            loss.update({
                                key: loss.get(key, paddle.zeros([1])) + value
                            })

                    # average across (dual_groups + 1)
                    for key, value in loss.items():
                        loss.update({key: value / (dual_groups + 1)})
                    return loss
                else:
                    dn_out_bboxes, dec_out_bboxes = paddle.split(
                        dec_out_bboxes, dn_meta['dn_num_split'], axis=2)
                    dn_out_logits, dec_out_logits = paddle.split(
                        dec_out_logits, dn_meta['dn_num_split'], axis=2)
                    dn_out_corners, dec_out_corners = paddle.split(
                        dec_out_corners, dn_meta['dn_num_split'], axis=2)
                    dn_out_refs, dec_out_refs = paddle.split(
                        dec_out_refs, dn_meta['dn_num_split'], axis=2)
                    dn_pre_bboxes, dec_pre_bboxes = paddle.split(
                        dec_pre_bboxes, dn_meta['dn_num_split'], axis=1)
                    dn_pre_logits, dec_pre_logits = paddle.split(
                        dec_pre_logits, dn_meta['dn_num_split'], axis=1)
            else:
                dn_out_bboxes, dn_out_logits = None, None
                dn_out_corners, dn_out_refs = None, None
                dn_pre_bboxes, dn_pre_logits = None, None

            out_bboxes = paddle.concat(
                [enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
            out_logits = paddle.concat(
                [enc_topk_logits.unsqueeze(0), dec_out_logits])

            return self.loss(
                out_bboxes,
                out_logits,
                dec_out_corners,
                dec_out_refs,
                dec_pre_bboxes,
                dec_pre_logits,
                inputs['gt_bbox'],
                inputs['gt_class'],
                dn_bboxes=dn_out_bboxes,
                dn_logits=dn_out_logits,
                dn_corners=dn_out_corners,
                dn_refs=dn_out_refs,
                dn_pre_bboxes=dn_pre_bboxes,
                dn_pre_logits=dn_pre_logits,
                dn_meta=dn_meta,
                gt_score=inputs.get('gt_score', None))
        else:
            return (dec_out_bboxes[self.eval_idx],
                    dec_out_logits[self.eval_idx], None)
