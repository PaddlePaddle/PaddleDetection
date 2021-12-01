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

import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingUniform, Uniform
from ppdet.core.workspace import register
from ppdet.modeling.heads.centernet_head import ConvLayer

__all__ = ['FairMOTEmbeddingHead']


@register
class FairMOTEmbeddingHead(nn.Layer):
    __shared__ = ['num_classes']
    """
    Args:
        in_channels (int): the channel number of input to FairMOTEmbeddingHead.
        ch_head (int): the channel of features before fed into embedding, 256 by default.
        ch_emb (int): the channel of the embedding feature, 128 by default.
        num_identities_dict (dict): the number of identities of each category,
            support single class and multi-calss, {0: 14455} as default. 
    """

    def __init__(self,
                 in_channels,
                 ch_head=256,
                 ch_emb=128,
                 num_classes=1,
                 num_identities_dict={0: 14455}):
        super(FairMOTEmbeddingHead, self).__init__()
        assert num_classes >= 1
        self.num_classes = num_classes
        self.ch_emb = ch_emb
        self.num_identities_dict = num_identities_dict
        self.reid = nn.Sequential(
            ConvLayer(
                in_channels, ch_head, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                ch_head, ch_emb, kernel_size=1, stride=1, padding=0, bias=True))
        param_attr = paddle.ParamAttr(initializer=KaimingUniform())
        bound = 1 / math.sqrt(ch_emb)
        bias_attr = paddle.ParamAttr(initializer=Uniform(-bound, bound))
        self.reid_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

        if num_classes == 1:
            nID = self.num_identities_dict[0]  # single class
            self.classifier = nn.Linear(
                ch_emb, nID, weight_attr=param_attr, bias_attr=bias_attr)
            # When num_identities(nID) is 1, emb_scale is set as 1
            self.emb_scale = math.sqrt(2) * math.log(nID - 1) if nID > 1 else 1
        else:
            self.classifiers = dict()
            self.emb_scale_dict = dict()
            for cls_id, nID in self.num_identities_dict.items():
                self.classifiers[str(cls_id)] = nn.Linear(
                    ch_emb, nID, weight_attr=param_attr, bias_attr=bias_attr)
                # When num_identities(nID) is 1, emb_scale is set as 1
                self.emb_scale_dict[str(cls_id)] = math.sqrt(2) * math.log(
                    nID - 1) if nID > 1 else 1

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def process_by_class(self, bboxes, embedding, bbox_inds, topk_clses):
        pred_dets, pred_embs = [], []
        for cls_id in range(self.num_classes):
            inds_masks = topk_clses == cls_id
            inds_masks = paddle.cast(inds_masks, 'float32')

            pos_num = inds_masks.sum().numpy()
            if pos_num == 0:
                continue

            cls_inds_mask = inds_masks > 0

            bbox_mask = paddle.nonzero(cls_inds_mask)
            cls_bboxes = paddle.gather_nd(bboxes, bbox_mask)
            pred_dets.append(cls_bboxes)

            cls_inds = paddle.masked_select(bbox_inds, cls_inds_mask)
            cls_inds = cls_inds.unsqueeze(-1)
            cls_embedding = paddle.gather_nd(embedding, cls_inds)
            pred_embs.append(cls_embedding)

        return paddle.concat(pred_dets), paddle.concat(pred_embs)

    def forward(self,
                neck_feat,
                inputs,
                bboxes=None,
                bbox_inds=None,
                topk_clses=None):
        reid_feat = self.reid(neck_feat)
        if self.training:
            if self.num_classes == 1:
                loss = self.get_loss(reid_feat, inputs)
            else:
                loss = self.get_mc_loss(reid_feat, inputs)
            return loss
        else:
            assert bboxes is not None and bbox_inds is not None
            reid_feat = F.normalize(reid_feat)
            embedding = paddle.transpose(reid_feat, [0, 2, 3, 1])
            embedding = paddle.reshape(embedding, [-1, self.ch_emb])
            # embedding shape: [bs * h * w, ch_emb]

            if self.num_classes == 1:
                pred_dets = bboxes
                pred_embs = paddle.gather(embedding, bbox_inds)
            else:
                pred_dets, pred_embs = self.process_by_class(
                    bboxes, embedding, bbox_inds, topk_clses)
            return pred_dets, pred_embs

    def get_loss(self, feat, inputs):
        index = inputs['index']
        mask = inputs['index_mask']
        target = inputs['reid']
        target = paddle.masked_select(target, mask > 0)
        target = paddle.unsqueeze(target, 1)

        feat = paddle.transpose(feat, perm=[0, 2, 3, 1])
        feat_n, feat_h, feat_w, feat_c = feat.shape
        feat = paddle.reshape(feat, shape=[feat_n, -1, feat_c])
        index = paddle.unsqueeze(index, 2)
        batch_inds = list()
        for i in range(feat_n):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)
        feat = paddle.gather_nd(feat, index=index)

        mask = paddle.unsqueeze(mask, axis=2)
        mask = paddle.expand_as(mask, feat)
        mask.stop_gradient = True
        feat = paddle.masked_select(feat, mask > 0)
        feat = paddle.reshape(feat, shape=[-1, feat_c])
        feat = F.normalize(feat)
        feat = self.emb_scale * feat
        logit = self.classifier(feat)
        target.stop_gradient = True
        loss = self.reid_loss(logit, target)
        valid = (target != self.reid_loss.ignore_index)
        valid.stop_gradient = True
        count = paddle.sum((paddle.cast(valid, dtype=np.int32)))
        count.stop_gradient = True
        if count > 0:
            loss = loss / count

        return loss

    def get_mc_loss(self, feat, inputs):
        # feat.shape = [bs, ch_emb, h, w]
        assert 'cls_id_map' in inputs and 'cls_tr_ids' in inputs
        index = inputs['index']
        mask = inputs['index_mask']
        cls_id_map = inputs['cls_id_map']  # [bs, h, w]
        cls_tr_ids = inputs['cls_tr_ids']  # [bs, num_classes, h, w]

        feat = paddle.transpose(feat, perm=[0, 2, 3, 1])
        feat_n, feat_h, feat_w, feat_c = feat.shape
        feat = paddle.reshape(feat, shape=[feat_n, -1, feat_c])

        index = paddle.unsqueeze(index, 2)
        batch_inds = list()
        for i in range(feat_n):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)
        feat = paddle.gather_nd(feat, index=index)

        mask = paddle.unsqueeze(mask, axis=2)
        mask = paddle.expand_as(mask, feat)
        mask.stop_gradient = True
        feat = paddle.masked_select(feat, mask > 0)
        feat = paddle.reshape(feat, shape=[-1, feat_c])

        reid_losses = 0
        for cls_id, id_num in self.num_identities_dict.items():
            # target
            cur_cls_tr_ids = paddle.reshape(
                cls_tr_ids[:, cls_id, :, :], shape=[feat_n, -1])  # [bs, h*w]
            cls_id_target = paddle.gather_nd(cur_cls_tr_ids, index=index)
            mask = inputs['index_mask']
            cls_id_target = paddle.masked_select(cls_id_target, mask > 0)
            cls_id_target.stop_gradient = True

            # feat
            cls_id_feat = self.emb_scale_dict[str(cls_id)] * F.normalize(feat)
            cls_id_pred = self.classifiers[str(cls_id)](cls_id_feat)

            loss = self.reid_loss(cls_id_pred, cls_id_target)
            valid = (cls_id_target != self.reid_loss.ignore_index)
            valid.stop_gradient = True
            count = paddle.sum((paddle.cast(valid, dtype=np.int32)))
            count.stop_gradient = True
            if count > 0:
                loss = loss / count
            reid_losses += loss

        return reid_losses
