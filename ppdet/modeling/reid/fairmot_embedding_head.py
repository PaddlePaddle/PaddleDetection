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
    """
    Args:
        in_channels (int): the channel number of input to FairMOTEmbeddingHead.
        ch_head (int): the channel of features before fed into embedding, 256 by default.
        ch_emb (int): the channel of the embedding feature, 128 by default.
        num_identifiers (int): the number of identifiers, 14455 by default.

    """

    def __init__(self,
                 in_channels,
                 ch_head=256,
                 ch_emb=128,
                 num_identifiers=14455):
        super(FairMOTEmbeddingHead, self).__init__()
        self.reid = nn.Sequential(
            ConvLayer(
                in_channels, ch_head, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                ch_head, ch_emb, kernel_size=1, stride=1, padding=0, bias=True))
        param_attr = paddle.ParamAttr(initializer=KaimingUniform())
        bound = 1 / math.sqrt(ch_emb)
        bias_attr = paddle.ParamAttr(initializer=Uniform(-bound, bound))
        self.classifier = nn.Linear(
            ch_emb,
            num_identifiers,
            weight_attr=param_attr,
            bias_attr=bias_attr)
        self.reid_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        # When num_identifiers is 1, emb_scale is set as 1
        self.emb_scale = math.sqrt(2) * math.log(
            num_identifiers - 1) if num_identifiers > 1 else 1

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self, feat, inputs):
        reid_feat = self.reid(feat)
        if self.training:
            loss = self.get_loss(reid_feat, inputs)
            return loss
        else:
            reid_feat = F.normalize(reid_feat)
            return reid_feat

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
