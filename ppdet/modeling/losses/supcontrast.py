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
import paddle.nn as nn
import paddle.nn.functional as F

import random
from ppdet.core.workspace import register


__all__ = ['SupContrast']


@register
class SupContrast(nn.Layer):
    __shared__ = [
        'num_classes'
    ]
    def __init__(self, num_classes=80, temperature=2.5, sample_num=4096, thresh=0.75):
        super(SupContrast, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.sample_num = sample_num
        self.thresh = thresh
    def forward(self, features, labels, scores):
        
        assert features.shape[0] == labels.shape[0] == scores.shape[0]
        positive_mask = (labels < self.num_classes)
        positive_features, positive_labels, positive_scores = features[positive_mask], labels[positive_mask], \
                                                              scores[positive_mask]
        
        negative_mask = (labels == self.num_classes)
        negative_features, negative_labels, negative_scores = features[negative_mask], labels[negative_mask], \
                                                              scores[negative_mask]
        
        N = negative_features.shape[0]
        S = self.sample_num - positive_mask.sum()   
        index = paddle.to_tensor(random.sample(range(N), int(S)), dtype='int32')

        negative_features = paddle.index_select(x=negative_features, index=index, axis=0)
        negative_labels = paddle.index_select(x=negative_labels, index=index, axis=0)
        negative_scores = paddle.index_select(x=negative_scores, index=index, axis=0)
        
        features = paddle.concat([positive_features, negative_features], 0)
        labels = paddle.concat([positive_labels, negative_labels], 0)
        scores = paddle.concat([positive_scores, negative_scores], 0)

        if len(labels.shape) == 1:
            labels = labels.reshape([-1, 1])
        label_mask = paddle.equal(labels, labels.T).detach()
        similarity = (paddle.matmul(features, features.T) / self.temperature)

        sim_row_max = paddle.max(similarity, axis=1, keepdim=True)
        similarity = similarity - sim_row_max

        logits_mask = paddle.ones_like(similarity).detach()
        logits_mask.fill_diagonal_(0)

        exp_sim = paddle.exp(similarity) * logits_mask
        log_prob = similarity - paddle.log(exp_sim.sum(axis=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
        keep = scores > self.thresh
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        return loss.mean()