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

# This code is referenced from: https://github.com/open-mmlab/mmdetection

import paddle
from paddle import nn

from ppdet.core.workspace import register

__all__ = ['EmbeddingRPNHead']


@register
class EmbeddingRPNHead(nn.Layer):
    __shared__ = ['proposal_embedding_dim']

    def __init__(self, num_proposals, proposal_embedding_dim=256):
        super(EmbeddingRPNHead, self).__init__()

        self.num_proposals = num_proposals
        self.proposal_embedding_dim = proposal_embedding_dim

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(self.num_proposals,
                                                   self.proposal_embedding_dim)

    def _init_weights(self):
        init_bboxes = paddle.empty_like(self.init_proposal_bboxes.weight)
        init_bboxes[:, :2] = 0.5
        init_bboxes[:, 2:] = 1.0
        self.init_proposal_bboxes.weight.set_value(init_bboxes)

    @staticmethod
    def bbox_cxcywh_to_xyxy(x):
        cxcy, wh = paddle.split(x, 2, axis=-1)
        return paddle.concat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)

    def forward(self, img_whwh):
        proposal_bboxes = self.init_proposal_bboxes.weight.clone()
        proposal_bboxes = self.bbox_cxcywh_to_xyxy(proposal_bboxes)
        proposal_bboxes = proposal_bboxes.unsqueeze(0) * img_whwh.unsqueeze(1)

        proposal_features = self.init_proposal_features.weight.clone()
        proposal_features = proposal_features.unsqueeze(0).tile(
            [img_whwh.shape[0], 1, 1])

        return proposal_bboxes, proposal_features
