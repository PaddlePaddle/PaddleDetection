#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable
from .. import ops


@register
@serializable
class ProposalGenerator(object):
    def __init__(self,
                 pre_nms_top_n=12000,
                 post_nms_top_n=2000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.,
                 topk_after_collect=False):
        super(ProposalGenerator, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta
        self.topk_after_collect = topk_after_collect

    def __call__(self, scores, bbox_deltas, anchors, im_shape):

        top_n = self.pre_nms_top_n if self.topk_after_collect else self.post_nms_top_n
        variances = paddle.ones_like(anchors)
        rpn_rois, rpn_rois_prob, rpn_rois_num = ops.generate_proposals(
            scores,
            bbox_deltas,
            im_shape,
            anchors,
            variances,
            pre_nms_top_n=self.pre_nms_top_n,
            post_nms_top_n=top_n,
            nms_thresh=self.nms_thresh,
            min_size=self.min_size,
            eta=self.eta,
            return_rois_num=True)
        return rpn_rois, rpn_rois_prob, rpn_rois_num, self.post_nms_top_n
