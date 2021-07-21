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

import paddle

from ppdet.core.workspace import register, serializable
from .. import ops


@register
@serializable
class ProposalGenerator(object):
    """
    Proposal generation module

    For more details, please refer to the document of generate_proposals 
    in ppdet/modeing/ops.py

    Args:
        pre_nms_top_n (int): Number of total bboxes to be kept per
            image before NMS. default 6000
        post_nms_top_n (int): Number of total bboxes to be kept per
            image after NMS. default 1000
        nms_thresh (float): Threshold in NMS. default 0.5
        min_size (flaot): Remove predicted boxes with either height or
             width < min_size. default 0.1
        eta (float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
             `adaptive_threshold = adaptive_threshold * eta` in each iteration.
             default 1.
        topk_after_collect (bool): whether to adopt topk after batch 
             collection. If topk_after_collect is true, box filter will not be 
             used after NMS at each image in proposal generation. default false
    """

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
