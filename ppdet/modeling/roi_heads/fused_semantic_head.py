# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling.ops import ConvNorm

__all__ = ['FusedSemanticHead']


@register
class FusedSemanticHead(object):
    def __init__(self, semantic_num_class=183):
        super(FusedSemanticHead, self).__init__()
        self.semantic_num_class = semantic_num_class

    def get_out(self,
                fpn_feats,
                out_c=256,
                num_convs=4,
                fusion_level='fpn_res3_sum'):
        new_feat = fpn_feats[fusion_level]
        new_feat_list = [new_feat, ]
        target_shape = fluid.layers.shape(new_feat)[2:]
        for k, v in fpn_feats.items():
            if k != fusion_level:
                v = fluid.layers.resize_bilinear(
                    v, target_shape, align_corners=True)
                v = fluid.layers.conv2d(v, out_c, 1)
                new_feat_list.append(v)
        new_feat = fluid.layers.sum(new_feat_list)

        for i in range(num_convs):
            new_feat = fluid.layers.conv2d(new_feat, out_c, 3, padding=1)

        # conv embedding
        semantic_feat = fluid.layers.conv2d(new_feat, out_c, 1)
        # conv logits 
        seg_pred = fluid.layers.conv2d(new_feat, self.semantic_num_class, 1)
        return semantic_feat, seg_pred

    def get_loss(self, logit, label, ignore_index=255):
        label = fluid.layers.resize_nearest(label,
                                            fluid.layers.shape(logit)[2:])
        label = fluid.layers.reshape(label, [-1, 1])
        label = fluid.layers.cast(label, 'int64')

        logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
        logit = fluid.layers.reshape(logit, [-1, self.semantic_num_class])

        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            soft_label=False,
            ignore_index=ignore_index,
            return_softmax=True)

        ignore_mask = (label.astype('int32') != 255).astype('int32')
        if ignore_mask is not None:
            ignore_mask = fluid.layers.cast(ignore_mask, 'float32')
            ignore_mask = fluid.layers.reshape(ignore_mask, [-1, 1])
            loss = loss * ignore_mask
            avg_loss = fluid.layers.mean(loss) / fluid.layers.mean(ignore_mask)
            ignore_mask.stop_gradient = True
        else:
            avg_loss = fluid.layers.mean(loss)
        label.stop_gradient = True

        return avg_loss
