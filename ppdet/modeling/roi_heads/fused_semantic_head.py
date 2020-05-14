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

    def get_out(self, fpn_feats, out_c=256, num_convs=4, fusion_level=3):
        r"""Multi-level fused semantic segmentation head.
        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||                  /-> 1x1 conv (mask prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*4)
                            |                  \-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
        """
        new_feat = fpn_feats['fpn_res3_sum']
        target_shape = fluid.layers.shape(new_feat)[2:]
        #fluid.layers.Print(target_shape)
        for k, v in fpn_feats.items():
            if k == 'fpn_res3_sum':
                v = fluid.layers.resize_bilinear(
                    v, target_shape, align_corners=True)
                v = fluid.layers.conv2d(v, out_c, 1)
                new_feat = fluid.layers.sum([new_feat, v])

        for i in range(num_convs):
            new_feat = fluid.layers.conv2d(new_feat, out_c, 3, padding=1)

        # conv embedding
        semantic_feat = fluid.layers.conv2d(new_feat, out_c, 1)
        # conv logits 
        seg_pred = fluid.layers.conv2d(new_feat, self.semantic_num_class, 1)
        return semantic_feat, seg_pred

    def get_loss(self, logit, label, weight=None, ignore_index=255):
        label = fluid.layers.resize_nearest(label,
                                            fluid.layers.shape(logit)[2:])
        #label = fluid.layers.elementwise_min(
        #    label, fluid.layers.assign(np.array([num_classes - 1], dtype=np.int32)))
        label = fluid.layers.reshape(label, [-1, 1])
        label = fluid.layers.cast(label, 'int64')

        logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
        logit = fluid.layers.reshape(logit, [-1, self.semantic_num_class])
        if weight is None:
            loss, probs = fluid.layers.softmax_with_cross_entropy(
                logit, label, ignore_index=ignore_index, return_softmax=True)
        else:
            label_one_hot = fluid.layers.one_hot(
                input=label, depth=self.semantic_num_class)
            if isinstance(weight, list):
                assert len(
                    weight
                ) == self.semantic_num_class, "weight length must equal num of classes"
                weight = fluid.layers.assign(
                    np.array(
                        [weight], dtype='float32'))
            elif isinstance(weight, str):
                assert weight.lower(
                ) == 'dynamic', 'if weight is string, must be dynamic!'
                tmp = []
                total_num = fluid.layers.cast(
                    fluid.layers.shape(label)[0], 'float32')
                for i in range(self.semantic_num_class):
                    cls_pixel_num = fluid.layers.reduce_sum(label_one_hot[:, i])
                    ratio = total_num / (cls_pixel_num + 1)
                    tmp.append(ratio)
                weight = fluid.layers.concat(tmp)
                weight = weight / fluid.layers.reduce_sum(
                    weight) * self.semantic_num_class
            elif isinstance(weight, fluid.layers.Variable):
                pass
            else:
                raise ValueError(
                    'Expect weight is a list, string or Variable, but receive {}'.
                    format(type(weight)))
            weight = fluid.layers.reshape(weight, [1, self.semantic_num_class])
            weighted_label_one_hot = fluid.layers.elementwise_mul(label_one_hot,
                                                                  weight)
            probs = fluid.layers.softmax(logit)
            loss = fluid.layers.cross_entropy(
                probs,
                weighted_label_one_hot,
                soft_label=True,
                ignore_index=ignore_index)
            weighted_label_one_hot.stop_gradient = True

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
