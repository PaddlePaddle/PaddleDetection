# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
import paddle.fluid.layers as layers
from paddle.fluid.layers import (tensor, iou_similarity, bipartite_match,
                                 target_assign, box_coder)
from ppdet.core.workspace import register, serializable

__all__ = ['SSDWithLmkLoss']


@register
@serializable
class SSDWithLmkLoss(object):
    """
    ssd_with_lmk_loss function.
    Args:
        background_label (int): The index of background label, 0 by default.
        overlap_threshold (float): If match_type is `per_prediction`,
            use `overlap_threshold` to determine the extra matching bboxes
            when finding matched boxes. 0.5 by default.
        neg_pos_ratio (float): The ratio of the negative boxes to the positive
            boxes, used only when mining_type is `max_negative`, 3.0 by default.
        neg_overlap (float): The negative overlap upper bound for the unmatched
            predictions. Use only when mining_type is `max_negative`, 0.5 by default.
        loc_loss_weight (float): Weight for localization loss, 1.0 by default.
        conf_loss_weight (float): Weight for confidence loss, 1.0 by default.
        match_type (str): The type of matching method during training, should be
            `bipartite` or `per_prediction`, `per_prediction` by default.
        normalize (bool): Whether to normalize the loss by the total number of
            output locations, True by default.
    """

    def __init__(self,
                 background_label=0,
                 overlap_threshold=0.5,
                 neg_pos_ratio=3.0,
                 neg_overlap=0.5,
                 loc_loss_weight=1.0,
                 conf_loss_weight=1.0,
                 match_type='per_prediction',
                 normalize=True):
        super(SSDWithLmkLoss, self).__init__()
        self.background_label = background_label
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_overlap = neg_overlap
        self.loc_loss_weight = loc_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.match_type = match_type
        self.normalize = normalize

    def __call__(self,
                 location,
                 confidence,
                 gt_box,
                 gt_label,
                 landmark_predict,
                 lmk_label,
                 lmk_ignore_flag,
                 prior_box,
                 prior_box_var=None):
        def _reshape_to_2d(var):
            return layers.flatten(x=var, axis=2)

        helper = LayerHelper('ssd_loss')  #, **locals())
        # Only support mining_type == 'max_negative' now.
        mining_type = 'max_negative'
        # The max `sample_size` of negative box, used only 
        # when mining_type is `hard_example`.
        sample_size = None
        num, num_prior, num_class = confidence.shape
        conf_shape = layers.shape(confidence)

        # 1. Find matched boundding box by prior box.
        # 1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
        iou = iou_similarity(x=gt_box, y=prior_box)
        # 1.2 Compute matched boundding box by bipartite matching algorithm.
        matched_indices, matched_dist = bipartite_match(iou, self.match_type,
                                                        self.overlap_threshold)

        # 2. Compute confidence for mining hard examples
        # 2.1. Get the target label based on matched indices
        gt_label = layers.reshape(
            x=gt_label, shape=(len(gt_label.shape) - 1) * (0, ) + (-1, 1))
        gt_label.stop_gradient = True
        target_label, _ = target_assign(
            gt_label, matched_indices, mismatch_value=self.background_label)
        # 2.2. Compute confidence loss.
        # Reshape confidence to 2D tensor.
        confidence = _reshape_to_2d(confidence)
        target_label = tensor.cast(x=target_label, dtype='int64')
        target_label = _reshape_to_2d(target_label)
        target_label.stop_gradient = True
        conf_loss = layers.softmax_with_cross_entropy(confidence, target_label)
        # 3. Mining hard examples
        actual_shape = layers.slice(conf_shape, axes=[0], starts=[0], ends=[2])
        actual_shape.stop_gradient = True
        conf_loss = layers.reshape(
            x=conf_loss, shape=(-1, 0), actual_shape=actual_shape)
        conf_loss.stop_gradient = True
        neg_indices = helper.create_variable_for_type_inference(dtype='int32')
        updated_matched_indices = helper.create_variable_for_type_inference(
            dtype=matched_indices.dtype)
        helper.append_op(
            type='mine_hard_examples',
            inputs={
                'ClsLoss': conf_loss,
                'LocLoss': None,
                'MatchIndices': matched_indices,
                'MatchDist': matched_dist,
            },
            outputs={
                'NegIndices': neg_indices,
                'UpdatedMatchIndices': updated_matched_indices
            },
            attrs={
                'neg_pos_ratio': self.neg_pos_ratio,
                'neg_dist_threshold': self.neg_overlap,
                'mining_type': mining_type,
                'sample_size': sample_size,
            })

        # 4. Assign classification and regression targets
        # 4.1. Encoded bbox according to the prior boxes.
        encoded_bbox = box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=gt_box,
            code_type='encode_center_size')
        # 4.2. Assign regression targets
        target_bbox, target_loc_weight = target_assign(
            encoded_bbox,
            updated_matched_indices,
            mismatch_value=self.background_label)
        # 4.3. Assign classification targets
        target_label, target_conf_weight = target_assign(
            gt_label,
            updated_matched_indices,
            negative_indices=neg_indices,
            mismatch_value=self.background_label)

        target_loc_weight = target_loc_weight * target_label
        encoded_lmk_label = self.decode_lmk(lmk_label, prior_box, prior_box_var)

        target_lmk, target_lmk_weight = target_assign(
            encoded_lmk_label,
            updated_matched_indices,
            mismatch_value=self.background_label)
        lmk_ignore_flag = layers.reshape(
            x=lmk_ignore_flag,
            shape=(len(lmk_ignore_flag.shape) - 1) * (0, ) + (-1, 1))
        target_ignore, nouse = target_assign(
            lmk_ignore_flag,
            updated_matched_indices,
            mismatch_value=self.background_label)

        target_lmk_weight = target_lmk_weight * target_ignore
        landmark_predict = _reshape_to_2d(landmark_predict)
        target_lmk = _reshape_to_2d(target_lmk)
        target_lmk_weight = _reshape_to_2d(target_lmk_weight)
        lmk_loss = layers.smooth_l1(landmark_predict, target_lmk)
        lmk_loss = lmk_loss * target_lmk_weight
        target_lmk.stop_gradient = True
        target_lmk_weight.stop_gradient = True
        target_ignore.stop_gradient = True
        nouse.stop_gradient = True

        # 5. Compute loss.
        # 5.1 Compute confidence loss.
        target_label = _reshape_to_2d(target_label)
        target_label = tensor.cast(x=target_label, dtype='int64')

        conf_loss = layers.softmax_with_cross_entropy(confidence, target_label)
        target_conf_weight = _reshape_to_2d(target_conf_weight)
        conf_loss = conf_loss * target_conf_weight

        # the target_label and target_conf_weight do not have gradient.
        target_label.stop_gradient = True
        target_conf_weight.stop_gradient = True

        # 5.2 Compute regression loss.
        location = _reshape_to_2d(location)
        target_bbox = _reshape_to_2d(target_bbox)

        loc_loss = layers.smooth_l1(location, target_bbox)
        target_loc_weight = _reshape_to_2d(target_loc_weight)
        loc_loss = loc_loss * target_loc_weight

        # the target_bbox and target_loc_weight do not have gradient.
        target_bbox.stop_gradient = True
        target_loc_weight.stop_gradient = True

        # 5.3 Compute overall weighted loss.
        loss = self.conf_loss_weight * conf_loss + self.loc_loss_weight * loc_loss + 0.4 * lmk_loss
        # reshape to [N, Np], N is the batch size and Np is the prior box number.
        loss = layers.reshape(x=loss, shape=(-1, 0), actual_shape=actual_shape)
        loss = layers.reduce_sum(loss, dim=1, keep_dim=True)
        if self.normalize:
            normalizer = layers.reduce_sum(target_loc_weight) + 1
            loss = loss / normalizer

        return loss

    def decode_lmk(self, lmk_label, prior_box, prior_box_var):
        label0, label1, label2, label3, label4 = fluid.layers.split(
            lmk_label, num_or_sections=5, dim=1)
        lmk_labels_list = [label0, label1, label2, label3, label4]
        encoded_lmk_list = []
        for label in lmk_labels_list:
            concat_label = fluid.layers.concat([label, label], axis=1)
            encoded_label = box_coder(
                prior_box=prior_box,
                prior_box_var=prior_box_var,
                target_box=concat_label,
                code_type='encode_center_size')
            encoded_lmk_label, _ = fluid.layers.split(
                encoded_label, num_or_sections=2, dim=2)
            encoded_lmk_list.append(encoded_lmk_label)

        encoded_lmk_concat = fluid.layers.concat(
            [
                encoded_lmk_list[0], encoded_lmk_list[1], encoded_lmk_list[2],
                encoded_lmk_list[3], encoded_lmk_list[4]
            ],
            axis=2)
        return encoded_lmk_concat
