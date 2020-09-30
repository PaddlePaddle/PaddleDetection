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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

from ..backbones.hourglass import _conv_norm, kaiming_init
from ppdet.core.workspace import register
import numpy as np
import logging
logger = logging.getLogger(__name__)

__all__ = ['CornerHead']


def corner_output(x, pool1, pool2, dim, name=None):
    p_conv1 = fluid.layers.conv2d(
        pool1 + pool2,
        filter_size=3,
        num_filters=dim,
        padding=1,
        param_attr=ParamAttr(
            name=name + "_p_conv1_weight",
            initializer=kaiming_init(pool1 + pool2, 3)),
        bias_attr=False,
        name=name + '_p_conv1')
    p_bn1 = fluid.layers.batch_norm(
        p_conv1,
        param_attr=ParamAttr(name=name + '_p_bn1_weight'),
        bias_attr=ParamAttr(name=name + '_p_bn1_bias'),
        moving_mean_name=name + '_p_bn1_running_mean',
        moving_variance_name=name + '_p_bn1_running_var',
        name=name + '_p_bn1')

    conv1 = fluid.layers.conv2d(
        x,
        filter_size=1,
        num_filters=dim,
        param_attr=ParamAttr(
            name=name + "_conv1_weight", initializer=kaiming_init(x, 1)),
        bias_attr=False,
        name=name + '_conv1')
    bn1 = fluid.layers.batch_norm(
        conv1,
        param_attr=ParamAttr(name=name + '_bn1_weight'),
        bias_attr=ParamAttr(name=name + '_bn1_bias'),
        moving_mean_name=name + '_bn1_running_mean',
        moving_variance_name=name + '_bn1_running_var',
        name=name + '_bn1')

    relu1 = fluid.layers.relu(p_bn1 + bn1)
    conv2 = _conv_norm(
        relu1, 3, dim, pad=1, bn_act='relu', name=name + '_conv2')
    return conv2


def corner_pool(x, dim, pool1, pool2, is_test=False, name=None):
    p1_conv1 = _conv_norm(
        x, 3, 128, pad=1, bn_act='relu', name=name + '_p1_conv1')
    pool1 = pool1(p1_conv1, is_test=is_test, name=name + '_pool1')
    p2_conv1 = _conv_norm(
        x, 3, 128, pad=1, bn_act='relu', name=name + '_p2_conv1')
    pool2 = pool2(p2_conv1, is_test=is_test, name=name + '_pool2')

    conv2 = corner_output(x, pool1, pool2, dim, name)
    return conv2


def gather_feat(feat, ind, batch_size=1):
    feats = []
    for bind in range(batch_size):
        feat_b = feat[bind]
        ind_b = ind[bind]
        ind_b.stop_gradient = True
        feat_bg = fluid.layers.gather(feat_b, ind_b)
        feats.append(fluid.layers.unsqueeze(feat_bg, axes=[0]))
    feat_g = fluid.layers.concat(feats, axis=0)
    return feat_g


def mask_feat(feat, ind, batch_size=1):
    feat_t = fluid.layers.transpose(feat, [0, 2, 3, 1])
    C = feat_t.shape[3]
    feat_r = fluid.layers.reshape(feat_t, [0, -1, C])
    return gather_feat(feat_r, ind, batch_size)


def nms(heat):
    hmax = fluid.layers.pool2d(heat, pool_size=3, pool_padding=1)
    keep = fluid.layers.cast(heat == hmax, 'float32')
    return heat * keep


def _topk(scores, batch_size, height, width, K):
    scores_r = fluid.layers.reshape(scores, [batch_size, -1])
    topk_scores, topk_inds = fluid.layers.topk(scores_r, K)
    topk_inds = fluid.layers.cast(topk_inds, 'int32')
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = fluid.layers.cast(topk_inds // width, 'float32')
    topk_xs = fluid.layers.cast(topk_inds % width, 'float32')
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def filter_scores(scores, index_list):
    for ind in index_list:
        tmp = scores * fluid.layers.cast((1 - ind), 'float32')
        scores = tmp - fluid.layers.cast(ind, 'float32')
    return scores


def decode(tl_heat,
           br_heat,
           tl_tag,
           br_tag,
           tl_regr,
           br_regr,
           ae_threshold=1,
           num_dets=1000,
           K=100,
           batch_size=1):
    shape = fluid.layers.shape(tl_heat)
    H, W = shape[2], shape[3]

    tl_heat = fluid.layers.sigmoid(tl_heat)
    br_heat = fluid.layers.sigmoid(br_heat)

    tl_heat_nms = nms(tl_heat)
    br_heat_nms = nms(br_heat)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat_nms, batch_size,
                                                       H, W, K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat_nms, batch_size,
                                                       H, W, K)
    tl_ys = fluid.layers.expand(
        fluid.layers.reshape(tl_ys, [-1, K, 1]), [1, 1, K])
    tl_xs = fluid.layers.expand(
        fluid.layers.reshape(tl_xs, [-1, K, 1]), [1, 1, K])
    br_ys = fluid.layers.expand(
        fluid.layers.reshape(br_ys, [-1, 1, K]), [1, K, 1])
    br_xs = fluid.layers.expand(
        fluid.layers.reshape(br_xs, [-1, 1, K]), [1, K, 1])

    tl_regr = mask_feat(tl_regr, tl_inds, batch_size)
    br_regr = mask_feat(br_regr, br_inds, batch_size)
    tl_regr = fluid.layers.reshape(tl_regr, [-1, K, 1, 2])
    br_regr = fluid.layers.reshape(br_regr, [-1, 1, K, 2])

    tl_xs = tl_xs + tl_regr[:, :, :, 0]
    tl_ys = tl_ys + tl_regr[:, :, :, 1]
    br_xs = br_xs + br_regr[:, :, :, 0]
    br_ys = br_ys + br_regr[:, :, :, 1]

    bboxes = fluid.layers.stack([tl_xs, tl_ys, br_xs, br_ys], axis=-1)

    tl_tag = mask_feat(tl_tag, tl_inds, batch_size)
    br_tag = mask_feat(br_tag, br_inds, batch_size)
    tl_tag = fluid.layers.expand(
        fluid.layers.reshape(tl_tag, [-1, K, 1]), [1, 1, K])
    br_tag = fluid.layers.expand(
        fluid.layers.reshape(br_tag, [-1, 1, K]), [1, K, 1])
    dists = fluid.layers.abs(tl_tag - br_tag)

    tl_scores = fluid.layers.expand(
        fluid.layers.reshape(tl_scores, [-1, K, 1]), [1, 1, K])
    br_scores = fluid.layers.expand(
        fluid.layers.reshape(br_scores, [-1, 1, K]), [1, K, 1])
    scores = (tl_scores + br_scores) / 2.

    tl_clses = fluid.layers.expand(
        fluid.layers.reshape(tl_clses, [-1, K, 1]), [1, 1, K])
    br_clses = fluid.layers.expand(
        fluid.layers.reshape(br_clses, [-1, 1, K]), [1, K, 1])
    cls_inds = fluid.layers.cast(tl_clses != br_clses, 'int32')
    dist_inds = fluid.layers.cast(dists > ae_threshold, 'int32')

    width_inds = fluid.layers.cast(br_xs < tl_xs, 'int32')
    height_inds = fluid.layers.cast(br_ys < tl_ys, 'int32')

    scores = filter_scores(scores,
                           [cls_inds, dist_inds, width_inds, height_inds])
    scores = fluid.layers.reshape(scores, [-1, K * K])

    scores, inds = fluid.layers.topk(scores, num_dets)
    scores = fluid.layers.reshape(scores, [-1, num_dets, 1])

    bboxes = fluid.layers.reshape(bboxes, [batch_size, -1, 4])
    bboxes = gather_feat(bboxes, inds, batch_size)

    clses = fluid.layers.reshape(tl_clses, [batch_size, -1, 1])
    clses = gather_feat(clses, inds, batch_size)

    tl_scores = fluid.layers.reshape(tl_scores, [batch_size, -1, 1])
    tl_scores = gather_feat(tl_scores, inds, batch_size)
    br_scores = fluid.layers.reshape(br_scores, [batch_size, -1, 1])
    br_scores = gather_feat(br_scores, inds, batch_size)

    bboxes = fluid.layers.cast(bboxes, 'float32')
    clses = fluid.layers.cast(clses, 'float32')
    return bboxes, scores, tl_scores, br_scores, clses


@register
class CornerHead(object):
    """
    CornerNet head with corner_pooling

    Args:
        train_batch_size(int): batch_size in training process
        test_batch_size(int): batch_size in test process, 1 by default
        num_classes(int): num of classes, 80 by default
        stack(int): stack of backbone, 2 by default
        pull_weight(float): weight of pull_loss, 0.1 by default
        push_weight(float): weight of push_loss, 0.1 by default
        ae_threshold(float|int): threshold for valid distance of predicted tags, 1 by default
        num_dets(int): num of detections, 1000 by default
        top_k(int): choose top_k pair of corners in prediction, 100 by default 
    """
    __shared__ = ['num_classes', 'stack', 'train_batch_size']

    def __init__(self,
                 train_batch_size=14,
                 test_batch_size=1,
                 num_classes=80,
                 stack=2,
                 pull_weight=0.1,
                 push_weight=0.1,
                 ae_threshold=1,
                 num_dets=1000,
                 top_k=100):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes
        self.stack = stack
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.ae_threshold = ae_threshold
        self.num_dets = num_dets
        self.K = top_k
        self.tl_heats = []
        self.br_heats = []
        self.tl_tags = []
        self.br_tags = []
        self.tl_offs = []
        self.br_offs = []

    def pred_mod(self, x, dim, name=None):
        conv0 = _conv_norm(
            x, 1, 256, with_bn=False, bn_act='relu', name=name + '_0')
        conv1 = fluid.layers.conv2d(
            input=conv0,
            filter_size=1,
            num_filters=dim,
            param_attr=ParamAttr(
                name=name + "_1_weight", initializer=kaiming_init(conv0, 1)),
            bias_attr=ParamAttr(
                name=name + "_1_bias", initializer=Constant(-2.19)),
            name=name + '_1')
        return conv1

    def get_output(self, input):
        try:
            from ppdet.ext_op import cornerpool_lib
        except:
            logger.error(
                "cornerpool_lib not found, compile in ppdet/ext_op at first")
        for ind in range(self.stack):
            cnv = input[ind]
            tl_modules = corner_pool(
                cnv,
                256,
                cornerpool_lib.top_pool,
                cornerpool_lib.left_pool,
                name='tl_modules_' + str(ind))
            br_modules = corner_pool(
                cnv,
                256,
                cornerpool_lib.bottom_pool,
                cornerpool_lib.right_pool,
                name='br_modules_' + str(ind))

            tl_heat = self.pred_mod(
                tl_modules, self.num_classes, name='tl_heats_' + str(ind))
            br_heat = self.pred_mod(
                br_modules, self.num_classes, name='br_heats_' + str(ind))

            tl_tag = self.pred_mod(tl_modules, 1, name='tl_tags_' + str(ind))
            br_tag = self.pred_mod(br_modules, 1, name='br_tags_' + str(ind))

            tl_off = self.pred_mod(tl_modules, 2, name='tl_offs_' + str(ind))
            br_off = self.pred_mod(br_modules, 2, name='br_offs_' + str(ind))

            self.tl_heats.append(tl_heat)
            self.br_heats.append(br_heat)
            self.tl_tags.append(tl_tag)
            self.br_tags.append(br_tag)
            self.tl_offs.append(tl_off)
            self.br_offs.append(br_off)

    def focal_loss(self, preds, gt, gt_masks):
        preds_clip = []
        none_pos = fluid.layers.cast(
            fluid.layers.reduce_sum(gt_masks) == 0, 'float32')
        none_pos.stop_gradient = True
        min = fluid.layers.assign(np.array([1e-4], dtype='float32'))
        max = fluid.layers.assign(np.array([1 - 1e-4], dtype='float32'))
        for pred in preds:
            pred_s = fluid.layers.sigmoid(pred)
            pred_min = fluid.layers.elementwise_max(pred_s, min)
            pred_max = fluid.layers.elementwise_min(pred_min, max)
            preds_clip.append(pred_max)

        ones = fluid.layers.ones_like(gt)

        fg_map = fluid.layers.cast(gt == ones, 'float32')
        fg_map.stop_gradient = True
        num_pos = fluid.layers.reduce_sum(fg_map)
        min_num = fluid.layers.ones_like(num_pos)
        num_pos = fluid.layers.elementwise_max(num_pos, min_num)
        num_pos.stop_gradient = True
        bg_map = fluid.layers.cast(gt < ones, 'float32')
        bg_map.stop_gradient = True
        neg_weights = fluid.layers.pow(1 - gt, 4) * bg_map
        neg_weights.stop_gradient = True
        loss = fluid.layers.assign(np.array([0], dtype='float32'))
        for ind, pred in enumerate(preds_clip):
            pos_loss = fluid.layers.log(pred) * fluid.layers.pow(1 - pred,
                                                                 2) * fg_map

            neg_loss = fluid.layers.log(1 - pred) * fluid.layers.pow(
                pred, 2) * neg_weights

            pos_loss = fluid.layers.reduce_sum(pos_loss)
            neg_loss = fluid.layers.reduce_sum(neg_loss)
            focal_loss_ = (neg_loss + pos_loss) / (num_pos + none_pos)
            loss -= focal_loss_
        return loss

    def ae_loss(self, tl_tag, br_tag, gt_masks):
        num = fluid.layers.reduce_sum(gt_masks, dim=1)
        num_stop_gradient = True
        tag0 = fluid.layers.squeeze(tl_tag, [2])
        tag1 = fluid.layers.squeeze(br_tag, [2])
        tag_mean = (tag0 + tag1) / 2

        tag0 = fluid.layers.pow(tag0 - tag_mean, 2)
        tag1 = fluid.layers.pow(tag1 - tag_mean, 2)

        tag0 = fluid.layers.elementwise_div(tag0, num + 1e-4, axis=0)
        tag1 = fluid.layers.elementwise_div(tag1, num + 1e-4, axis=0)
        tag0 = tag0 * gt_masks
        tag1 = tag1 * gt_masks
        tag0 = fluid.layers.reduce_sum(tag0)
        tag1 = fluid.layers.reduce_sum(tag1)

        pull = tag0 + tag1

        mask_1 = fluid.layers.expand(
            fluid.layers.unsqueeze(gt_masks, [1]), [1, gt_masks.shape[1], 1])
        mask_2 = fluid.layers.expand(
            fluid.layers.unsqueeze(gt_masks, [2]), [1, 1, gt_masks.shape[1]])
        mask = fluid.layers.cast((mask_1 + mask_2) == 2, 'float32')
        mask.stop_gradient = True

        num2 = (num - 1) * num
        num2.stop_gradient = True
        tag_mean_1 = fluid.layers.expand(
            fluid.layers.unsqueeze(tag_mean, [1]), [1, tag_mean.shape[1], 1])
        tag_mean_2 = fluid.layers.expand(
            fluid.layers.unsqueeze(tag_mean, [2]), [1, 1, tag_mean.shape[1]])
        dist = tag_mean_1 - tag_mean_2
        dist = 1 - fluid.layers.abs(dist)
        dist = fluid.layers.relu(dist)
        dist = fluid.layers.elementwise_sub(dist, 1 / (num + 1e-4), axis=0)
        dist = fluid.layers.elementwise_div(dist, (num2 + 1e-4), axis=0)
        dist = dist * mask
        push = fluid.layers.reduce_sum(dist)
        return pull, push

    def off_loss(self, off, gt_off, gt_masks):
        mask = fluid.layers.unsqueeze(gt_masks, [2])
        mask = fluid.layers.expand_as(mask, gt_off)
        mask.stop_gradient = True
        off_loss = fluid.layers.smooth_l1(off, gt_off, mask, mask)
        off_loss = fluid.layers.reduce_sum(off_loss)
        total_num = fluid.layers.reduce_sum(gt_masks)
        total_num.stop_gradient = True
        return off_loss / (total_num + 1e-4)

    def get_loss(self, targets):
        gt_tl_heat = targets['tl_heatmaps']
        gt_br_heat = targets['br_heatmaps']
        gt_masks = targets['tag_masks']
        gt_tl_off = targets['tl_regrs']
        gt_br_off = targets['br_regrs']
        gt_tl_ind = targets['tl_tags']
        gt_br_ind = targets['br_tags']
        gt_masks = fluid.layers.cast(gt_masks, 'float32')

        focal_loss = 0
        focal_loss_ = self.focal_loss(self.tl_heats, gt_tl_heat, gt_masks)
        focal_loss += focal_loss_
        focal_loss_ = self.focal_loss(self.br_heats, gt_br_heat, gt_masks)
        focal_loss += focal_loss_

        pull_loss = 0
        push_loss = 0

        ones = fluid.layers.assign(np.array([1], dtype='float32'))
        tl_tags = [
            mask_feat(tl_tag, gt_tl_ind, self.train_batch_size)
            for tl_tag in self.tl_tags
        ]
        br_tags = [
            mask_feat(br_tag, gt_br_ind, self.train_batch_size)
            for br_tag in self.br_tags
        ]

        pull_loss, push_loss = 0, 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_masks)
            pull_loss += pull
            push_loss += push

        tl_offs = [
            mask_feat(tl_off, gt_tl_ind, self.train_batch_size)
            for tl_off in self.tl_offs
        ]
        br_offs = [
            mask_feat(br_off, gt_br_ind, self.train_batch_size)
            for br_off in self.br_offs
        ]

        off_loss = 0
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_masks)
            off_loss += self.off_loss(br_off, gt_br_off, gt_masks)

        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        loss = (
            focal_loss + pull_loss + push_loss + off_loss) / len(self.tl_heats)
        return {'loss': loss}

    def get_prediction(self, input):
        try:
            from ppdet.ext_op import cornerpool_lib
        except:
            logger.error(
                "cornerpool_lib not found, compile in ppdet/ext_op at first")
        ind = self.stack - 1
        tl_modules = corner_pool(
            input,
            256,
            cornerpool_lib.top_pool,
            cornerpool_lib.left_pool,
            is_test=True,
            name='tl_modules_' + str(ind))
        br_modules = corner_pool(
            input,
            256,
            cornerpool_lib.bottom_pool,
            cornerpool_lib.right_pool,
            is_test=True,
            name='br_modules_' + str(ind))
        tl_heat = self.pred_mod(
            tl_modules, self.num_classes, name='tl_heats_' + str(ind))
        br_heat = self.pred_mod(
            br_modules, self.num_classes, name='br_heats_' + str(ind))
        tl_tag = self.pred_mod(tl_modules, 1, name='tl_tags_' + str(ind))
        br_tag = self.pred_mod(br_modules, 1, name='br_tags_' + str(ind))

        tl_off = self.pred_mod(tl_modules, 2, name='tl_offs_' + str(ind))
        br_off = self.pred_mod(br_modules, 2, name='br_offs_' + str(ind))

        return decode(tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off,
                      self.ae_threshold, self.num_dets, self.K,
                      self.test_batch_size)
