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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import cycle, islice
from collections import abc
import paddle
import paddle.nn as nn

from ppdet.core.workspace import register, serializable

__all__ = ['HrHRNetLoss', 'KeyPointMSELoss']


@register
@serializable
class KeyPointMSELoss(nn.Layer):
    def __init__(self, use_target_weight=True, loss_scale=0.5):
        """
        KeyPointMSELoss layer

        Args:
            use_target_weight (bool): whether to use target weight
        """
        super(KeyPointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_scale = loss_scale

    def forward(self, output, records):
        target = records['target']
        target_weight = records['target_weight']
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(num_joints, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_joints, -1)).split(num_joints, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.loss_scale * self.criterion(
                    heatmap_pred.multiply(target_weight[:, idx]),
                    heatmap_gt.multiply(target_weight[:, idx]))
            else:
                loss += self.loss_scale * self.criterion(heatmap_pred,
                                                         heatmap_gt)
        keypoint_losses = dict()
        keypoint_losses['loss'] = loss / num_joints
        return keypoint_losses


@register
@serializable
class HrHRNetLoss(nn.Layer):
    def __init__(self, num_joints, swahr):
        """
        HrHRNetLoss layer

        Args:
            num_joints (int): number of keypoints
        """
        super(HrHRNetLoss, self).__init__()
        if swahr:
            self.heatmaploss = HeatMapSWAHRLoss(num_joints)
        else:
            self.heatmaploss = HeatMapLoss()
        self.aeloss = AELoss()
        self.ziploss = ZipLoss(
            [self.heatmaploss, self.heatmaploss, self.aeloss])

    def forward(self, inputs, records):
        targets = []
        targets.append([records['heatmap_gt1x'], records['mask_1x']])
        targets.append([records['heatmap_gt2x'], records['mask_2x']])
        targets.append(records['tagmap'])
        keypoint_losses = dict()
        loss = self.ziploss(inputs, targets)
        keypoint_losses['heatmap_loss'] = loss[0] + loss[1]
        keypoint_losses['pull_loss'] = loss[2][0]
        keypoint_losses['push_loss'] = loss[2][1]
        keypoint_losses['loss'] = recursive_sum(loss)
        return keypoint_losses


class HeatMapLoss(object):
    def __init__(self, loss_factor=1.0):
        super(HeatMapLoss, self).__init__()
        self.loss_factor = loss_factor

    def __call__(self, preds, targets):
        heatmap, mask = targets
        loss = ((preds - heatmap)**2 * mask.cast('float').unsqueeze(1))
        loss = paddle.clip(loss, min=0, max=2).mean()
        loss *= self.loss_factor
        return loss


class HeatMapSWAHRLoss(object):
    def __init__(self, num_joints, loss_factor=1.0):
        super(HeatMapSWAHRLoss, self).__init__()
        self.loss_factor = loss_factor
        self.num_joints = num_joints

    def __call__(self, preds, targets):
        heatmaps_gt, mask = targets
        heatmaps_pred = preds[0]
        scalemaps_pred = preds[1]

        heatmaps_scaled_gt = paddle.where(heatmaps_gt > 0, 0.5 * heatmaps_gt * (
            1 + (1 +
                 (scalemaps_pred - 1.) * paddle.log(heatmaps_gt + 1e-10))**2),
                                          heatmaps_gt)

        regularizer_loss = paddle.mean(
            paddle.pow((scalemaps_pred - 1.) * (heatmaps_gt > 0).astype(float),
                       2))
        omiga = 0.01
        # thres = 2**(-1/omiga), threshold for positive weight
        hm_weight = heatmaps_scaled_gt**(
            omiga
        ) * paddle.abs(1 - heatmaps_pred) + paddle.abs(heatmaps_pred) * (
            1 - heatmaps_scaled_gt**(omiga))

        loss = (((heatmaps_pred - heatmaps_scaled_gt)**2) *
                mask.cast('float').unsqueeze(1)) * hm_weight
        loss = loss.mean()
        loss = self.loss_factor * (loss + 1.0 * regularizer_loss)
        return loss


class AELoss(object):
    def __init__(self, pull_factor=0.001, push_factor=0.001):
        super(AELoss, self).__init__()
        self.pull_factor = pull_factor
        self.push_factor = push_factor

    def apply_single(self, pred, tagmap):
        if tagmap.numpy()[:, :, 3].sum() == 0:
            return (paddle.zeros([1]), paddle.zeros([1]))
        nonzero = paddle.nonzero(tagmap[:, :, 3] > 0)
        if nonzero.shape[0] == 0:
            return (paddle.zeros([1]), paddle.zeros([1]))
        p_inds = paddle.unique(nonzero[:, 0])
        num_person = p_inds.shape[0]
        if num_person == 0:
            return (paddle.zeros([1]), paddle.zeros([1]))

        pull = 0
        tagpull_num = 0
        embs_all = []
        person_unvalid = 0
        for person_idx in p_inds.numpy():
            valid_single = tagmap[person_idx.item()]
            validkpts = paddle.nonzero(valid_single[:, 3] > 0)
            valid_single = paddle.index_select(valid_single, validkpts)
            emb = paddle.gather_nd(pred, valid_single[:, :3])
            if emb.shape[0] == 1:
                person_unvalid += 1
            mean = paddle.mean(emb, axis=0)
            embs_all.append(mean)
            pull += paddle.mean(paddle.pow(emb - mean, 2), axis=0)
            tagpull_num += emb.shape[0]
        pull /= max(num_person - person_unvalid, 1)
        if num_person < 2:
            return pull, paddle.zeros([1])

        embs_all = paddle.stack(embs_all)
        A = embs_all.expand([num_person, num_person])
        B = A.transpose([1, 0])
        diff = A - B

        diff = paddle.pow(diff, 2)
        push = paddle.exp(-diff)
        push = paddle.sum(push) - num_person

        push /= 2 * num_person * (num_person - 1)
        return pull, push

    def __call__(self, preds, tagmaps):
        bs = preds.shape[0]
        losses = [
            self.apply_single(preds[i:i + 1].squeeze(),
                              tagmaps[i:i + 1].squeeze()) for i in range(bs)
        ]
        pull = self.pull_factor * sum(loss[0] for loss in losses) / len(losses)
        push = self.push_factor * sum(loss[1] for loss in losses) / len(losses)
        return pull, push


class ZipLoss(object):
    def __init__(self, loss_funcs):
        super(ZipLoss, self).__init__()
        self.loss_funcs = loss_funcs

    def __call__(self, inputs, targets):
        assert len(self.loss_funcs) == len(targets) >= len(inputs)

        def zip_repeat(*args):
            longest = max(map(len, args))
            filled = [islice(cycle(x), longest) for x in args]
            return zip(*filled)

        return tuple(
            fn(x, y)
            for x, y, fn in zip_repeat(inputs, targets, self.loss_funcs))


def recursive_sum(inputs):
    if isinstance(inputs, abc.Sequence):
        return sum([recursive_sum(x) for x in inputs])
    return inputs
