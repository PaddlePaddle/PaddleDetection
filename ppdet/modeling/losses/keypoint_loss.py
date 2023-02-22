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
import numpy as np
import paddle
import paddle.nn as nn

from ppdet.core.workspace import register, serializable

__all__ = ['HrHRNetLoss', 'KeyPointMSELoss', 'OKSLoss', 'CenterFocalLoss', 'L1Loss']


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


def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas):
    if not kpt_gts.astype('bool').any():
        return kpt_preds.sum()*0
    
    sigmas = paddle.to_tensor(sigmas, dtype=kpt_preds.dtype)
    variances = (sigmas * 2)**2

    assert kpt_preds.shape[0] == kpt_gts.shape[0]
    kpt_preds = kpt_preds.reshape((-1, kpt_preds.shape[-1] // 2, 2))
    kpt_gts = kpt_gts.reshape((-1, kpt_gts.shape[-1] // 2, 2))

    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
        (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
    assert (kpt_valids.sum(-1) > 0).all()
    squared_distance0 = squared_distance / (
        kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance1 = paddle.exp(-squared_distance0)
    squared_distance1 = squared_distance1 * kpt_valids
    oks = squared_distance1.sum(axis=1) / kpt_valids.sum(axis=1)

    return oks


def oks_loss(pred,
             target,
             weight,
             valid=None,
             area=None,
             linear=False,
             sigmas=None,
             eps=1e-6,
             avg_factor=None, 
             reduction=None):
    """Oks loss.

    Computing the oks loss between a set of predicted poses and target poses.
    The loss is calculated as negative log of oks.

    Args:
        pred (Tensor): Predicted poses of format (x1, y1, x2, y2, ...),
            shape (n, K*2).
        target (Tensor): Corresponding gt poses, shape (n, K*2).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: Loss tensor.
    """
    oks = oks_overlaps(pred, target, valid, area, sigmas).clip(min=eps)
    if linear:
        loss = 1 - oks
    else:
        loss = -oks.log()

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.shape[0] == loss.shape[0]:
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.reshape((-1, 1))
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.reshape((loss.shape[0], -1))
        assert weight.ndim == loss.ndim
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = 1e-10
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')


    return loss

@register
@serializable
class OKSLoss(nn.Layer):
    """OKSLoss.

    Computing the oks loss between a set of predicted poses and target poses.

    Args:
        linear (bool): If True, use linear scale of loss instead of log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 linear=False,
                 num_keypoints=17,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(OKSLoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def forward(self,
                pred,
                target,
                valid,
                area,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            valid (Tensor): The visible flag of the target pose.
            area (Tensor): The area of the target pose.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not paddle.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * oks_loss(
            pred,
            target,
            weight,
            valid=valid,
            area=area,
            linear=self.linear,
            sigmas=self.sigmas,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


def center_focal_loss(pred, gt, weight=None, mask=None, avg_factor=None, reduction=None):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory.

    Args:
        pred (Tensor): The prediction with shape [bs, c, h, w].
        gt (Tensor): The learning target of the prediction in gaussian
            distribution, with shape [bs, c, h, w].
        mask (Tensor): The valid mask. Defaults to None.
    """
    if not gt.astype('bool').any():
        return pred.sum()*0
    pos_inds = gt.equal(1).astype('float32')
    if mask is None:
        neg_inds = gt.less_than(paddle.to_tensor([1], dtype='float32')).astype('float32')
    else:
        neg_inds = gt.less_than(paddle.to_tensor([1], dtype='float32')).astype('float32') * mask.equal(0).astype('float32')

    neg_weights = paddle.pow(1 - gt, 4)

    loss = 0

    pos_loss = paddle.log(pred) * paddle.pow(1 - pred, 2) * pos_inds
    neg_loss = paddle.log(1 - pred) * paddle.pow(pred, 2) * neg_weights * \
        neg_inds

    num_pos = pos_inds.astype('float32').sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.shape[0] == loss.shape[0]:
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.reshape((-1, 1))
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.reshape((loss.shape[0], -1))
        assert weight.ndim == loss.ndim
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = 1e-10
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss

@register
@serializable
class CenterFocalLoss(nn.Layer):
    """CenterFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 reduction='none',
                 loss_weight=1.0):
        super(CenterFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                mask=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction in gaussian
                distribution.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            mask (Tensor): The valid mask. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * center_focal_loss(
            pred,
            target,
            weight,
            mask=mask,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg

def l1_loss(pred, target, weight=None, reduction='mean', avg_factor=None):
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if not target.astype('bool').any():
        return pred.sum() * 0

    assert pred.shape == target.shape
    loss = paddle.abs(pred - target)

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.shape[0] == loss.shape[0]:
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.reshape((-1, 1))
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.reshape((loss.shape[0], -1))
        assert weight.ndim == loss.ndim
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = 1e-10
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')


    return loss

@register
@serializable
class L1Loss(nn.Layer):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

