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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .centernet_head import ConvLayer
from ..keypoint_utils import get_affine_transform

__all__ = ['CenterTrackHead']


@register
class CenterTrackHead(nn.Layer):
    """
    Args:
        in_channels (int): the channel number of input to CenterNetHead.
        num_classes (int): the number of classes, 1 (MOT17 dataset) by default.
        head_planes (int): the channel number in all head, 256 by default.
        task (str): the type of task for regression, 'tracking' by default.
        loss_weight (dict): the weight of each loss.
        add_ltrb_amodal (bool): whether to add ltrb_amodal branch, False by default.
    """

    __shared__ = ['num_classes']

    def __init__(self,
                 in_channels,
                 num_classes=1,
                 head_planes=256,
                 task='tracking',
                 loss_weight={
                     'tracking': 1.0,
                     'ltrb_amodal': 0.1,
                 },
                 add_ltrb_amodal=True):
        super(CenterTrackHead, self).__init__()
        self.task = task
        self.loss_weight = loss_weight
        self.add_ltrb_amodal = add_ltrb_amodal

        # tracking head
        self.tracking = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            ConvLayer(
                head_planes, 2, kernel_size=1, stride=1, padding=0, bias=True))

        # ltrb_amodal head
        if self.add_ltrb_amodal and 'ltrb_amodal' in self.loss_weight:
            self.ltrb_amodal = nn.Sequential(
                ConvLayer(
                    in_channels,
                    head_planes,
                    kernel_size=3,
                    padding=1,
                    bias=True),
                nn.ReLU(),
                ConvLayer(
                    head_planes,
                    4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))

        # TODO: add more tasks

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self,
                feat,
                inputs,
                bboxes=None,
                bbox_inds=None,
                topk_clses=None,
                topk_ys=None,
                topk_xs=None):
        tracking = self.tracking(feat)
        head_outs = {'tracking': tracking}
        if self.add_ltrb_amodal and 'ltrb_amodal' in self.loss_weight:
            ltrb_amodal = self.ltrb_amodal(feat)
            head_outs.update({'ltrb_amodal': ltrb_amodal})

        if self.training:
            losses = self.get_loss(inputs, self.loss_weight, head_outs)
            return losses
        else:
            ret = self.generic_decode(head_outs, bboxes, bbox_inds, topk_ys,
                                      topk_xs)
            return ret

    def get_loss(self, inputs, weights, head_outs):
        index = inputs['index'].unsqueeze(2)
        mask = inputs['index_mask'].unsqueeze(2)
        batch_inds = list()
        for i in range(head_outs['tracking'].shape[0]):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)

        # 1.tracking head loss: L1 loss
        tracking = head_outs['tracking'].transpose([0, 2, 3, 1])
        tracking_target = inputs['tracking']
        bs, _, _, c = tracking.shape
        tracking = tracking.reshape([bs, -1, c])
        pos_tracking = paddle.gather_nd(tracking, index=index)
        tracking_mask = paddle.cast(
            paddle.expand_as(mask, pos_tracking), dtype=pos_tracking.dtype)
        pos_num = tracking_mask.sum()
        tracking_mask.stop_gradient = True
        tracking_target.stop_gradient = True
        tracking_loss = F.l1_loss(
            pos_tracking * tracking_mask,
            tracking_target * tracking_mask,
            reduction='sum')
        tracking_loss = tracking_loss / (pos_num + 1e-4)

        # 2.ltrb_amodal head loss(optinal): L1 loss
        if self.add_ltrb_amodal and 'ltrb_amodal' in self.loss_weight:
            ltrb_amodal = head_outs['ltrb_amodal'].transpose([0, 2, 3, 1])
            ltrb_amodal_target = inputs['ltrb_amodal']
            bs, _, _, c = ltrb_amodal.shape
            ltrb_amodal = ltrb_amodal.reshape([bs, -1, c])
            pos_ltrb_amodal = paddle.gather_nd(ltrb_amodal, index=index)
            ltrb_amodal_mask = paddle.cast(
                paddle.expand_as(mask, pos_ltrb_amodal),
                dtype=pos_ltrb_amodal.dtype)
            pos_num = ltrb_amodal_mask.sum()
            ltrb_amodal_mask.stop_gradient = True
            ltrb_amodal_target.stop_gradient = True
            ltrb_amodal_loss = F.l1_loss(
                pos_ltrb_amodal * ltrb_amodal_mask,
                ltrb_amodal_target * ltrb_amodal_mask,
                reduction='sum')
            ltrb_amodal_loss = ltrb_amodal_loss / (pos_num + 1e-4)

        losses = {'tracking_loss': tracking_loss, }
        plugin_loss = weights['tracking'] * tracking_loss

        if self.add_ltrb_amodal and 'ltrb_amodal' in self.loss_weight:
            losses.update({'ltrb_amodal_loss': ltrb_amodal_loss})
            plugin_loss += weights['ltrb_amodal'] * ltrb_amodal_loss
        losses.update({'plugin_loss': plugin_loss})
        return losses

    def generic_decode(self, head_outs, bboxes, bbox_inds, topk_ys, topk_xs):
        topk_ys = paddle.floor(topk_ys)  # note: More accurate
        topk_xs = paddle.floor(topk_xs)
        cts = paddle.concat([topk_xs, topk_ys], 1)
        ret = {'bboxes': bboxes, 'cts': cts}

        regression_heads = ['tracking']  # todo: add more tasks
        for head in regression_heads:
            if head in head_outs:
                ret[head] = _tranpose_and_gather_feat(head_outs[head],
                                                      bbox_inds)

        if 'ltrb_amodal' in head_outs:
            ltrb_amodal = head_outs['ltrb_amodal']
            ltrb_amodal = _tranpose_and_gather_feat(ltrb_amodal, bbox_inds)
            bboxes_amodal = paddle.concat(
                [
                    topk_xs * 1.0 + ltrb_amodal[..., 0:1],
                    topk_ys * 1.0 + ltrb_amodal[..., 1:2],
                    topk_xs * 1.0 + ltrb_amodal[..., 2:3],
                    topk_ys * 1.0 + ltrb_amodal[..., 3:4]
                ],
                axis=1)
            ret['bboxes'] = paddle.concat([bboxes[:, 0:2], bboxes_amodal], 1)
            # cls_id, score, x0, y0, x1, y1

        return ret

    def centertrack_post_process(self, dets, meta, out_thresh):
        if not ('bboxes' in dets):
            return [{}]

        preds = []
        c, s = meta['center'].numpy(), meta['scale'].numpy()
        h, w = meta['out_height'].numpy(), meta['out_width'].numpy()
        trans = get_affine_transform(
            center=c[0],
            input_size=s[0],
            rot=0,
            output_size=[w[0], h[0]],
            shift=(0., 0.),
            inv=True).astype(np.float32)
        for i, dets_bbox in enumerate(dets['bboxes']):
            if dets_bbox[1] < out_thresh:
                break
            item = {}
            item['score'] = dets_bbox[1]
            item['class'] = int(dets_bbox[0]) + 1
            item['ct'] = transform_preds_with_trans(
                dets['cts'][i].reshape([1, 2]), trans).reshape(2)

            if 'tracking' in dets:
                tracking = transform_preds_with_trans(
                    (dets['tracking'][i] + dets['cts'][i]).reshape([1, 2]),
                    trans).reshape(2)
                item['tracking'] = tracking - item['ct']

            if 'bboxes' in dets:
                bbox = transform_preds_with_trans(
                    dets_bbox[2:6].reshape([2, 2]), trans).reshape(4)
                item['bbox'] = bbox

            preds.append(item)
        return preds


def transform_preds_with_trans(coords, trans):
    target_coords = np.ones((coords.shape[0], 3), np.float32)
    target_coords[:, :2] = coords
    target_coords = np.dot(trans, target_coords.transpose()).transpose()
    return target_coords[:, :2]


def _tranpose_and_gather_feat(feat, bbox_inds):
    feat = feat.transpose([0, 2, 3, 1])
    feat = feat.reshape([-1, feat.shape[3]])
    feat = paddle.gather(feat, bbox_inds)
    return feat
