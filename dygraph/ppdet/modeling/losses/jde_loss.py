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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from IPython import embed
__all__ = ['JDELoss']


@register
class JDELoss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=1):
        super(JDELoss, self).__init__()
        self.num_classes = num_classes

    def jde_loss(self, p_det, p_ide, anchor, t_conf, t_box, t_ide, l_conf_p,
                 l_box_p, l_ide_p, emb_scale, classifier):
        pshape = paddle.shape(p_det)
        pshape.stop_gradient = True
        nB, nGh, nGw = pshape[0], pshape[-2], pshape[-1]
        nA = len(anchor)
        p_det = paddle.reshape(
            p_det, [nB, nA, self.num_classes + 5, nGh, nGw]).transpose(
                (0, 1, 3, 4, 2))  # [1, 4, 19, 34, 6]
        p_ide = p_ide.transpose((0, 2, 3, 1))  # [1, 19, 34, 512]

        loss = dict()
        # 1. loss_conf: cross_entropy
        p_conf = p_det[:, :, :, :, 4:6]  # [1, 4, 19, 34, 2]
        p_conf_flatten = paddle.reshape(p_conf, [-1, 2])
        t_conf_flatten = t_conf.flatten()
        t_conf_flatten = paddle.cast(t_conf_flatten, dtype="int64")
        t_conf_flatten.stop_gradient = True
        loss_conf = F.cross_entropy(
            p_conf_flatten, t_conf_flatten, ignore_index=-1, reduction='mean')
        loss['loss_conf'] = loss_conf

        # 2. loss_box: smooth_l1_loss
        p_box = p_det[:, :, :, :, :4]  # [1, 4, 19, 34, 4]
        p_box_flatten = paddle.reshape(p_box, [-1, 4])
        t_box_flatten = paddle.reshape(t_box, [-1, 4])
        fg_inds = paddle.nonzero(t_conf_flatten > 0).flatten()
        if fg_inds.numel() > 0:
            reg_delta = paddle.gather(p_box_flatten, fg_inds)
            reg_target = paddle.gather(t_box_flatten, fg_inds)
        else:
            reg_delta = paddle.to_tensor([0, 0, 0, 0], dtype='float32')
            reg_delta.stop_gradient = False
            reg_target = paddle.to_tensor([0, 0, 0, 0], dtype='float32')
        reg_target.stop_gradient = True
        loss_box = F.smooth_l1_loss(
            reg_delta, reg_target, reduction='mean', delta=1.0)
        loss['loss_box'] = loss_box

        # 3. loss_ide: cross_entropy
        p_ide_flatten = paddle.reshape(p_ide, [-1, 512])
        mask = t_conf > 0
        mask = paddle.cast(mask, dtype="int64")
        mask.stop_gradient = True
        emb_mask = mask.max(1).flatten()
        emb_mask_inds = paddle.nonzero(emb_mask > 0).flatten()
        emb_mask_inds.stop_gradient = True

        # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
        t_ide_flatten = t_ide.max(1).flatten()
        t_ide_flatten = paddle.cast(t_ide_flatten, dtype="int64")

        if emb_mask_inds.numel() == 0:
            # bug in some paddle version
            # embedding = p_ide_flatten[0]
            # ide_target = t_ide_flatten[0] * 2

            emb_mask_inds = paddle.to_tensor(
                [1], dtype='int64')  # rand select an index
            ide_target = paddle.gather(t_ide_flatten, emb_mask_inds) * 2
            # gt ide -1 is ignore_index, cross_entropy loss will be nan
        else:
            ide_target = paddle.gather(t_ide_flatten, emb_mask_inds)

        embedding = paddle.gather(p_ide_flatten, emb_mask_inds)
        logits = classifier(emb_scale * F.normalize(embedding))
        ide_target.stop_gradient = True
        loss_ide = F.cross_entropy(
            logits, ide_target, ignore_index=-1, reduction='mean')
        loss['loss_ide'] = loss_ide

        loss['loss'] = l_conf_p(loss_conf) + l_box_p(loss_box) + l_ide_p(
            loss_ide)
        return loss

    def forward(self, det_outs, ide_outs, targets, anchors, emb_scale,
                classifier, loss_conf_params, loss_box_params, loss_ide_params):
        assert len(det_outs) == len(ide_outs) == len(anchors)
        jde_losses = dict()
        for i, (p_det, p_ide, anchor, l_conf_p, l_box_p, l_ide_p) in enumerate(
                zip(det_outs, ide_outs, anchors, loss_conf_params,
                    loss_box_params, loss_ide_params)):
            t_conf = targets['tconf{}'.format(i)]
            t_box = targets['tbox{}'.format(i)]
            t_ide = targets['tide{}'.format(i)]

            jde_loss = self.jde_loss(p_det, p_ide, anchor, t_conf, t_box, t_ide,
                                     l_conf_p, l_box_p, l_ide_p, emb_scale,
                                     classifier)

            for k, v in jde_loss.items():
                if k in jde_losses:
                    jde_losses[k] += v
                else:
                    jde_losses[k] = v

        return jde_losses
