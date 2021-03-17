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

__all__ = ['JDEDetectionLoss', 'JDEEmbeddingLoss', 'JDELoss']


@register
class JDEDetectionLoss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=1):
        super(JDEDetectionLoss, self).__init__()
        self.num_classes = num_classes

    def det_loss(self, p_det, anchor, t_conf, t_box):
        pshape = paddle.shape(p_det)
        pshape.stop_gradient = True
        nB, nGh, nGw = pshape[0], pshape[-2], pshape[-1]
        nA = len(anchor)
        p_det = paddle.reshape(
            p_det, [nB, nA, self.num_classes + 5, nGh, nGw]).transpose(
                (0, 1, 3, 4, 2))  # [1, 4, 19, 34, 6]

        # 1. loss_conf: cross_entropy
        p_conf = p_det[:, :, :, :, 4:6]  # [1, 4, 19, 34, 2]
        p_conf_flatten = paddle.reshape(p_conf, [-1, 2])
        t_conf_flatten = t_conf.flatten()
        t_conf_flatten = paddle.cast(t_conf_flatten, dtype="int64")
        t_conf_flatten.stop_gradient = True
        loss_conf = F.cross_entropy(
            p_conf_flatten, t_conf_flatten, ignore_index=-1, reduction='mean')

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

        return loss_conf, loss_box

    def forward(self, det_outs, targets, anchors):
        assert len(det_outs) == len(anchors)
        loss_confs = []
        loss_boxes = []
        for i, (p_det, anchor) in enumerate(zip(det_outs, anchors)):
            t_conf = targets['tconf{}'.format(i)]
            t_box = targets['tbox{}'.format(i)]

            loss_conf, loss_box = self.det_loss(p_det, anchor, t_conf, t_box)
            loss_confs.append(loss_conf)
            loss_boxes.append(loss_box)
        return loss_confs, loss_boxes


@register
class JDEEmbeddingLoss(nn.Layer):
    def __init__(self, ):
        super(JDEEmbeddingLoss, self).__init__()

    def emb_loss(self, p_ide, t_conf, t_ide, emb_scale, classifier):
        emb_dim = p_ide.shape[1]
        p_ide = p_ide.transpose((0, 2, 3, 1))  # [1, 19, 34, 512]
        p_ide_flatten = paddle.reshape(p_ide, [-1, emb_dim])
        mask = t_conf > 0
        mask = paddle.cast(mask, dtype="int64")
        mask.stop_gradient = True
        emb_mask = mask.max(1).flatten()
        emb_mask_inds = paddle.nonzero(emb_mask > 0).flatten()
        emb_mask_inds.stop_gradient = True
        # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
        t_ide_flatten = t_ide.max(1).flatten()
        t_ide_flatten = paddle.cast(t_ide_flatten, dtype="int64")

        weight = paddle.to_tensor([1], dtype='float32')
        if emb_mask_inds.numel() == 0:
            weight = paddle.to_tensor([0], dtype='float32')
            # emb_mask_inds = paddle.to_tensor(
            #     [1], dtype='int64')  # rand select an index
            logits = p_ide_flatten[0]
            ide_target = -t_ide_flatten[0]
            # ide_target = paddle.gather(de_flattent_ide_flatten, emb_mask_inds)
            ide_target.stop_gradient = True
        else:
            embedding = paddle.gather(p_ide_flatten, emb_mask_inds)
            embedding = emb_scale * F.normalize(embedding)
            logits = classifier(embedding)

            ide_target = paddle.gather(t_ide_flatten, emb_mask_inds)
            ide_target.stop_gradient = True

        loss_ide = F.cross_entropy(
            logits, ide_target, ignore_index=-1, reduction='mean') * weight
        return loss_ide

    def forward(self, ide_outs, targets, emb_scale, classifier):
        loss_ides = []
        for i, p_ide in enumerate(ide_outs):
            t_conf = targets['tconf{}'.format(i)]
            t_ide = targets['tide{}'.format(i)]

            loss_ide = self.emb_loss(p_ide, t_conf, t_ide, emb_scale,
                                     classifier)
            loss_ides.append(loss_ide)
        return loss_ides


@register
class JDELoss(nn.Layer):
    def __init__(self, use_uncertainy=True):
        super(JDELoss, self).__init__()
        self.use_uncertainy = use_uncertainy

    def forward(self, loss_confs, loss_boxes, loss_ides, loss_params_cls,
                loss_params_reg, loss_params_ide):
        assert len(loss_confs) == len(loss_boxes) == len(loss_ides)
        assert len(loss_params_cls) == len(loss_params_reg) == len(
            loss_params_ide)
        assert len(loss_confs) == len(loss_params_cls)
        jde_losses = []
        for i, (loss_conf, loss_box, loss_ide, l_conf_p, l_box_p,
                l_ide_p) in enumerate(
                    zip(loss_confs, loss_boxes, loss_ides, loss_params_cls,
                        loss_params_reg, loss_params_ide)):

            if self.use_uncertainy:
                jde_loss = l_conf_p(loss_conf) + l_box_p(loss_box) + l_ide_p(
                    loss_ide)
            else:
                s_c, s_r, s_id = -4.15, -4.85, -2.3
                jde_loss = paddle.exp(-s_c) * loss_conf + paddle.exp(
                    -s_r) * loss_box + paddle.exp(-s_id) * loss_ide + (s_c + s_r
                                                                       + s_id)
                jde_loss *= 0.5
            jde_losses.append(jde_loss)

        loss_all = {
            "loss_conf": sum(loss_confs),
            "loss_box": sum(loss_boxes),
            "loss_ide": sum(loss_ides),
            "loss": sum(jde_losses),
            # "nTargets": sum(nTargets) / 3,
        }
        return loss_all
