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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from paddle.nn.initializer import Normal, Constant

__all__ = ['JDEEmbeddingHead']


class LossParam(nn.Layer):
    def __init__(self, init_value=0., use_uncertainy=True):
        super(LossParam, self).__init__()
        self.loss_param = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=init_value)),
            dtype="float32")

    def forward(self, inputs):
        out = paddle.exp(-self.loss_param) * inputs + self.loss_param
        return out * 0.5


@register
class JDEEmbeddingHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['emb_loss', 'jde_loss']
    """
    JDEEmbeddingHead
    Args:
        num_classes(int): Number of classes. Only support one class tracking.
        num_identities(int): Number of identities.
        anchor_levels(int): Number of anchor levels, same as FPN levels.
        anchor_scales(int): Number of anchor scales on each FPN level.
        embedding_dim(int): Embedding dimension. Default: 512.
        emb_loss(object): Instance of 'JDEEmbeddingLoss'
        jde_loss(object): Instance of 'JDELoss'
    """

    def __init__(
            self,
            num_classes=1,
            num_identities=14455,  # dataset.num_identities_dict[0]
            anchor_levels=3,
            anchor_scales=4,
            embedding_dim=512,
            emb_loss='JDEEmbeddingLoss',
            jde_loss='JDELoss'):
        super(JDEEmbeddingHead, self).__init__()
        self.num_classes = num_classes
        self.num_identities = num_identities
        self.anchor_levels = anchor_levels
        self.anchor_scales = anchor_scales
        self.embedding_dim = embedding_dim
        self.emb_loss = emb_loss
        self.jde_loss = jde_loss

        self.emb_scale = math.sqrt(2) * math.log(
            self.num_identities - 1) if self.num_identities > 1 else 1

        self.identify_outputs = []
        self.loss_params_cls = []
        self.loss_params_reg = []
        self.loss_params_ide = []
        for i in range(self.anchor_levels):
            name = 'identify_output.{}'.format(i)
            identify_output = self.add_sublayer(
                name,
                nn.Conv2D(
                    in_channels=64 * (2**self.anchor_levels) // (2**i),
                    out_channels=self.embedding_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.))))
            self.identify_outputs.append(identify_output)

            loss_p_cls = self.add_sublayer('cls.{}'.format(i), LossParam(-4.15))
            self.loss_params_cls.append(loss_p_cls)
            loss_p_reg = self.add_sublayer('reg.{}'.format(i), LossParam(-4.85))
            self.loss_params_reg.append(loss_p_reg)
            loss_p_ide = self.add_sublayer('ide.{}'.format(i), LossParam(-2.3))
            self.loss_params_ide.append(loss_p_ide)

        self.classifier = self.add_sublayer(
            'classifier',
            nn.Linear(
                self.embedding_dim,
                self.num_identities,
                weight_attr=ParamAttr(
                    learning_rate=1., initializer=Normal(
                        mean=0.0, std=0.01)),
                bias_attr=ParamAttr(
                    learning_rate=2., regularizer=L2Decay(0.))))

    def forward(self,
                identify_feats,
                targets,
                loss_confs=None,
                loss_boxes=None,
                bboxes=None,
                boxes_idx=None,
                nms_keep_idx=None):
        assert self.num_classes == 1, 'JDE only support sindle class MOT.'
        assert len(identify_feats) == self.anchor_levels
        ide_outs = []
        for feat, ide_head in zip(identify_feats, self.identify_outputs):
            ide_outs.append(ide_head(feat))

        if self.training:
            assert len(loss_confs) == len(loss_boxes) == self.anchor_levels
            loss_ides = self.emb_loss(ide_outs, targets, self.emb_scale,
                                      self.classifier)
            jde_losses = self.jde_loss(
                loss_confs, loss_boxes, loss_ides, self.loss_params_cls,
                self.loss_params_reg, self.loss_params_ide, targets)
            return jde_losses
        else:
            assert bboxes is not None
            assert boxes_idx is not None
            assert nms_keep_idx is not None

            emb_outs = self.get_emb_outs(ide_outs)
            emb_valid = paddle.gather_nd(emb_outs, boxes_idx)
            pred_embs = paddle.gather_nd(emb_valid, nms_keep_idx)

            input_shape = targets['image'].shape[2:]
            # input_shape: [h, w], before data transforms, set in model config
            im_shape = targets['im_shape'][0].numpy()
            # im_shape: [new_h, new_w], after data transforms
            scale_factor = targets['scale_factor'][0].numpy()
            bboxes[:, 2:] = self.scale_coords(bboxes[:, 2:], input_shape,
                                              im_shape, scale_factor)
            # cls_ids, scores, tlwhs 
            pred_dets = bboxes
            return pred_dets, pred_embs

    def scale_coords(self, coords, input_shape, im_shape, scale_factor):
        ratio = scale_factor[0]
        pad_w = (input_shape[1] - int(im_shape[1])) / 2
        pad_h = (input_shape[0] - int(im_shape[0])) / 2
        coords = paddle.cast(coords, 'float32')
        coords[:, 0::2] -= pad_w
        coords[:, 1::2] -= pad_h
        coords[:, 0:4] /= ratio
        coords[:, :4] = paddle.clip(
            coords[:, :4], min=0, max=coords[:, :4].max())
        return coords.round()

    def get_emb_and_gt_outs(self, ide_outs, targets):
        emb_and_gts = []
        for i, p_ide in enumerate(ide_outs):
            t_conf = targets['tconf{}'.format(i)]
            t_ide = targets['tide{}'.format(i)]

            p_ide = p_ide.transpose((0, 2, 3, 1))
            p_ide_flatten = paddle.reshape(p_ide, [-1, self.embedding_dim])

            mask = t_conf > 0
            mask = paddle.cast(mask, dtype="int64")
            emb_mask = mask.max(1).flatten()
            emb_mask_inds = paddle.nonzero(emb_mask > 0).flatten()
            if len(emb_mask_inds) > 0:
                t_ide_flatten = paddle.reshape(t_ide.max(1), [-1, 1])
                tids = paddle.gather(t_ide_flatten, emb_mask_inds)

                embedding = paddle.gather(p_ide_flatten, emb_mask_inds)
                embedding = self.emb_scale * F.normalize(embedding)
                emb_and_gt = paddle.concat([embedding, tids], axis=1)
                emb_and_gts.append(emb_and_gt)

        if len(emb_and_gts) > 0:
            return paddle.concat(emb_and_gts, axis=0)
        else:
            return paddle.zeros((1, self.embedding_dim + 1))

    def get_emb_outs(self, ide_outs):
        emb_outs = []
        for i, p_ide in enumerate(ide_outs):
            p_ide = p_ide.transpose((0, 2, 3, 1))

            p_ide_repeat = paddle.tile(p_ide, [self.anchor_scales, 1, 1, 1])
            embedding = F.normalize(p_ide_repeat, axis=-1)
            emb = paddle.reshape(embedding, [-1, self.embedding_dim])
            emb_outs.append(emb)

        if len(emb_outs) > 0:
            return paddle.concat(emb_outs, axis=0)
        else:
            return paddle.zeros((1, self.embedding_dim))
