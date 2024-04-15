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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from .. import layers as L

__all__ = ['METRO_Body']


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.reshape((-1, 1, 3))
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = paddle.shape(X_trans)
    X_2d = (camera[:, :, 0] * X_trans.reshape((shape[0], -1))).reshape(shape)
    return X_2d


@register
class METRO_Body(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(
            self,
            num_joints,
            backbone='HRNet',
            trans_encoder='',
            loss='Pose3DLoss', ):
        """
        Modified from METRO network, see https://arxiv.org/abs/2012.09760

        Args:
            backbone (nn.Layer): backbone instance
        """
        super(METRO_Body, self).__init__()
        self.num_joints = num_joints
        self.backbone = backbone
        self.loss = loss
        self.deploy = False

        self.trans_encoder = trans_encoder
        self.conv_learn_tokens = paddle.nn.Conv1D(49, num_joints + 10, 1)
        self.cam_param_fc = paddle.nn.Linear(3, 2)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        trans_encoder = create(cfg['trans_encoder'])

        return {'backbone': backbone, 'trans_encoder': trans_encoder}

    def _forward(self):
        batch_size = self.inputs['image'].shape[0]

        image_feat = self.backbone(self.inputs)
        image_feat_flatten = image_feat.reshape((batch_size, 2048, 49))
        image_feat_flatten = image_feat_flatten.transpose(perm=(0, 2, 1))
        # and apply a conv layer to learn image token for each 3d joint/vertex position
        features = self.conv_learn_tokens(image_feat_flatten)  # (B, J, C)

        if self.training:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            meta_masks = self.inputs['mjm_mask'].expand((-1, -1, 2048))
            constant_tensor = paddle.ones_like(features) * 0.01
            features = features * meta_masks + constant_tensor * (1 - meta_masks
                                                                  )
        pred_out = self.trans_encoder(features)

        pred_3d_joints = pred_out[:, :self.num_joints, :]
        cam_features = pred_out[:, self.num_joints:, :]

        # learn camera parameters
        pred_2d_joints = self.cam_param_fc(cam_features)
        return pred_3d_joints, pred_2d_joints

    def get_loss(self):
        preds_3d, preds_2d = self._forward()
        loss = self.loss(preds_3d, preds_2d, self.inputs)
        output = {'loss': loss}
        return output

    def get_pred(self):
        preds_3d, preds_2d = self._forward()
        outputs = {'pose3d': preds_3d, 'pose2d': preds_2d}
        return outputs
