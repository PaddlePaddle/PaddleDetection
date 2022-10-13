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

__all__ = ['METRO_Body', 'METRO_Body_temp']


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
    # X_2d = (camera[:, :, 0] * X_trans.reshape((shape[0], -1))).reshape(shape)
    X_2d = (100 * (1 + camera[:, :, 0]) * X_trans.reshape(
        (shape[0], -1))).reshape(shape)
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
        METRO network, see https://arxiv.org/abs/

        Args:
            backbone (nn.Layer): backbone instance
        """
        super(METRO_Body, self).__init__()
        self.num_joints = num_joints
        self.backbone = backbone
        self.loss = loss
        self.deploy = False

        self.trans_encoder = trans_encoder
        self.conv_learn_tokens = paddle.nn.Conv1D(49, 10 + num_joints, 1)
        self.cam_param_fc = paddle.nn.Linear(3, 1)
        self.cam_param_fc2 = paddle.nn.Linear(10, 250)
        self.cam_param_fc3 = paddle.nn.Linear(250, 3)
        # self.temp_fc1 = paddle.nn.Linear(81, 1024)
        # self.temp_fc2 = paddle.nn.Linear(1024, 81)
        # self.rest_joints = paddle.Tensor(
        #     [[-9.1982e-02,  6.4817e-01,  8.3707e-02],
        #     [-1.0776e-01,  2.4976e-01,  4.1392e-02],
        #     [-1.3762e-01, -2.4549e-01,  2.1714e-02],
        #     [ 1.3390e-01, -2.4533e-01,  1.6776e-02],
        #     [ 1.0200e-01,  2.4327e-01,  3.9534e-02],
        #     [ 8.8406e-02,  6.4123e-01,  8.3230e-02],
        #     [-6.8420e-01, -6.6623e-01,  1.0311e-01],
        #     [-4.2890e-01, -6.5845e-01,  9.7555e-02],
        #     [-1.7516e-01, -6.7178e-01,  7.6154e-02],
        #     [ 1.7244e-01, -6.7262e-01,  7.1353e-02],
        #     [ 4.3205e-01, -6.5985e-01,  9.8809e-02],
        #     [ 6.8128e-01, -6.6883e-01,  9.9980e-02],
        #     [ 1.5762e-03, -7.3406e-01,  5.9082e-02],
        #     [-1.1176e-04, -1.0022e+00,  5.0133e-02],
        #     [-2.5436e-03, -2.4862e-01,  4.0846e-02],
        #     [ 1.4044e-03, -6.9365e-01,  7.3388e-02],
        #     [-6.3274e-03, -4.9322e-01,  7.2980e-02],
        #     [ 8.4006e-03, -8.2030e-01, -9.1983e-03],
        #     [-6.1768e-05, -9.3824e-01,  5.4323e-02],
        #     [ 1.2258e-03, -8.5931e-01, -8.0475e-02],
        #     [ 3.4353e-02, -8.9460e-01, -4.2776e-02],
        #     [-3.2222e-02, -8.9410e-01, -4.1220e-02],
        #     [ 7.4144e-02, -8.6779e-01,  4.8845e-02],
        #     [-7.2048e-02, -8.6516e-01,  5.0976e-02]]) #[1,24,3]

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
        #temp test
        # B,J,D = pred_out.shape
        # pred_out_temp = pred_out.reshape((-1, B, J*D))
        # inner_temp = self.temp_fc1(pred_out_temp)
        # pred_out_temp = self.temp_fc2(inner_temp)
        # pred_out = pred_out_temp.reshape((B,J,D))

        pred_3d_joints = pred_out[:, :self.num_joints, :]
        cam_features = pred_out[:, self.num_joints:, :]

        # learn camera parameters
        x = self.cam_param_fc(cam_features)
        x = x.transpose(perm=(0, 2, 1))
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(perm=(0, 2, 1))
        pred_camera = cam_param.squeeze()
        pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

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


@register
class METRO_Body_temp(METRO_Body):
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
        #temp test
        # B,J,D = pred_out.shape
        # pred_out_temp = pred_out.reshape((-1, B, J*D))
        # inner_temp = self.temp_fc1(pred_out_temp)
        # pred_out_temp = self.temp_fc2(inner_temp)
        # pred_out = pred_out_temp.reshape((B,J,D))

        pred_3d_joints = pred_out[:, :self.num_joints, :]
        cam_features = pred_out[:, self.num_joints:, :]

        # learn camera parameters
        x = self.cam_param_fc(cam_features)
        x = x.transpose(perm=(0, 2, 1))
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(perm=(0, 2, 1))
        pred_camera = cam_param.squeeze()
        pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

        return pred_3d_joints, pred_2d_joints
