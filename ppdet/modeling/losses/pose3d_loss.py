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

from itertools import cycle, islice
from collections import abc
import paddle
import paddle.nn as nn
import numpy as np

from ppdet.core.workspace import register, serializable

__all__ = ['Pose3DLoss', 'KeyPointRegressionMSELoss']


@register
@serializable
class KeyPointRegressionMSELoss(nn.Layer):
    def __init__(self, reduction='none'):
        """
        KeyPointRegressionMSELoss layer
        """
        super(KeyPointRegressionMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def multi_frames_weights(self, data):
        # [64, 3, 24, 3]
        data = paddle.sum(paddle.abs(data), axis=-1, keepdim=True)
        data_max = paddle.max(data, axis=-1, keepdim=True)
        weights = data / (data_max + 1e-6) + 0.01
        return weights

    def l1(self, pre, target, weights=None):
        if weights is not None:
            pre = pre * weights
            target = target * weights

        return paddle.sum(paddle.abs(pre - target))

    def forward(self, output, records):
        gt_joints = records['kps3d']

        bs = gt_joints.shape[0]

        gt_joints_merge = paddle.reshape(
            gt_joints, (gt_joints.shape[0] * gt_joints.shape[1],
                        gt_joints.shape[2], gt_joints.shape[3]))

        pred_joints = output

        pred_joints_unmerge = paddle.reshape(pred_joints, gt_joints.shape)

        loss = paddle.sqrt(((pred_joints - gt_joints_merge)**2).sum(
            axis=-1)).mean()  # 3

        # frame sequence loss
        gt_disp_3d = gt_joints[:, 1::
                               2, :, :] - gt_joints[:, 0::
                                                    2, :, :]  # [32, 3, 24, 3]
        pred_disp_3d = pred_joints_unmerge[:, 1::
                                           2, :, :] - pred_joints_unmerge[:, 0::
                                                                          2, :, :]

        w_disp_3d = self.multi_frames_weights(gt_disp_3d)

        loss_disp_3d = self.l1(pred_disp_3d, gt_disp_3d,
                               weights=w_disp_3d) / bs  # 9

        final_loss = loss + loss_disp_3d

        #print("final_loss:",final_loss,loss_disp_3d_regs,loss_disp_3d)

        keypoint_losses = dict()
        keypoint_losses['loss'] = final_loss
        return keypoint_losses


@register
@serializable
class Pose3DLoss(nn.Layer):
    def __init__(self, weight_3d=1.0, weight_2d=0.0, reduction='none'):
        """
        KeyPointMSELoss layer

        Args:
            weight_3d (float): weight of 3d loss
            weight_2d (float): weight of 2d loss
            reduction (bool): whether use reduction to loss
        """
        super(Pose3DLoss, self).__init__()
        self.weight_3d = weight_3d
        self.weight_2d = weight_2d
        self.criterion_2dpose = nn.MSELoss(reduction=reduction)
        self.criterion_3dpose = nn.MSELoss(reduction=reduction)
        self.criterion_smoothl1 = nn.SmoothL1Loss(
            reduction=reduction, delta=1.0)
        self.criterion_vertices = nn.L1Loss()

    def forward(self, pred3d, pred2d, inputs):
        """
        mpjpe: mpjpe loss between 3d joints
        keypoint_2d_loss: 2d joints loss compute by criterion_2dpose
        """
        gt_3d_joints = inputs['joints_3d']  # [batch_size, num_joints, 3]
        if 'joints_2d' in inputs.keys():
            gt_2d_joints = inputs['joints_2d']
        else:
            gt_2d_joints = None
        has_3d_joints = inputs['has_3d_joints']
        has_2d_joints = inputs['has_2d_joints']

        loss_3d = mpjpe(pred3d, gt_3d_joints, has_3d_joints)

        #print("loss_3d:",loss_3d)

        if np.any(has_2d_joints.numpy()) and 'joints_2d' in inputs.keys():
            loss_2d = keypoint_2d_loss(self.criterion_2dpose, pred2d,
                                       gt_2d_joints, has_2d_joints)
        else:
            loss_2d = 0

        loss = self.weight_3d * loss_3d + self.weight_2d * loss_2d

        result = dict()
        result['loss'] = loss

        return result


def filter_3d_joints(pred, gt, has_3d_joints):
    """ 
    filter 3d joints
    """

    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :3]
    pred = pred[has_3d_joints == 1]

    gt_pelvis = (gt[:, 2, :] + gt[:, 3, :]) / 2
    gt = gt - gt_pelvis[:, None, :]
    pred_pelvis = (pred[:, 2, :] + pred[:, 3, :]) / 2
    pred = pred - pred_pelvis[:, None, :]
    return pred, gt


@register
@serializable
def mpjpe(pred, gt, has_3d_joints):
    """ 
    mPJPE loss
    """
    #pred, gt = filter_3d_joints(pred, gt, has_3d_joints)
    #print("pred:",pred[0])
    #print("gt:",gt[0])
    #print("in mpjpe:",((pred - gt)**2).sum(axis=-1))

    error = paddle.sqrt(((pred - gt)**2).sum(axis=-1)).mean()
    return error


@register
@serializable
def mpjpe_criterion(pred, gt, has_3d_joints, criterion_pose3d):
    """ 
    mPJPE loss of self define criterion
    """
    pred, gt = filter_3d_joints(pred, gt, has_3d_joints)
    error = paddle.sqrt(criterion_pose3d(pred, gt).sum(axis=-1)).mean()
    return error


@register
@serializable
def weighted_mpjpe(pred, gt, has_3d_joints):
    """ 
    Weighted_mPJPE
    """
    pred, gt = filter_3d_joints(pred, gt, has_3d_joints)
    weight = paddle.linalg.norm(pred, p=2, axis=-1)
    weight = paddle.to_tensor(
        [1.5, 1.3, 1.2, 1.2, 1.3, 1.5, 1.5, 1.3, 1.2, 1.2, 1.3, 1.5, 1., 1.])
    error = (weight * paddle.linalg.norm(pred - gt, p=2, axis=-1)).mean()
    return error


@register
@serializable
def normed_mpjpe(pred, gt, has_3d_joints):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert pred.shape == gt.shape
    pred, gt = filter_3d_joints(pred, gt, has_3d_joints)

    norm_predicted = paddle.mean(
        paddle.sum(pred**2, axis=3, keepdim=True), axis=2, keepdim=True)
    norm_target = paddle.mean(
        paddle.sum(gt * pred, axis=3, keepdim=True), axis=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * pred, gt)


@register
@serializable
def mpjpe_np(pred, gt, has_3d_joints):
    """ 
    mPJPE_NP
    """
    pred, gt = filter_3d_joints(pred, gt, has_3d_joints)
    error = np.sqrt(((pred - gt)**2).sum(axis=-1)).mean()
    return error


@register
@serializable
def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with paddle.no_grad():
        error = paddle.sqrt(((pred - gt)**2).sum(axis=-1)).mean()
        return error


@register
@serializable
def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d,
                     has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d,
                                       gt_keypoints_2d[:, :, :-1])).mean()
    return loss


@register
@serializable
def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d,
                     has_pose_3d):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (
            pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d,
                                           gt_keypoints_3d)).mean()
    else:
        return paddle.to_tensor([1.]).fill_(0.)


@register
@serializable
def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape,
                                  gt_vertices_with_shape)
    else:
        return paddle.to_tensor([1.]).fill_(0.)


@register
@serializable
def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose
