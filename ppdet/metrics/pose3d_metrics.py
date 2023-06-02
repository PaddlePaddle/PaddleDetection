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

import paddle
from paddle.distributed import ParallelEnv
import os
import json
from collections import defaultdict, OrderedDict
import numpy as np
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['Pose3DEval']


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :3]
    pred = pred[has_3d_joints == 1]

    with paddle.no_grad():
        gt_pelvis = (gt[:, 2, :] + gt[:, 3, :]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2, :] + pred[:, 3, :]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = paddle.sqrt(((pred - gt)**2).sum(axis=-1)).mean(axis=-1).numpy()
        return error


def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt(((S1_hat - S2)**2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


def all_gather(data):
    if paddle.distributed.get_world_size() == 1:
        return data
    vlist = []
    paddle.distributed.all_gather(vlist, data)
    data = paddle.concat(vlist, 0)
    return data


class Pose3DEval(object):
    def __init__(self, output_eval, save_prediction_only=False):
        super(Pose3DEval, self).__init__()
        self.output_eval = output_eval
        self.res_file = os.path.join(output_eval, "pose3d_results.json")
        self.save_prediction_only = save_prediction_only
        self.reset()

    def reset(self):
        self.PAmPJPE = AverageMeter()
        self.mPJPE = AverageMeter()
        self.eval_results = {}

    def get_human36m_joints(self, input):
        J24_TO_J14 = paddle.to_tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18])
        J24_TO_J17 = paddle.to_tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19])
        return paddle.index_select(input, J24_TO_J14, axis=1)

    def update(self, inputs, outputs):
        gt_3d_joints = all_gather(inputs['joints_3d'].cuda(ParallelEnv()
                                                           .local_rank))
        has_3d_joints = all_gather(inputs['has_3d_joints'].cuda(ParallelEnv()
                                                                .local_rank))
        pred_3d_joints = all_gather(outputs['pose3d'])
        if gt_3d_joints.shape[1] == 24:
            gt_3d_joints = self.get_human36m_joints(gt_3d_joints)
        if pred_3d_joints.shape[1] == 24:
            pred_3d_joints = self.get_human36m_joints(pred_3d_joints)
        mPJPE_val = mean_per_joint_position_error(pred_3d_joints, gt_3d_joints,
                                                  has_3d_joints).mean()
        PAmPJPE_val = reconstruction_error(
            pred_3d_joints.numpy(),
            gt_3d_joints[:, :, :3].numpy(),
            reduction=None).mean()
        count = int(np.sum(has_3d_joints.numpy()))
        self.PAmPJPE.update(PAmPJPE_val * 1000., count)
        self.mPJPE.update(mPJPE_val * 1000., count)

    def accumulate(self):
        if self.save_prediction_only:
            logger.info(f'The pose3d result is saved to {self.res_file} '
                        'and do not evaluate the model.')
            return
        self.eval_results['pose3d'] = [-self.mPJPE.avg, -self.PAmPJPE.avg]

    def log(self):
        if self.save_prediction_only:
            return
        stats_names = ['mPJPE', 'PAmPJPE']
        num_values = len(stats_names)
        print(' '.join(['| {}'.format(name) for name in stats_names]) + ' |')
        print('|---' * (num_values + 1) + '|')

        print(' '.join([
            '| {:.3f}'.format(abs(value))
            for value in self.eval_results['pose3d']
        ]) + ' |')

    def get_results(self):
        return self.eval_results
