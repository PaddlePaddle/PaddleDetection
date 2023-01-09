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
"""
This code is based on https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
"""

import numpy as np
import scipy.linalg

use_numba = True
try:
    import numba as nb

    @nb.njit(fastmath=True, cache=True)
    def nb_project(mean, covariance, std, _update_mat):
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(_update_mat, mean)
        covariance = np.dot(np.dot(_update_mat, covariance), _update_mat.T)
        return mean, covariance + innovation_cov

    @nb.njit(fastmath=True, cache=True)
    def nb_multi_predict(mean, covariance, motion_cov, motion_mat):
        mean = np.dot(mean, motion_mat.T)
        left = np.dot(motion_mat, covariance)
        covariance = np.dot(left, motion_mat.T) + motion_cov
        return mean, covariance

    @nb.njit(fastmath=True, cache=True)
    def nb_update(mean, covariance, proj_mean, proj_cov, measurement, meas_mat):
        kalman_gain = np.linalg.solve(proj_cov, (covariance @meas_mat.T).T).T
        innovation = measurement - proj_mean
        mean = mean + innovation @kalman_gain.T
        covariance = covariance - kalman_gain @proj_cov @kalman_gain.T
        return mean, covariance

except:
    use_numba = False
    print(
        'Warning: Unable to use numba in PP-Tracking, please install numba, for example(python3.7): `pip install numba==0.56.4`'
    )
    pass

__all__ = ['KalmanFilter']
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with
                center position (x, y), aspect ratio a, and height h.

        Returns:
            The mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are 
            initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3], 1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3], 1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, np.float32(covariance)

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state
                at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the
                object state at the previous time step.

        Returns:
            The mean vector and covariance matrix of the predicted state. 
            Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3], self._std_weight_position *
            mean[3], 1e-2, self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3], self._std_weight_velocity *
            mean[3], 1e-5, self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Args
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            The projected mean and covariance matrix of the given state estimate.
        """
        std = np.array(
            [
                self._std_weight_position * mean[3], self._std_weight_position *
                mean[3], 1e-1, self._std_weight_position * mean[3]
            ],
            dtype=np.float32)

        if use_numba:
            return nb_project(mean, covariance, std, self._update_mat)

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance,
                                          self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step (Vectorized version).
        
        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states
                at the previous time step.
            covariance (ndarray): The Nx8x8 dimensional covariance matrics of the
                object states at the previous time step.

        Returns:
            The mean vector and covariance matrix of the predicted state.
            Unobserved velocities are initialized to 0 mean.
        """
        std_pos = np.array([
            self._std_weight_position * mean[:, 3], self._std_weight_position *
            mean[:, 3], 1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ])
        std_vel = np.array([
            self._std_weight_velocity * mean[:, 3], self._std_weight_velocity *
            mean[:, 3], 1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ])
        sqr = np.square(np.r_[std_pos, std_vel]).T

        if use_numba:

            means = []
            covariances = []
            for i in range(len(mean)):
                a, b = nb_multi_predict(mean[i], covariance[i],
                                        np.diag(sqr[i]), self._motion_mat)
                means.append(a)
                covariances.append(b)
            return np.asarray(means), np.asarray(covariances)

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the aspect
                ratio, and h the height of the bounding box.

        Returns:
            The measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if use_numba:

            return nb_update(mean, covariance, projected_mean, projected_cov,
                             measurement, self._update_mat)

        kalman_gain = np.linalg.solve(projected_cov,
                                      (covariance @self._update_mat.T).T).T
        innovation = measurement - projected_mean
        mean = mean + innovation @kalman_gain.T
        covariance = covariance - kalman_gain @projected_cov @kalman_gain.T
        return mean, covariance

    def gating_distance(self,
                        mean,
                        covariance,
                        measurements,
                        only_position=False,
                        metric='maha'):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        
        Args:
            mean (ndarray): Mean vector over the state distribution (8
                dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8
                dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N measurements,
                each in format (x, y, a, h) where (x, y) is the bounding box center
                position, a the aspect ratio, and h the height.
            only_position (Optional[bool]): If True, distance computation is 
                done with respect to the bounding box center position only.
            metric (str): Metric type, 'gaussian' or 'maha'.

        Returns
            An array of length N, where the i-th element contains the squared
            Mahalanobis distance between (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor,
                d.T,
                lower=True,
                check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
