# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
This code is based on https://github.com/danbochman/SORT/blob/danny_opencv/kalman_filter.py
"""

import numpy as np
from numpy import dot, zeros, eye
from numpy.linalg import inv

use_numba = True
try:
    import numba as nb

    @nb.njit(fastmath=True, cache=True)
    def nb_predict(x, F, P, Q):
        x = dot(F, x)
        P = dot(dot(F, P), F.T) + Q
        return x, P

    @nb.njit(fastmath=True, cache=True)
    def nb_update(x, z, H, P, R, _I):

        y = z - np.dot(H, x)
        PHT = dot(P, H.T)

        S = dot(H, PHT) + R
        K = dot(PHT, inv(S))

        x = x + dot(K, y)

        I_KH = _I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)
        return x, P
except:
    use_numba = False
    print(
        'Warning: Unable to use numba in PP-Tracking, please install numba, for example(python3.7): `pip install numba==0.56.4`'
    )
    pass


class OCSORTKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = zeros((dim_x, 1))
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self.F = eye(dim_x)
        self.H = zeros((dim_z, dim_x))
        self.R = eye(dim_z)
        self.M = zeros((dim_z, dim_z))

        self._I = eye(dim_x)

    def predict(self):
        if use_numba:
            self.x, self.P = nb_predict(self.x, self.F, self.P, self.Q)
        else:
            self.x = dot(self.F, self.x)
            self.P = dot(dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):

        if z is None:
            return

        if use_numba:
            self.x, self.P = nb_update(self.x, z, self.H, self.P, self.R,
                                       self._I)
        else:
            y = z - np.dot(self.H, self.x)
            PHT = dot(self.P, self.H.T)

            S = dot(self.H, PHT) + self.R
            K = dot(PHT, inv(S))

            self.x = self.x + dot(K, y)

            I_KH = self._I - dot(K, self.H)
            self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K, self.R), K.T)
