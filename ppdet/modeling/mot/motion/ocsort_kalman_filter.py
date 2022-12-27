import numpy as np
from numpy import dot, zeros, eye
from numpy.linalg import inv
import numba as nb


class OCSORTKalmanFilter:
    '''
    Kalman filtering, also known as linear quadratic estimation (LQE), is an algorithm that uses a series of measurements
    observed over time, containing statistical noise and other inaccuracies,
    and produces estimates of unknown variables that tend to be more accurate than those based on a single measurement
    alone, by estimating a joint probability distribution over the variables for each time frame.
    '''

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

        self._I = eye(
            dim_x
        )  # This helps the I matrix to always be compatible to the state vector's dim

    def predict(self):
        '''
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        '''
        self.x, self.P = self._predict(self.x, self.F, self.P, self.Q)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _predict(x, F, P, Q):
        x = dot(F, x)
        P = dot(dot(F, P), F.T) + Q
        return x, P

    def update(self, z):
        '''
        At the time step k, this update step computes the posterior mean x and covariance P
        of the system state given a new measurement z.
        '''
        if z is None:
            return
        self.x, self.P = self._update(self.x, z, self.H, self.P, self.R,
                                      self._I)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _update(x, z, H, P, R, _I):
        # y = z - Hx (Residual between measurement and prediction)
        y = z - np.dot(H, x)
        PHT = dot(P, H.T)

        # S = HPH' + R (Project system uncertainty into measurement space)
        S = dot(H, PHT) + R

        # K = PH'S^-1  (map system uncertainty into Kalman gain)
        K = dot(PHT, inv(S))

        # x = x + Ky  (predict new x with residual scaled by the Kalman gain)
        x = x + dot(K, y)

        # P = (I-KH)P
        I_KH = _I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)
        return x, P
