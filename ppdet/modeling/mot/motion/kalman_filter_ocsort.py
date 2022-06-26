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

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z


class KalmanFilterNew(object):
    """ Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults  will
    not give you a functional filter.
    For now the best documentation is my free book Kalman and Bayesian
    Filters in Python [2]_. The test files in this directory also give you a
    basic idea of use, albeit without much description.
    In brief, you will first construct this object, specifying the size of
    the state vector with dim_x and the size of the measurement vector that
    you will be using with dim_z. These are mostly used to perform size checks
    when you assign values to the various matrices. For example, if you
    specified dim_z=2 and then try to assign a 3x3 matrix to R (the
    measurement noise matrix you will get an assert exception because R
    should be 2x2. (If for whatever reason you need to alter the size of
    things midstream just use the underscore version of the matrices to
    assign directly: your_filter._R = a_3x3_matrix.)
    After construction the filter will have default matrices created for you,
    but you must specify the values for each. It’s usually easiest to just
    overwrite them rather than assign to each element yourself. This will be
    clearer in the example below. All are of type numpy.array.
    Examples
    --------
    Here is a filter that tracks position and velocity using a sensor that only
    reads position.
    First construct the object with the required dimensionality. Here the state
    (`dim_x`) has 2 coefficients (position and velocity), and the measurement
    (`dim_z`) has one. In FilterPy `x` is the state, `z` is the measurement.
    .. code::
        from filterpy.kalman import KalmanFilter
        f = KalmanFilter (dim_x=2, dim_z=1)
    Assign the initial value for the state (position and velocity). You can do this
    with a two dimensional array like so:
        .. code::
            f.x = np.array([[2.],    # position
                            [0.]])   # velocity
    or just use a one dimensional array, which I prefer doing.
    .. code::
        f.x = np.array([2., 0.])
    Define the state transition matrix:
        .. code::
            f.F = np.array([[1.,1.],
                            [0.,1.]])
    Define the measurement function. Here we need to convert a position-velocity
    vector into just a position vector, so we use:
        .. code::
        f.H = np.array([[1., 0.]])
    Define the state's covariance matrix P. 
    .. code::
        f.P = np.array([[1000.,    0.],
                        [   0., 1000.] ])
    Now assign the measurement noise. Here the dimension is 1x1, so I can
    use a scalar
    .. code::
        f.R = 5
    I could have done this instead:
    .. code::
        f.R = np.array([[5.]])
    Note that this must be a 2 dimensional array.
    Finally, I will assign the process noise. Here I will take advantage of
    another FilterPy library function:
    .. code::
        from filterpy.common import Q_discrete_white_noise
        f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    Now just perform the standard predict/update loop:
    .. code::
        while some_condition_is_true:
            z = get_sensor_reading()
            f.predict()
            f.update(z)
            do_something_with_estimate (f.x)
    **Procedural Form**
    This module also contains stand alone functions to perform Kalman filtering.
    Use these if you are not a fan of objects.
    **Example**
    .. code::
        while True:
            z, R = read_sensor()
            x, P = predict(x, P, F, Q)
            x, P = update(x, P, z, R, H)
    See my book Kalman and Bayesian Filters in Python [2]_.
    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    dim_u : int (optional)
        size of the control input, if it is being used.
        Default value of 0 indicates it is not used.
    compute_log_likelihood : bool (default = True)
        Computes log likelihood by default, but this can be a slow
        computation, so if you never use it you can turn this computation
        off.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.
    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.
    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convenience; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.
    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    z : numpy.array
        Last measurement used in update(). Read only.
    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix. Also known as the
        observation covariance.
    Q : numpy.array(dim_x, dim_x)
        Process noise covariance matrix. Also known as the transition
        covariance.
    F : numpy.array()
        State Transition matrix. Also known as `A` in some formulation.
    H : numpy.array(dim_z, dim_x)
        Measurement function. Also known as the observation matrix, or as `C`.
    y : numpy.array
        Residual of the update step. Read only.
    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.
    S :  numpy.array
        System uncertainty (P projected to measurement space). Read only.
    SI :  numpy.array
        Inverse system uncertainty. Read only.
    log_likelihood : float
        log-likelihood of the last measurement. Read only.
    likelihood : float
        likelihood of last measurement. Read only.
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
    mahalanobis : float
        mahalanobis distance of the innovation. Read only.
    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv
        This is only used to invert self.S. If you know it is diagonal, you
        might choose to set it to filterpy.common.inv_diagonal, which is
        several times faster than numpy.linalg.inv for diagonal matrices.
    alpha : float
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
    References
    ----------
    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)
    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))  # state
        self.P = eye(dim_x)  # uncertainty covariance
        self.Q = eye(dim_x)  # process uncertainty
        self.B = None  # control transition matrix
        self.F = eye(dim_x)  # state transition matrix
        self.H = zeros((dim_z, dim_x))  # measurement function
        self.R = eye(dim_z)  # measurement uncertainty
        self._alpha_sq = 1.  # fading memory control
        self.M = np.zeros(
            (dim_x, dim_z))  # process-measurement cross correlation
        self.z = np.array([[None] * self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # keep all observations 
        self.history_obs = []

        self.inv = np.linalg.inv

        self.attr_saved = None
        self.observed = False

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def freeze(self):
        """
            Save the parameters before non-observation forward
        """
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self):
        if self.attr_saved is not None:
            new_history = deepcopy(self.history_obs)
            self.__dict__ = self.attr_saved
            # self.history_obs = new_history 
            self.history_obs = self.history_obs[:-1]
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            index1 = indices[-2]
            index2 = indices[-1]
            box1 = new_history[index1]
            x1, y1, s1, r1 = box1
            w1 = np.sqrt(s1 * r1)
            h1 = np.sqrt(s1 / r1)
            box2 = new_history[index2]
            x2, y2, s2, r2 = box2
            w2 = np.sqrt(s2 * r2)
            h2 = np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx = (x2 - x1) / time_gap
            dy = (y2 - y1) / time_gap
            dw = (w2 - w1) / time_gap
            dh = (h2 - h1) / time_gap
            for i in range(index2 - index1):
                """
                    The default virtual trajectory generation is by linear
                    motion (constant speed hypothesis), you could modify this 
                    part to implement your own. 
                """
                x = x1 + (i + 1) * dx
                y = y1 + (i + 1) * dy
                w = w1 + (i + 1) * dw
                h = h1 + (i + 1) * dh
                s = w * h
                r = w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                """
                    I still use predict-update loop here to refresh the parameters,
                    but this can be faster by directly modifying the internal parameters
                    as suggested in the paper. I keep this naive but slow way for 
                    easy read and understanding
                """
                self.update(new_box)
                if not i == (index2 - index1 - 1):
                    self.predict()

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # append the observation
        self.history_obs.append(z)

        if z is None:
            if self.observed:
                """
                    Got no observation so freeze the current parameters for future
                    potential online smoothing.
                """
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        # self.observed = True
        if not self.observed:
            """
                Get observation, use online smoothing to re-update parameters
            """
            self.unfreeze()
        self.observed = True

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_steadystate(self, u=0, B=None):
        """
        Predict state (prior) using the Kalman filter state propagation
        equations. Only x is updated, P is left unchanged. See
        update_steadstate() for a longer explanation of when to use this
        method.
        Parameters
        ----------
        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        """

        if B is None:
            B = self.B

        # x = Fx + Bu
        if B is not None:
            self.x = dot(self.F, self.x) + dot(B, u)
        else:
            self.x = dot(self.F, self.x)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update_steadystate(self, z):
        """
        Add a new measurement (z) to the Kalman filter without recomputing
        the Kalman gain K, the state covariance P, or the system
        uncertainty S.
        You can use this for LTI systems since the Kalman gain and covariance
        converge to a fixed value. Precompute these and assign them explicitly,
        or run the Kalman filter using the normal predict()/update(0 cycle
        until they converge.
        The main advantage of this call is speed. We do significantly less
        computation, notably avoiding a costly matrix inversion.
        Use in conjunction with predict_steadystate(), otherwise P will grow
        without bound.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        Examples
        --------
        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> # let filter converge on representative data, then save k and P
        >>> for i in range(100):
        >>>     cv.predict()
        >>>     cv.update([i, i, i])
        >>> saved_k = np.copy(cv.K)
        >>> saved_P = np.copy(cv.P)
        later on:
        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> cv.K = np.copy(saved_K)
        >>> cv.P = np.copy(saved_P)
        >>> for i in range(100):
        >>>     cv.predict_steadystate()
        >>>     cv.update_steadystate([i, i, i])
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(self.H, self.x)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update_correlated(self, z, R=None, H=None):
        """ Add a new measurement (z) to the Kalman filter assuming that
        process noise and measurement noise are correlated as defined in
        the `self.M` matrix.
        A partial derivation can be found in [1]
        If z is None, nothing is changed.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.
        References
        ----------
        .. [1] Bulut, Y. (2011). Applied Kalman filter theory (Doctoral dissertation, Northeastern University).
               http://people.duke.edu/~hpgavin/SystemID/References/Balut-KalmanFilter-PhD-NEU-2011.pdf
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if self.x.ndim == 1 and shape(z) == (1, 1):
            z = z[0]

        if shape(z) == ():  # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + dot(H, self.M) + dot(self.M.T, H.T) + R
        self.SI = self.inv(self.S)

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT + self.M, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(H, self.P) + self.M.T)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def batch_filter(self,
                     zs,
                     Fs=None,
                     Qs=None,
                     Hs=None,
                     Rs=None,
                     Bs=None,
                     us=None,
                     update_first=False,
                     saver=None):
        """ Batch processes a sequences of measurements.
        Parameters
        ----------
        zs : list-like
            list of measurements at each time step `self.dt`. Missing
            measurements must be represented by `None`.
        Fs : None, list-like, default=None
            optional value or list of values to use for the state transition
            matrix F.
            If Fs is None then self.F is used for all epochs.
            Otherwise it must contain a list-like list of F's, one for
            each epoch.  This allows you to have varying F per epoch.
        Qs : None, np.array or list-like, default=None
            optional value or list of values to use for the process error
            covariance Q.
            If Qs is None then self.Q is used for all epochs.
            Otherwise it must contain a list-like list of Q's, one for
            each epoch.  This allows you to have varying Q per epoch.
        Hs : None, np.array or list-like, default=None
            optional list of values to use for the measurement matrix H.
            If Hs is None then self.H is used for all epochs.
            If Hs contains a single matrix, then it is used as H for all
            epochs.
            Otherwise it must contain a list-like list of H's, one for
            each epoch.  This allows you to have varying H per epoch.
        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.
            If Rs is None then self.R is used for all epochs.
            Otherwise it must contain a list-like list of R's, one for
            each epoch.  This allows you to have varying R per epoch.
        Bs : None, np.array or list-like, default=None
            optional list of values to use for the control transition matrix B.
            If Bs is None then self.B is used for all epochs.
            Otherwise it must contain a list-like list of B's, one for
            each epoch.  This allows you to have varying B per epoch.
        us : None, np.array or list-like, default=None
            optional list of values to use for the control input vector;
            If us is None then None is used for all epochs (equivalent to 0,
            or no control input).
            Otherwise it must contain a list-like list of u's, one for
            each epoch.
       update_first : bool, optional, default=False
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.
        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch
        Returns
        -------
        means : np.array((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.
        covariance : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.
        means_predictions : np.array((n,dim_x,1))
            array of the state for each time step after the predictions. Each
            entry is an np.array. In other words `means[k,:]` is the state at
            step `k`.
        covariance_predictions : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the prediction.
            In other words `covariance[k,:,:]` is the covariance at step `k`.
        Examples
        --------
        .. code-block:: Python
            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts. This requires
            # that F be recomputed for each epoch. The output is then smoothed
            # with an RTS smoother.
            zs = [t + random.randn()*4 for t in range (40)]
            Fs = [np.array([[1., dt], [0, 1]] for dt in dts]
            (mu, cov, _, _) = kf.batch_filter(zs, Fs=Fs)
            (xs, Ps, Ks, Pps) = kf.rts_smoother(mu, cov, Fs=Fs)
        """

        #pylint: disable=too-many-statements
        n = np.size(zs, 0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((n, self.dim_x))
            means_p = zeros((n, self.dim_x))
        else:
            means = zeros((n, self.dim_x, 1))
            means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B,
                    u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                if saver is not None:
                    saver.save()
        else:
            for i, (z, F, Q, H, R, B,
                    u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                if saver is not None:
                    saver.save()

        return (means, covariances, means_p, covariances_p)

    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None, inv=np.linalg.inv):
        """
        Runs the Rauch-Tung-Striebel Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input
        would come from the output of `KalmanFilter.batch_filter()`.
        Parameters
        ----------
        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.
        Ps : numpy.array
            array of the covariances of the output of a kalman filter.
        Fs : list-like collection of numpy.array, optional
            State transition matrix of the Kalman filter at each time step.
            Optional, if not provided the filter's self.F will be used
        Qs : list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used
        inv : function, default numpy.linalg.inv
            If you prefer another inverse function, such as the Moore-Penrose
            pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv
        Returns
        -------
        x : numpy.ndarray
           smoothed means
        P : numpy.ndarray
           smoothed state covariances
        K : numpy.ndarray
            smoother gain at each step
        Pp : numpy.ndarray
           Predicted state covariances
        Examples
        --------
        .. code-block:: Python
            zs = [t + random.randn()*4 for t in range (40)]
            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K, Pp) = rts_smoother(mu, cov, kf.F, kf.Q)
        """

        if len(Xs) != len(Ps):
            raise ValueError('length of Xs and Ps must be the same')

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        K = zeros((n, dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            Pp[k] = dot(dot(Fs[k + 1], P[k]), Fs[k + 1].T) + Qs[k + 1]

            #pylint: disable=bad-whitespace
            K[k] = dot(dot(P[k], Fs[k + 1].T), inv(Pp[k]))
            x[k] += dot(K[k], x[k + 1] - dot(Fs[k + 1], x[k]))
            P[k] += dot(dot(K[k], P[k + 1] - Pp[k]), K[k].T)

        return (x, P, K, Pp)

    def get_prediction(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations and returns it without modifying the object.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        Returns
        -------
        (x, P) : tuple
            State vector and covariance array of the prediction.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            x = dot(F, self.x) + dot(B, u)
        else:
            x = dot(F, self.x)

        # P = FPF' + Q
        P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        return x, P

    def get_update(self, z=None):
        """
        Computes the new estimate based on measurement `z` and returns it
        without altering the state of the filter.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        Returns
        -------
        (x, P) : tuple
            State vector and covariance array of the update.
       """

        if z is None:
            return self.x, self.P
        z = reshape_z(z, self.dim_z, self.x.ndim)

        R = self.R
        H = self.H
        P = self.P
        x = self.x

        # error (residual) between measurement and prediction
        y = z - dot(H, x)

        # common subexpression for speed
        PHT = dot(P, H.T)

        # project system uncertainty into measurement space
        S = dot(H, PHT) + R

        # map system uncertainty into kalman gain
        K = dot(PHT, self.inv(S))

        # predict new x with residual scaled by the kalman gain
        x = x + dot(K, y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

        return x, P

    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        z = reshape_z(z, self.dim_z, self.x.ndim)
        return z - dot(self.H, self.x_prior)

    def measurement_of_state(self, x):
        """
        Helper function that converts a state into a measurement.
        Parameters
        ----------
        x : np.array
            kalman state vector
        Returns
        -------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        """

        return dot(self.H, x)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
        """
        return self._alpha_sq**.5

    def log_likelihood_of(self, z):
        """
        log likelihood of the measurement `z`. This should only be called
        after a call to update(). Calling after predict() will yield an
        incorrect result."""

        if z is None:
            return log(sys.float_info.min)
        return logpdf(z, dot(self.H, self.x), self.S)

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError('alpha must be a float greater than 1')

        self._alpha_sq = value**2

    def __repr__(self):
        return '\n'.join([
            'KalmanFilter object', pretty_str('dim_x', self.dim_x),
            pretty_str('dim_z', self.dim_z), pretty_str('dim_u', self.dim_u),
            pretty_str('x', self.x), pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('x_post', self.x_post),
            pretty_str('P_post', self.P_post), pretty_str('F', self.F),
            pretty_str('Q', self.Q), pretty_str('R', self.R), pretty_str(
                'H', self.H), pretty_str('K', self.K), pretty_str('y', self.y),
            pretty_str('S', self.S), pretty_str('SI', self.SI),
            pretty_str('M', self.M), pretty_str('B', self.B), pretty_str(
                'z', self.z), pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('likelihood', self.likelihood),
            pretty_str('mahalanobis', self.mahalanobis),
            pretty_str('alpha', self.alpha), pretty_str('inv', self.inv)
        ])

    def test_matrix_dimensions(self, z=None, H=None, R=None, F=None, Q=None):
        """
        Performs a series of asserts to check that the size of everything
        is what it should be. This can help you debug problems in your design.
        If you pass in H, R, F, Q those will be used instead of this object's
        value for those matrices.
        Testing `z` (the measurement) is problamatic. x is a vector, and can be
        implemented as either a 1D array or as a nx1 column vector. Thus Hx
        can be of different shapes. Then, if Hx is a single value, it can
        be either a 1D array or 2D vector. If either is true, z can reasonably
        be a scalar (either '3' or np.array('3') are scalars under this
        definition), a 1D, 1 element array, or a 2D, 1 element array. You are
        allowed to pass in any combination that works.
        """

        if H is None:
            H = self.H
        if R is None:
            R = self.R
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        x = self.x
        P = self.P

        assert x.ndim == 1 or x.ndim == 2, \
                "x must have one or two dimensions, but has {}".format(x.ndim)

        if x.ndim == 1:
            assert x.shape[0] == self.dim_x, \
                   "Shape of x must be ({},{}), but is {}".format(
                       self.dim_x, 1, x.shape)
        else:
            assert x.shape == (self.dim_x, 1), \
                   "Shape of x must be ({},{}), but is {}".format(
                       self.dim_x, 1, x.shape)

        assert P.shape == (self.dim_x, self.dim_x), \
               "Shape of P must be ({},{}), but is {}".format(
                   self.dim_x, self.dim_x, P.shape)

        assert Q.shape == (self.dim_x, self.dim_x), \
               "Shape of Q must be ({},{}), but is {}".format(
                   self.dim_x, self.dim_x, P.shape)

        assert F.shape == (self.dim_x, self.dim_x), \
               "Shape of F must be ({},{}), but is {}".format(
                   self.dim_x, self.dim_x, F.shape)

        assert np.ndim(H) == 2, \
               "Shape of H must be (dim_z, {}), but is {}".format(
                   P.shape[0], shape(H))

        assert H.shape[1] == P.shape[0], \
               "Shape of H must be (dim_z, {}), but is {}".format(
                   P.shape[0], H.shape)

        # shape of R must be the same as HPH'
        hph_shape = (H.shape[0], H.shape[0])
        r_shape = shape(R)

        if H.shape[0] == 1:
            # r can be scalar, 1D, or 2D in this case
            assert r_shape in [(), (1,), (1, 1)], \
            "R must be scalar or one element array, but is shaped {}".format(
                r_shape)
        else:
            assert r_shape == hph_shape, \
            "shape of R should be {} but it is {}".format(hph_shape, r_shape)

        if z is not None:
            z_shape = shape(z)
        else:
            z_shape = (self.dim_z, 1)

        # H@x must have shape of z
        Hx = dot(H, x)

        if z_shape == ():  # scalar or np.array(scalar)
            assert Hx.ndim == 1 or shape(Hx) == (1, 1), \
            "shape of z should be {}, not {} for the given H".format(
                shape(Hx), z_shape)

        elif shape(Hx) == (1, ):
            assert z_shape[
                0] == 1, 'Shape of z must be {} for the given H'.format(
                    shape(Hx))

        else:
            assert (z_shape == shape(Hx) or
                    (len(z_shape) == 1 and shape(Hx) == (z_shape[0], 1))), \
                    "shape of z should be {}, not {} for the given H".format(
                        shape(Hx), z_shape)

        if np.ndim(Hx) > 1 and shape(Hx) != (1, 1):
            assert shape(Hx) == z_shape, \
               'shape of z should be {} for the given H, but it is {}'.format(
                   shape(Hx), z_shape)


def update(x, P, z, R, H=None, return_all=False):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.
    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.
    update(1, 2, 1, 1, 1)  # univariate
    update(x, P, 1
    Parameters
    ----------
    x : numpy.array(dim_x, 1), or float
        State estimate vector
    P : numpy.array(dim_x, dim_x), or float
        Covariance matrix
    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.
    R : numpy.array(dim_z, dim_z), or float
        Measurement noise matrix
    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.
    return_all : bool, default False
        If true, y, K, S, and log_likelihood are returned, otherwise
        only x and P are returned.
    Returns
    -------
    x : numpy.array
        Posterior state estimate vector
    P : numpy.array
        Posterior covariance matrix
    y : numpy.array or scalar
        Residua. Difference between measurement and state in measurement space
    K : numpy.array
        Kalman gain
    S : numpy.array
        System uncertainty in measurement space
    log_likelihood : float
        log likelihood of the measurement
    """

    #pylint: disable=bare-except

    if z is None:
        if return_all:
            return x, P, None, None, None, None
        return x, P

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    Hx = np.atleast_1d(dot(H, x))
    z = reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # project system uncertainty into measurement space
    S = dot(dot(H, P), H.T) + R

    # map system uncertainty into kalman gain
    try:
        K = dot(dot(P, H.T), linalg.inv(S))
    except:
        # can't invert a 1D array, annoyingly
        K = dot(dot(P, H.T), 1. / S)

    # predict new x with residual scaled by the kalman gain
    x = x + dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, H)

    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array([1 - KH])
    P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

    if return_all:
        # compute log likelihood
        log_likelihood = logpdf(z, dot(H, x), S)
        return x, P, y, K, S, log_likelihood
    return x, P


def update_steadystate(x, z, K, H=None):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.
    Parameters
    ----------
    x : numpy.array(dim_x, 1), or float
        State estimate vector
    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.
    K : numpy.array, or float
        Kalman gain matrix
    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.
    Returns
    -------
    x : numpy.array
        Posterior state estimate vector
    Examples
    --------
    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.
    >>> update_steadystate(1, 2, 1)  # univariate
    >>> update_steadystate(x, P, z, H)
    """

    if z is None:
        return x

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    Hx = np.atleast_1d(dot(H, x))
    z = reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # estimate new x with residual scaled by the kalman gain
    return x + dot(K, y)


def predict(x, P, F=1, Q=0, u=0, B=1, alpha=1.):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations.
    Parameters
    ----------
    x : numpy.array
        State estimate vector
    P : numpy.array
        Covariance matrix
    F : numpy.array()
        State Transition matrix
    Q : numpy.array, Optional
        Process noise matrix
    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.
    B : numpy.array, optional, default 0.
        Control transition matrix.
    alpha : float, Optional, default=1.0
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon
    Returns
    -------
    x : numpy.array
        Prior state estimate vector
    P : numpy.array
        Prior covariance matrix
    """

    if np.isscalar(F):
        F = np.array(F)
    x = dot(F, x) + dot(B, u)
    P = (alpha * alpha) * dot(dot(F, P), F.T) + Q

    return x, P


def predict_steadystate(x, F=1, u=0, B=1):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations. This steady state form only computes x, assuming that the
    covariance is constant.
    Parameters
    ----------
    x : numpy.array
        State estimate vector
    P : numpy.array
        Covariance matrix
    F : numpy.array()
        State Transition matrix
    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.
    B : numpy.array, optional, default 0.
        Control transition matrix.
    Returns
    -------
    x : numpy.array
        Prior state estimate vector
    """

    if np.isscalar(F):
        F = np.array(F)
    x = dot(F, x) + dot(B, u)

    return x


def batch_filter(x,
                 P,
                 zs,
                 Fs,
                 Qs,
                 Hs,
                 Rs,
                 Bs=None,
                 us=None,
                 update_first=False,
                 saver=None):
    """
    Batch processes a sequences of measurements.
    Parameters
    ----------
    zs : list-like
        list of measurements at each time step. Missing measurements must be
        represented by None.
    Fs : list-like
        list of values to use for the state transition matrix matrix.
    Qs : list-like
        list of values to use for the process error
        covariance.
    Hs : list-like
        list of values to use for the measurement matrix.
    Rs : list-like
        list of values to use for the measurement error
        covariance.
    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.
    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.
    update_first : bool, optional
        controls whether the order of operations is update followed by
        predict, or predict followed by update. Default is predict->update.
        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch
    Returns
    -------
    means : np.array((n,dim_x,1))
        array of the state for each time step after the update. Each entry
        is an np.array. In other words `means[k,:]` is the state at step
        `k`.
    covariance : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the update.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    means_predictions : np.array((n,dim_x,1))
        array of the state for each time step after the predictions. Each
        entry is an np.array. In other words `means[k,:]` is the state at
        step `k`.
    covariance_predictions : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the prediction.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    Examples
    --------
    .. code-block:: Python
        zs = [t + random.randn()*4 for t in range (40)]
        Fs = [kf.F for t in range (40)]
        Hs = [kf.H for t in range (40)]
        (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                          Bs=None, us=None, update_first=False)
        (xs, Ps, Ks, Pps) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)
    """

    n = np.size(zs, 0)
    dim_x = x.shape[0]

    # mean estimates from Kalman Filter
    if x.ndim == 1:
        means = zeros((n, dim_x))
        means_p = zeros((n, dim_x))
    else:
        means = zeros((n, dim_x, 1))
        means_p = zeros((n, dim_x, 1))

    # state covariances from Kalman Filter
    covariances = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))

    if us is None:
        us = [0.] * n
        Bs = [0.] * n

    if update_first:
        for i, (z, F, Q, H, R, B,
                u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            if saver is not None:
                saver.save()
    else:
        for i, (z, F, Q, H, R, B,
                u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            if saver is not None:
                saver.save()

    return (means, covariances, means_p, covariances_p)


def rts_smoother(Xs, Ps, Fs, Qs):
    """
    Runs the Rauch-Tung-Striebel Kalman smoother on a set of
    means and covariances computed by a Kalman filter. The usual input
    would come from the output of `KalmanFilter.batch_filter()`.
    Parameters
    ----------
    Xs : numpy.array
       array of the means (state variable x) of the output of a Kalman
       filter.
    Ps : numpy.array
        array of the covariances of the output of a kalman filter.
    Fs : list-like collection of numpy.array
        State transition matrix of the Kalman filter at each time step.
    Qs : list-like collection of numpy.array, optional
        Process noise of the Kalman filter at each time step.
    Returns
    -------
    x : numpy.ndarray
       smoothed means
    P : numpy.ndarray
       smoothed state covariances
    K : numpy.ndarray
        smoother gain at each step
    pP : numpy.ndarray
       predicted state covariances
    Examples
    --------
    .. code-block:: Python
        zs = [t + random.randn()*4 for t in range (40)]
        (mu, cov, _, _) = kalman.batch_filter(zs)
        (x, P, K, pP) = rts_smoother(mu, cov, kf.F, kf.Q)
    """

    if len(Xs) != len(Ps):
        raise ValueError('length of Xs and Ps must be the same')

    n = Xs.shape[0]
    dim_x = Xs.shape[1]

    # smoother gain
    K = zeros((n, dim_x, dim_x))
    x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()

    for k in range(n - 2, -1, -1):
        pP[k] = dot(dot(Fs[k], P[k]), Fs[k].T) + Qs[k]

        #pylint: disable=bad-whitespace
        K[k] = dot(dot(P[k], Fs[k].T), linalg.inv(pP[k]))
        x[k] += dot(K[k], x[k + 1] - dot(Fs[k], x[k]))
        P[k] += dot(dot(K[k], P[k + 1] - pP[k]), K[k].T)

    return (x, P, K, pP)
