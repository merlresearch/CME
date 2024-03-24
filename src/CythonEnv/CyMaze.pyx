# distutils: language = c++

# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os

import numpy as np

cimport numpy as np
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from CyInterface cimport CyInterface
from libcpp cimport bool
from libcpp.vector cimport vector


# DON'T use this class directly. operate throguh CyMazeWrapper
# state means internal states of simulator
cdef class _CyMaze:
    cdef CyInterface* c_env
    cdef double *c_state
    cdef public int dim_state
    cdef int inner_dim_state

    def __cinit__(self, path_config, *argv, **argdict):
        if not os.path.exists(path_config):
            raise ValueError("invalid path : ", path_config)

        self.c_env = new CyInterface(bytes(path_config, encoding="ascii"))
        self.inner_dim_state = self.c_env.dimState()
        self.dim_state = self.inner_dim_state
        self.c_state = <double*>PyMem_Malloc(self.inner_dim_state * sizeof(double))

    def __dealloc__(self):
        PyMem_Free(self.c_state)
        del self.c_env

    def __init__(self, path_config):
        pass

    def reset(self):
        with nogil:
            arr = self.c_env.reset()
        res = np.array(arr)

        return res

    def get_ring_idx(self):
        """get the index of ring

        CAUTION : the returned value of get_ring_idx is not equal to set_ring_idx.
        get_ring_idx returns the ring idx calculated from mujoco internal state.
        :return:
        """
        cdef int ringIdx
        with nogil:
            ringIdx = self.c_env.getRingIdx()

        return ringIdx

    def get_theta_vec(self):
        """get the theta representation of the current state.
        """
        cdef vector[double] vec;

        with nogil:
            vec = self.c_env.getThetaVector()

        array = np.array(vec)
        return array

    def get_xyz_vec(self):
        """get the theta representation of the current state.
        """
        cdef vector[double] vec;

        with nogil:
            vec = self.c_env.getXYZVector()

        array = np.array(vec)
        return array


    def set_state_vector(self, np.ndarray[np.float64_t, ndim=1] state):
        """set state

        the state is composed of 4 states

        - board_x
        - board_y
        - ball_theta
        - ball_theta_vel

        If you want to change the ring where the ball belongs to, call set_ring.

        :param np.ndarray state:
        :return:
        """
        assert len(state) == self.dim_state

        cdef vector[double] set_state = state

        with nogil:
            self.c_env.setState(set_state)

    def set_state_with_ring_and_theta(self, int ring_idx, double theta):
        with nogil:
            self.c_env.setRingState(ring_idx, theta)

    def set_state_vector_with_ring(self, ring_idx, np.ndarray[np.float64_t, ndim=1] state):
        """set state

        the state is composed of 4 states

        - board_x
        - board_y
        - ball_theta
        - ball_theta_vel

        If you want to change the ring where the ball belongs to, call set_ring.

        :param np.ndarray state:
        :return:
        """
        assert len(state) == self.dim_state

        self.set_state_vector(state)
        self.set_ring_idx(ring_idx)

    def set_ring_idx(self, ring_idx):
        self.c_env.setRingIdx(ring_idx)

    def get_state_vector(self):
        """get internal state

        内部状態を返す. 子クラスでoverrideしないこと
        :return:
        """
        with nogil:
            self.c_env.getState(self.c_state)

        cdef double [:] arr = <double[:self.inner_dim_state]> self.c_state
        res = np.array(arr, copy=True)
        return res

    def cost_state(self):
        """calculates the cost of current state.

        :rtype: float
        :return:
        """
        cdef double cost;

        with nogil:
            cost = self.c_env.costState()

        return cost

    def step(self, np.ndarray[np.float64_t, ndim=1] action):
        cdef double reward
        cdef int done

        with nogil:
            self.c_env.step(action[0], action[1], self.c_state, &reward, &done)

        cdef double [:] arr = <double[:self.inner_dim_state]> self.c_state
        # arr = np.asarray(arr)
        arr = np.array(arr, copy=True)
        return np.asarray(arr), reward, done, None

    def set_fric(self, fric_slide, fric_spin, fric_roll):
        # setBallFirc represents all fric now
        self.c_env.setBallFric(fric_slide, fric_spin, fric_roll)

    def set_fricloss(self, fric_loss):
        # setBallFirc represents all fric now
        self.c_env.setFricloss(fric_loss)

    def Lx(self):
        """calculates the derivative of cost with respect to the current state.

        :return:
          Lx shape (dim_state)
        """
        cdef vector[double] lx;

        with nogil:
            lx = self.c_env.Lx()

        raise NotImplementedError("use Lx_theta")
        return lx

    def Lxx(self):
        """calculates the derivative of cost with respect to the current state.

        :param state:
        :rtype: np.ndarray
        :return:
          Lxx shape (dim_state, dim_state)
        """
        cdef vector[double] lxx
        with nogil:
            lxx = self.c_env.Lxx()

        np_lxx = np.array(lxx)
        raise NotImplementedError("use Lxx_theta")
        return np_lxx.reshape(self.inner_dim_state, self.inner_dim_state)

    def FxFu(self, np.ndarray[np.float64_t, ndim=1] action):
        """calculates the derivative of dynamics with respect to the current state.

        :return:
          (fx, fu)
          fx shape (dim_state, dim_state)
          fu shape (dim_state, 2)
        """
        cdef double* fx = <double*>PyMem_Malloc(self.inner_dim_state * self.inner_dim_state * sizeof(double))
        cdef double* fu = <double*>PyMem_Malloc(self.inner_dim_state * 2 * sizeof(double))

        with nogil:
            self.c_env.FxFu(action[0], action[1], fx, fu)

        cdef double [:] fx_view = <double[:self.inner_dim_state*self.inner_dim_state]> fx
        cdef double [:] fu_view = <double[:self.inner_dim_state*2]> fu

        np_fx = np.array(fx_view, dtype=np.float64, copy=True)
        np_fu = np.array(fu_view, dtype=np.float64, copy=True)

        PyMem_Free(fx)
        PyMem_Free(fu)

        raise NotImplementedError("use FxFu_theta")
        #  return np_fx, np_fu
        return np_fx.reshape((self.inner_dim_state, self.inner_dim_state)), np_fu.reshape((self.inner_dim_state, 2))

    def FxFu_theta(self, np.ndarray[np.float64_t, ndim=1] action):
        """calculates the derivative of dynamics with respect to the current state.

        :return:
          (fx, fu)
          fx shape (dim_state, dim_state)
          fu shape (dim_state, 2)
        """
        cdef int dim_theta = 4

        cdef double* fx = <double*>PyMem_Malloc(dim_theta * dim_theta * sizeof(double))
        cdef double* fu = <double*>PyMem_Malloc(dim_theta * 2 * sizeof(double))

        with nogil:
            self.c_env.FxFuTheta(action[0], action[1], fx, fu)

        cdef double [:] fx_view = <double[:dim_theta*dim_theta]> fx
        cdef double [:] fu_view = <double[:dim_theta*2]> fu

        np_fx = np.array(fx_view, dtype=np.float64, copy=True)
        np_fu = np.array(fu_view, dtype=np.float64, copy=True)

        PyMem_Free(fx)
        PyMem_Free(fu)

        #  return np_fx, np_fu
        return np_fx.reshape((dim_theta, dim_theta)), np_fu.reshape((dim_theta, 2))

    def FxFu_theta_on_theta(self, np.ndarray[np.float64_t, ndim=1] action):
        """calculates the derivative of dynamics with respect to the current state.

        :return:
          (fx, fu)
          fx shape (dim_state, dim_state)
          fu shape (dim_state, 2)
        """
        cdef int dim_theta = 4

        cdef double* fx = <double*>PyMem_Malloc(dim_theta * dim_theta * sizeof(double))
        cdef double* fu = <double*>PyMem_Malloc(dim_theta * 2 * sizeof(double))

        with nogil:
            self.c_env.FxFuThetaOnTheta(action[0], action[1], fx, fu)

        cdef double [:] fx_view = <double[:dim_theta*dim_theta]> fx
        cdef double [:] fu_view = <double[:dim_theta*2]> fu

        np_fx = np.array(fx_view, dtype=np.float64, copy=True)
        np_fu = np.array(fu_view, dtype=np.float64, copy=True)

        PyMem_Free(fx)
        PyMem_Free(fu)

        #  return np_fx, np_fu
        return np_fx.reshape((dim_theta, dim_theta)), np_fu.reshape((dim_theta, 2))


    def Lx_theta(self):
        """calculates the derivative of cost with respect to the current state.

        :return:
          Lx shape (dim_state)
        """
        cdef vector[double] lx;

        with nogil:
            lx = self.c_env.LxTheta()

        return lx


    def Lxx_theta(self):
        """calculates the derivative of cost with respect to the current state.

        :param state:
        :rtype: np.ndarray
        :return:
          Lxx shape (dim_state, dim_state)
        """
        cdef vector[double] lxx

        with nogil:
            lxx = self.c_env.LxxTheta()

        np_lxx = np.array(lxx)
        return np_lxx.reshape(4,4)

    def Lx_theta_on_theta(self):
        """calculates the derivative of cost with respect to the current state.

        In this function, gradients are calculated by ThetaDyanamics.cpp.
        So in both case, ThetaDynamics and XYZDyanamics, calculated gradients are same.

        :return:
          Lx shape (dim_state)
        """
        cdef vector[double] lx;

        with nogil:
            lx = self.c_env.LxThetaOnTheta()

        return lx


    def Lxx_theta_on_theta(self):
        """calculates the derivative of cost with respect to the current state.

        In this function, gradients are calculated by ThetaDyanamics.cpp.
        So in both case, ThetaDynamics and XYZDyanamics, calculated gradients are same.

        :param state:
        :rtype: np.ndarray
        :return:
          Lxx shape (dim_state, dim_state)
        """
        cdef vector[double] lxx

        with nogil:
            lxx = self.c_env.LxxThetaOnTheta()

        np_lxx = np.array(lxx)
        return np_lxx.reshape(4,4)
