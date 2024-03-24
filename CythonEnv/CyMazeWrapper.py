# -*- coding: utf-8 -*-
# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import numpy as np
from gym import spaces

from CythonEnv.CyMaze import _CyMaze


class CyThetaObsMaze(_CyMaze):
    """wrapper of _CyMaze to handle python attributes

    observationが内部状態に関わらず4次元のtheta表現となるMaze.
    内部状態にXYZもThetaも取れる.

    """

    def __init__(
        self, path_config, lam_control=0.0, lam_vel=0.0, lam_tilt_angle=0.0, clip_grad=20, force_theta_grad=False
    ):
        """

        cost_control = lam_cost_control * ||u||^2

        :param path_config:
        :param lam_control:
        :param lam_vel:
        :param lam_tilt_angle:
        :param clip_grad: 勾配のclipping上限
        :param force_theta_grad: 勾配計算時にXYZ表現でもThetaDynamicsの計算方法を用いるかどうか
        """
        super().__init__(path_config)
        limit = math.radians(0.5)
        self.action_space = spaces.Box(low=-limit, high=limit, shape=(2,))
        self.dim_obs = 4
        self.clip_grad = clip_grad
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim_obs,), dtype=np.float64)
        self._lam_control = lam_control
        self._lam_vel = lam_vel
        self._lam_tilt_angle = lam_tilt_angle

        self._force_theta_grad = force_theta_grad

    def reset(self):
        state = super().reset()
        self.set_state_vector(state)
        theta_vec = super().get_theta_vec()
        return theta_vec

    def step(self, action):
        """computes s_t+1 from (s_t, u_t)

        :param np.ndarray action:
        :return:
         (state, reward, done, dict_cost)

         dict_cost contains
           "cost_state"   : cost_state(s_t)
           "cost_control" : cost_control(u_t)
           "cost_total" CyMaze : cost_state + cost_control
        """
        # reward containCyMaze angular based cost difference only.
        state, reward, dCyMazene, _ = super().step(action)

        # cut the elemenCyMaze which cannot be expressed by the current representation.
        self.set_state_vector(state)
        theta_vec = super().get_theta_vec()

        return theta_vec, reward, dCyMazene, _

    def cost_control(self, control):
        n = np.linalg.norm(control)
        cost = n * n * self._lam_control
        return cost

    def Lu(self, control):
        return 2 * self._lam_control * control

    def Luu(self, action):
        return 2 * self._lam_control * np.eye(2)

    def get_obs(self):
        return super().get_theta_vec()

    def cost_state(self):
        """computes cost of the current state.

        cost_state = angular based cost (in super().cost_state) + angular velocity based cost (in Python)

        If you edit this function, you must edit Lx and Lxx.

        :return:
        """
        vec_state = self.get_theta_vec()

        if len(vec_state) != 4:
            raise NotImplementedError("cost_state is implemented only for Theta representation.")

        board_cost = self._lam_tilt_angle * np.sum(vec_state[:2] ** 2)

        vel = vec_state[3]
        vel_cost = self._lam_vel * vel * vel
        pos_cost = super().cost_state()

        return board_cost + pos_cost + vel_cost

    def Lx(self):
        if self._force_theta_grad:
            lx = super().Lx_theta_on_theta()
        else:
            lx = super().Lx_theta()

        vec_state = self.get_theta_vec()
        if len(vec_state) != 4:
            raise NotImplementedError("cost_state is implemented only for Theta representation.")

        board_cost = 2 * self._lam_tilt_angle * vec_state[:2]
        board_cost = np.concatenate([board_cost, [0, 0]], axis=0)
        vel = vec_state[3]

        lx += 2 * self._lam_vel * np.array([0, 0, 0, vel], dtype=np.float64)
        lx += board_cost

        return lx

    def Lxx(self):
        if self._force_theta_grad:
            lxx = super().Lxx_theta_on_theta()
        else:
            lxx = super().Lxx_theta()

        vel_lxx = np.zeros((self.dim_obs, self.dim_obs), dtype=np.float64)
        vel_lxx[3, 3] = 2 * self._lam_vel

        board_lxx = np.zeros((self.dim_obs, self.dim_obs), dtype=np.float64)
        board_lxx[0, 0] = 1
        board_lxx[1, 1] = 1

        lxx += vel_lxx + self._lam_tilt_angle * board_lxx

        return lxx

    def FxFu(self, action):
        if self._force_theta_grad:
            fx, fu = super().FxFu_theta_on_theta(action)
        else:
            fx, fu = super().FxFu_theta(action)

        fx = np.clip(fx, -self.clip_grad, self.clip_grad)
        fu = np.clip(fu, -self.clip_grad, self.clip_grad)

        return fx, fu

    def set_state_vector(self, state):
        assert len(state) == self.dim_state
        super().set_state_vector(state)

    def total_cost(self, s0, U):
        """a reference implementation to calculate the total cost

        :return:
        """
        T = len(U)
        total_cost = 0
        self.set_state_vector(s0)

        for i in range(T):
            u = U[i]

            cost_state = self.cost_state()
            cost_control = self.cost_control(u)

            total_cost += cost_state + cost_control

            obs, reward, done, _ = self.step(u)

        total_cost += self.cost_state()

        return total_cost
