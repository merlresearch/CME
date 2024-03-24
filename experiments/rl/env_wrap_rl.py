# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import cv2
import numpy as np
from gym import spaces

from CythonEnv.CyMazeWrapper import CyThetaObsMaze
from CythonEnv.CyRender import CyRender


class CyThetaObsMazeRLWrapper:
    def __init__(self, is_real=True, is_render=False, input_state=False, reset_to_random_ring=False):
        dir_root = os.environ["MAZE_SIMULATOR_ROOT"]
        if is_real:
            config_path = os.path.join(dir_root, "experimental_settings/sim2sim/config_xyz_less_friction.txt")
        else:
            config_path = os.path.join(dir_root, "experimental_settings/sim2sim/config_theta_less_friction.txt")
        self._env = CyThetaObsMaze(config_path, lam_control=10.0, lam_vel=0.01)
        if is_render:
            self._renderer = CyRender(config_path)

        self._reset_to_random_ring = reset_to_random_ring
        self._is_real = is_real
        self._input_state = input_state
        if input_state:
            assert is_real is True, "is_real should be True if you specify input_state is True"
            dim_input = 8
        else:
            dim_input = 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim_input,), dtype=np.float64)

    def _get_obs(self):
        if self._input_state:
            return self._env.get_state_vector()
        else:
            theta_vec = self._env.get_theta_vec()
            ring_idx = self._env.get_ring_idx()
            return np.insert(theta_vec, theta_vec.shape[0], ring_idx)

    def reset(self):
        self._env.reset()
        if self._reset_to_random_ring:
            ring_idx = np.random.randint(1, 5)
            initial_state = np.random.uniform(-np.pi, np.pi)
            self._env.set_state_with_ring_and_theta(ring_idx, initial_state)
        return self._get_obs()

    def render(self):
        state = self._env.get_state_vector()
        obs = self._renderer.render(state)
        cv2.imshow("real_step", obs)
        cv2.waitKey(1)

    def step(self, action):
        _, _, done, _ = self._env.step(action.astype(np.float64))
        cost = self._env.cost_state()
        cost += self._env.cost_control(action)
        rew = cost * (-1.0)
        return self._get_obs(), rew, done, _

    def __getattr__(self, attr):
        # Enable to call functions defined in `self._wrapped_env`
        return self._env.__getattribute__(attr)
