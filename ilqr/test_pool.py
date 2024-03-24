# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import time
import unittest

import numpy as np

from CythonEnv.CyMazeWrapper import CyMaze
from ilqr.ilqr import MjIterativeLQR
from ilqr.pools import EnvProcessPool


class TestCircularMaze(unittest.TestCase):
    def _get_config_path(self, config_name=None):
        os.environ["MAZE_SIMULATOR_ROOT"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../install")
        config_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")

        if config_name is None:
            config_name = "config_nowarm.txt"

        config_path = os.path.join(config_root, config_name)
        return config_path

    def test_forward_multiprocess(self):
        def make_env():
            lam_control = 0.1
            lam_vel = 0.2
            env = CyMaze(self._get_config_path(), lam_control=lam_control, lam_vel=lam_vel)
            return env

        ilqr = MjIterativeLQR(make_env, timesteps=15, force_single_thread=True, num_alpha=4)
        ilqr.initialize()
        ilqr.optimize()

        X, U = ilqr.current_obs_sequence, ilqr.current_control_sequence
        list_lx, list_lxx, list_lu, list_luu, list_lux, list_fx, list_fu = ilqr.derivative_sequential(X, U)
        k, K = ilqr.backward(X, U, list_lx, list_lxx, list_lu, list_luu, list_lux, list_fx, list_fu)

        newX, newU, min_cost = ilqr.sequential_forward(X, U, k, K, alpha_list=ilqr._alpha_List)

        ilqr = MjIterativeLQR(make_env, timesteps=15, force_single_thread=False, num_alpha=4)
        multiX, multiU, multi_cost = ilqr._env_pool_forward.forward(X, U, k, K, alpha_list=ilqr._alpha_List)

        for s, th in zip(newX, multiX):
            np.testing.assert_equal(s, th)
        for s, th in zip(newU, multiU):
            np.testing.assert_equal(s, th)

        np.testing.assert_equal(min_cost, multi_cost)
        env = make_env()
        total_cost = env.total_cost(newX[0], newU)
        np.testing.assert_equal(min_cost, total_cost)

    def test_gradient_multiprocess(self):
        def make_env():
            lam_control = 0.1
            lam_vel = 0.2
            env = CyMaze(self._get_config_path(), lam_control=lam_control, lam_vel=lam_vel)
            return env

        ilqr = MjIterativeLQR(make_env, timesteps=15, force_single_thread=True, num_alpha=1)
        ilqr.initialize()
        ilqr.optimize()
        X, U = ilqr.current_obs_sequence, ilqr.current_control_sequence

        # re-createing MjIterativeLQR is very important.
        # solver internal state affects greatly the final result
        ilqr = MjIterativeLQR(make_env, timesteps=15, force_single_thread=False, num_alpha=4)

        dim_state = X[0].shape[0]
        dim_action = U[0].shape[0]

        # MEMO: the number of threads affect sequential model performance
        pool = EnvProcessPool(make_env, dim_state, dim_action, len(U), num_processes=4, debug=True)
        ilqr._work_env.set_state_vector(np.zeros((4,), dtype=np.float64))

        # time.sleep(1)
        t1 = time.time() * 1000
        seq_lx, seq_lxx, seq_lu, seq_luu, seq_lux, seq_fx, seq_fu = ilqr.derivative_sequential(X, U)
        t2 = time.time() * 1000
        th_lx, th_lxx, th_lu, th_luu, th_lux, th_fx, th_fu = pool.derivatives(X, U)
        t3 = time.time() * 1000
        print("seq {} ms process {} ms".format(t2 - t1, t3 - t2))

        for s, th in zip(seq_lx, th_lx):
            np.testing.assert_equal(s, th)
        for s, th in zip(seq_lxx, th_lxx):
            np.testing.assert_equal(s, th)
        for s, th in zip(seq_lu, th_lu):
            np.testing.assert_equal(s, th)
        for s, th in zip(seq_luu, th_luu):
            np.testing.assert_equal(s, th)
        for s, th in zip(seq_fx, th_fx):
            np.testing.assert_almost_equal(s, th)
        for s, th in zip(seq_fu, th_fu):
            np.testing.assert_almost_equal(s, th)


if __name__ == "__main__":
    unittest.main()
