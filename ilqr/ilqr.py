# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import time

import numpy as np

import ilqr.util as util
from ilqr.logger import initialize_logger
from ilqr.pools import EnvForwardPool, EnvProcessPool
from ilqr.util import NP_DTYPE


class MjIterativeLQR:
    def __init__(
        self,
        make_env,
        timesteps,
        min_eta=1e-8,
        max_eta=1e16,
        tol_fun=1e-7,
        eta_factor=1.5,
        num_alpha=4,
        force_single_thread=False,
        debug=False,
        logging_level="DEBUG",
    ):
        """

        Parameters
        ----------
        make_env : function
            Create an environmental object by calling `make_env()`
        timesteps : int
            Number of steps in an episode
        min_eta : float
            Minimum coefficient for identity matrix regularization
        max_eta : float
            Maximum coefficient for identity matrix regularization
        tol_fun : float
        eta_factor : float
        num_alpha : int
        force_single_thread : bool
            True if you want to force to run on single thread.
            Note this is False by default to accelerate computation
        debug : bool
        logging_level : str
            Logging level which should be specified from ["DEBUG", "INFO", "WARNING", "ERROR"]
        """
        self._logger = initialize_logger(logging_level=logging.getLevelName(logging_level), save_log=False)

        self._work_env = make_env()
        self.dim_state = self._work_env.dim_state
        self._timesteps = timesteps

        self._tolFun = tol_fun
        self._min_eta = min_eta
        self._max_eta = max_eta
        self._etaFactor = eta_factor

        self._alpha_List = np.power(10, np.linspace(0, -3, num_alpha), dtype=NP_DTYPE)

        if num_alpha == 1 or force_single_thread:
            self._multi_mode = False
        else:
            self._multi_mode = True
            num_threads = num_alpha

            dim_obs = self._work_env.dim_obs
            dim_state = self._work_env.dim_state
            dim_control = self._work_env.action_space.shape[0]
            self._env_pool_derivative = EnvProcessPool(
                make_env, dim_obs, dim_state, dim_control, timesteps, num_processes=num_threads
            )
            self._env_pool_forward = EnvForwardPool(
                make_env, dim_obs, dim_state, dim_control, timesteps, self._alpha_List, num_processes=num_threads
            )

        self._measure_time = debug

        self._X = None
        # Manage internal states. The length will be same with X, but its dimension might be bigger than X.
        self._states = None
        self._U = None

    @property
    def current_obs_sequence(self):
        return self._X

    @property
    def current_state_sequence(self):
        return self._states

    @property
    def current_control_sequence(self):
        return self._U

    @property
    def current_cost(self):
        return self._cost

    @property
    def timesteps(self):
        return self._timesteps

    def initialize(self, initial_state=None, policy="zeros"):
        if initial_state is not None:
            assert (
                self._work_env.dim_state == initial_state.shape[0]
            ), "Unmatched dim_state: {}, initial_state: {}".format(self._work_env.dim_state, initial_state.shape[0])
        self._X, self._states, self._U, self._cost = self.rollout(initial_state, policy)

    def rollout(self, initial_state=None, policy="zeros"):
        """make a trajectory with controls

        :param np.ndarray initial_state:
        :param policy: "zeros" or "random"
        :param max_steps: if None, rollout while done is False.
        :return:
         (X, U, cost)
        """
        env = self._work_env

        if initial_state is None:
            x0 = env.reset()
        else:
            assert len(initial_state) == self.dim_state
            env.set_state_vector(initial_state)
            x0 = env.get_obs()

        x0_internal = env.get_state_vector()

        dim_action = env.action_space.shape[0]

        X = []
        states = []
        U = []
        cost = 0

        X.append(x0)
        states.append(x0_internal)

        for i in range(self._timesteps):
            if policy == "zeros":
                u = np.zeros(dim_action, dtype=NP_DTYPE)
            elif policy == "random":
                u = env.action_space.sample().astype(np.float64)
            else:
                raise ValueError("unknown policy : {}".format(policy))

            cost_state = env.cost_state()
            cost_control = env.cost_control(u)
            cur_cost = cost_state + cost_control
            cost += cur_cost

            obs, reward, done, _ = env.step(u)
            internal_state = env.get_state_vector()

            U.append(u)
            X.append(obs)
            states.append(internal_state)

        cost += env.cost_state()

        return X, states, U, cost

    def update_s1(self, current_state):
        """set the first state of the sequence to the given state

        :param np.ndarray current_state:
        :return:
        """
        assert len(current_state) == self.dim_state, "Dimension mismatch occurred. input: {}, dim_state: {}".format(
            len(current_state), self.dim_state
        )

        t0 = time.time() * 1000
        self._work_env.set_state_vector(current_state)
        x = self._work_env.get_obs()
        new_X = [x]
        new_states = [current_state]

        zero_action = np.zeros(self._work_env.action_space.shape, dtype=np.float64)
        self._U = np.concatenate((self._U[1:], [zero_action]), axis=0)

        for u in self._U:
            x, _, _, _ = self._work_env.step(u)
            state = self._work_env.get_state_vector()
            new_X.append(x)
            new_states.append(state)

        self._X = new_X
        self._states = new_states

        t1 = time.time() * 1000

        if self._measure_time:
            print("rollout time : {:.1f} ms ".format(t1 - t0), end="")

        self._X, self._states, self._U, self._cost = self._optimize(self._X, self._states, self._U)
        return self._U[0]

    def optimize(self, optimize_iterations=1):
        """update internal sequence

        :return:
        """
        # self._X, self._U, self._cost = self._optimize(self._X, self._U, self._cost)
        self._X, self._states, self._U, self._cost = self._optimize(
            self._X, self._states, self._U, None, optimize_iterations=optimize_iterations
        )

    def _optimize(self, X, states, U, cost=None, optimize_iterations=1):
        """

        :param np.ndarray X: initial states [T+1, N]
        :param np.ndarray U: initial controls [T, M]
        """
        n_not_decrease_cost = 0
        threshold_to_terminate_opt = 5
        best_X, best_U, total_min_cost = None, None, 10**10
        for iteration in range(optimize_iterations):
            t0 = time.time() * 1000

            if self._multi_mode:
                (
                    list_lx,
                    list_lxx,
                    list_lu,
                    list_luu,
                    list_lux,
                    list_fx,
                    list_fu,
                ) = self._env_pool_derivative.derivatives(X, states, U)
            else:
                list_lx, list_lxx, list_lu, list_luu, list_lux, list_fx, list_fu = self.derivative_sequential(
                    X, states, U
                )

            k, K = self.backward(X, U, list_lx, list_lxx, list_lu, list_luu, list_lux, list_fx, list_fu)
            t1 = time.time() * 1000

            if self._multi_mode:
                newX, new_states, newU, min_cost = self._env_pool_forward.forward(X, states, U, k, K, self._alpha_List)
            else:
                newX, new_states, newU, min_cost = self.sequential_forward(X, states, U, k, K, self._alpha_List)

            t2 = time.time() * 1000

            if cost is not None and total_min_cost < min_cost:
                self._logger.debug(
                    "cannot decrease cost : {:3.4f} > {:3.4f}, total_min_cost: {:3.4f} at iter: {}/{}".format(
                        total_min_cost, cost, total_min_cost, iteration, optimize_iterations
                    )
                )
                # break
                n_not_decrease_cost += 1
                if n_not_decrease_cost >= threshold_to_terminate_opt:
                    self._logger.debug(
                        "Terminate optimization because cost does not decreasae {} times".format(
                            threshold_to_terminate_opt
                        )
                    )
                    break
            if cost is None or min_cost < total_min_cost:
                best_X, best_states, best_U = newX, new_states, newU
                total_min_cost = min_cost
                n_not_decrease_cost = 0

            X, states, U = newX, new_states, newU
            cost = min_cost

            if self._measure_time:
                print("backward {:.1f} msec, forward {:.1f} msec  ".format(t1 - t0, t2 - t1))

        return best_X, best_states, best_U, total_min_cost

    def derivative_sequential(self, X, states, U):
        T = len(U)
        dim_control = U[0].shape[0]
        dim_state = X[0].shape[0]
        list_lx = []
        list_lxx = []
        list_fx = []
        list_fu = []
        list_lu = []
        list_luu = []
        list_lux = []

        for t in range(T + 1):
            state = states[t]
            self._work_env.set_state_vector(state)
            lx = self._work_env.Lx()
            lxx = self._work_env.Lxx()

            list_lx.append(lx)
            list_lxx.append(lxx)

            if t != T:
                u = U[t]
                lu = self._work_env.Lu(u)
                luu = self._work_env.Luu(u)
                lux = np.zeros((dim_control, dim_state), dtype=NP_DTYPE)
                fx, fu = self._work_env.FxFu(u)

                list_fx.append(fx)
                list_fu.append(fu)
                list_lu.append(lu)
                list_luu.append(luu)
                list_lux.append(lux)

        return list_lx, list_lxx, list_lu, list_luu, list_lux, list_fx, list_fu

    def backward(self, X, U, list_lx, list_lxx, list_lu, list_luu, list_lux, list_fx, list_fu):
        """
        Perform backward pass using available trajectories.
        This step calculates a new policy object.
        :param list[np.ndarray] X: state vectors [T+1, N]
        :param list[np.ndarray] U: control vectors [T, M]
        :return
            kList
            KList
        """
        # Get planning horizon and system dimensions
        T = len(U)
        dim_obs = X[0].shape[0]
        eta = 0

        fail = True
        while fail:
            fail = False

            Vx = list_lx[-1]
            Vxx = list_lxx[-1]

            kList = []
            KList = []

            etaEye = eta * np.eye(dim_obs, dtype=NP_DTYPE)

            # Compute state-action-state function at each time step
            # Ignore the second order dynamics for model
            for t in range(T - 1, -1, -1):
                lx, lxx, lu, luu, lux = list_lx[t], list_lxx[t], list_lu[t], list_luu[t], list_lux[t]
                fx, fu = list_fx[t], list_fu[t]

                # Compute the second-order expansion of pseudo-Hamiltonian Q
                Qx = lx + np.dot(fx.T, Vx)
                Qu = lu + np.dot(fu.T, Vx)
                Qxx = lxx + np.dot(np.dot(fx.T, Vxx), fx)
                Quu_d = luu + np.dot(np.dot(fu.T, (Vxx + etaEye)), fu)
                Qux_d = lux + np.dot(np.dot(fu.T, (Vxx + etaEye)), fx)

                Quu = luu + np.dot(np.dot(fu.T, Vxx), fu)
                Qux = lux + np.dot(np.dot(fu.T, Vxx), fx)

                if not util.is_pos_def(Quu_d):
                    self._logger.warning(
                        "eta iteration is run in backward(). This slows iLQR!!! "
                        + "Use bigger lam_control or parallelize backward()."
                    )
                    eta = np.max([eta * self._etaFactor, self._min_eta])

                    if eta > self._max_eta:
                        raise ValueError(
                            "Reached max iterations to find a PD Q mat-- Something is wrong! eta: {}".format(eta)
                        )
                    fail = True
                    break

                QuuInv = np.linalg.inv(Quu_d)
                k = -np.dot(QuuInv, Qu)
                K = -np.dot(QuuInv, Qux_d)

                kList.append(k)
                KList.append(K)

                Vx = Qx + np.dot(np.dot(K.T, Quu), k) + np.dot(K.T, Qu) + np.dot(Qux.T, k)
                Vxx = Qxx + np.dot(np.dot(K.T, Quu), K) + np.dot(K.T, Qux) + np.dot(Qux.T, K)

        kList.reverse()
        KList.reverse()

        return kList, KList

    def sequential_forward(self, X, states, U, k, K, alpha_list):
        min_cost = 8999999999

        for cur_alpha in alpha_list:
            cur_X, cur_states, cur_U, cur_cost = self.forward(X, states, U, k, K, cur_alpha)

            if cur_cost < min_cost:
                newX, new_states, newU = cur_X, cur_states, cur_U
                min_cost = cur_cost

        return newX, new_states, newU, min_cost

    def forward(self, X, states, U, k, K, alpha):
        """
        perform the LQR forward pass. Computes the state-action marginals from dynamics
        and policy
        :param list[np.ndarray] X: the previous state vectors [T+1, N]
        :param list[np.ndarray] U: the previous control vectors [T, M]
        :param list[np.ndarray] k: the previous control vectors [T, M]
        :param list[np.ndarray] K: the previous control vectors [T, M]

        :return:
        """
        T = len(U)

        newX = [X[0]]
        new_states = [states[0]]
        newU = []

        cost = 0
        x = X[0]
        state = states[0]
        env = self._work_env
        env.set_state_vector(state)

        for t in range(T):
            u = U[t] + alpha * k[t] + np.dot(K[t], x - X[t])
            u = np.clip(u, env.action_space.low, env.action_space.high)
            newU.append(u)

            cost_state = env.cost_state()
            cost_control = env.cost_control(u)
            cost += cost_state + cost_control

            # TODO : don't ignore a done flag
            x, reward, done, _ = env.step(u)
            state = env.get_state_vector()

            newX.append(x)
            new_states.append(state)

        cost += env.cost_state()

        return newX, new_states, newU, cost

    def close(self):
        del self._work_env
