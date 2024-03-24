# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import ctypes
import multiprocessing as mp

import cloudpickle
import numpy as np

from ilqr.util import NP_DTYPE

CT_DTYPE = ctypes.c_double


def worker_process(
    pickled_env, arg_areas, result_areas, idx_process, num_processes, T, barrier_start, barrier_finish, debug
):
    make_env = cloudpickle.loads(pickled_env)
    lx_area, lxx_area, fx_area, fu_area = result_areas
    x_area, s_area, u_area = arg_areas

    cur_env = make_env()
    if debug:
        cur_env.set_state_vector(np.zeros(4, dtype=np.float64))

    np_lx = lx_area.get_np()
    np_lxx = lxx_area.get_np()
    np_fx = fx_area.get_np()
    np_fu = fu_area.get_np()
    np_x = x_area.get_np()
    np_s = s_area.get_np()
    np_u = u_area.get_np()

    barrier_start.wait()

    while True:
        barrier_start.wait()

        for idx_x in range(idx_process, T + 1, num_processes):
            cur_env.set_state_vector(np_s[idx_x])
            np_lx[idx_x] = cur_env.Lx()
            np_lxx[idx_x] = cur_env.Lxx()

            # list_U is shorter than list_X
            if idx_x != T:
                u = np_u[idx_x]
                np_fx[idx_x], np_fu[idx_x] = cur_env.FxFu(u)

        # we can use barrier_start
        barrier_finish.wait()


class SectionArea:
    def __init__(self, ctx, shape: tuple):
        size = int(np.prod(shape))
        self.shared = ctx.Array(CT_DTYPE, size, lock=False)
        self.shape = shape
        self.dtype = NP_DTYPE

    def get_np(self):
        return np.reshape(np.frombuffer(self.shared, dtype=self.dtype), self.shape)


class EnvProcessPool:
    """environments pool"""

    def __init__(self, make_env, dim_obs, dim_state, dim_action, T, num_processes, debug=False):
        self._num_processes = num_processes
        ctx = mp.get_context("fork")
        self.ctx = ctx

        self.dim_obs = dim_obs
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.T = T

        lx_shape = (T + 1, dim_obs)
        lxx_shape = (T + 1, dim_obs, dim_obs)
        fx_shape = (T, dim_obs, dim_obs)
        fu_shape = (T, dim_obs, dim_action)

        # result area
        self.lx_area = SectionArea(ctx, lx_shape)
        self.lxx_area = SectionArea(ctx, lxx_shape)
        self.fx_area = SectionArea(ctx, fx_shape)
        self.fu_area = SectionArea(ctx, fu_shape)

        # argument area
        x_shape = (T + 1, dim_obs)
        s_shape = (T + 1, dim_state)
        u_shape = (T, dim_action)
        self.x_area = SectionArea(ctx, x_shape)
        self.s_area = SectionArea(ctx, s_shape)
        self.u_area = SectionArea(ctx, u_shape)

        self.barrier_start = ctx.Barrier(parties=num_processes + 1)
        self.barrier_finish = ctx.Barrier(parties=num_processes + 1)
        self.list_proc = []

        pickled_env = cloudpickle.dumps(make_env)
        result_areas = (self.lx_area, self.lxx_area, self.fx_area, self.fu_area)
        arg_areas = (self.x_area, self.s_area, self.u_area)

        for idx_process in range(num_processes):
            args = (
                pickled_env,
                arg_areas,
                result_areas,
                idx_process,
                num_processes,
                T,
                self.barrier_start,
                self.barrier_finish,
                debug,
            )

            proc = ctx.Process(target=worker_process, args=args, daemon=True)
            proc.start()
            self.list_proc.append(proc)

        # wait for make_env
        self._work_env = make_env()
        self.barrier_start.wait()

        self.np_lx = self.lx_area.get_np()
        self.np_lxx = self.lxx_area.get_np()
        self.np_fx = self.fx_area.get_np()
        self.np_fu = self.fu_area.get_np()
        self.np_x = self.x_area.get_np()
        self.np_s = self.s_area.get_np()
        self.np_u = self.u_area.get_np()

    def __del__(self):
        print("Temporary comment out to prevent following bug")
        print("AttributeError: 'ForkProcess' object has no attribute 'close'")
        # for p in self.list_proc:
        #     p.close()

    def derivatives(self, list_X, list_S, list_U):
        x = np.array(list_X)
        s = np.array(list_S)
        u = np.array(list_U)

        np.copyto(self.np_x, x)
        np.copyto(self.np_s, s)
        np.copyto(self.np_u, u)

        self.barrier_start.wait()

        list_lu = np.zeros((self.T, self.dim_action), dtype=NP_DTYPE)
        list_luu = np.zeros((self.T, self.dim_action, self.dim_action), dtype=NP_DTYPE)
        list_lux = np.zeros((self.T, self.dim_action, self.dim_obs), dtype=NP_DTYPE)

        for idx_u in range(self.T):
            list_lu[idx_u] = self._work_env.Lu(u[idx_u])
            list_luu[idx_u] = self._work_env.Luu(u[idx_u])

        self.barrier_finish.wait()

        return self.np_lx, self.np_lxx, list_lu, list_luu, list_lux, self.np_fx, self.np_fu


def forward_process(
    pickled_env, arg_areas, result_areas, idx_process, num_processes, T, barrier_start, barrier_finish, debug
):
    make_env = cloudpickle.loads(pickled_env)

    newX_area, newS_area, newU_area, cost_area = result_areas
    X_area, S_area, U_area, k_area, K_area, alpha_area = arg_areas

    env = make_env()

    newX = newX_area.get_np()
    newS = newS_area.get_np()
    newU = newU_area.get_np()
    costRes = cost_area.get_np()

    X = X_area.get_np()
    S = S_area.get_np()
    U = U_area.get_np()
    K = K_area.get_np()
    k = k_area.get_np()
    alpha_list = alpha_area.get_np()

    T = len(U)

    barrier_start.wait()

    while True:
        barrier_start.wait()
        alpha = alpha_list[idx_process]

        state = S[0]
        newS[idx_process, 0] = state
        x = X[0]
        newX[idx_process, 0] = x

        env.set_state_vector(state)
        cost = 0

        for t in range(T):
            u = U[t] + alpha * k[t] + np.dot(K[t], x - X[t])
            u = np.clip(u, env.action_space.low, env.action_space.high)

            newU[idx_process, t] = u

            cost_state = env.cost_state()
            cost_control = env.cost_control(u)
            cost += cost_state + cost_control

            x, reward, done, _ = env.step(u)
            state = env.get_state_vector()

            newX[idx_process, t + 1] = x
            newS[idx_process, t + 1] = state

        cost += env.cost_state()

        costRes[idx_process] = cost
        barrier_finish.wait()


class EnvForwardPool:
    """environments pool"""

    def __init__(self, make_env, dim_obs, dim_state, dim_action, T, alpha_list, num_processes, debug=False):
        self._num_processes = num_processes
        ctx = mp.get_context("spawn")

        self.alpha_list = alpha_list
        num_alpha = len(self.alpha_list)

        self.ctx = ctx

        self.dim_obs = dim_obs
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.T = T

        newX_shape = (num_alpha, T + 1, dim_obs)
        newState_shape = (num_alpha, T + 1, dim_state)
        newU_shape = (num_alpha, T, dim_action)
        cost_shape = (num_alpha,)

        # result area
        # newX, newU, cost
        self.newX_area = SectionArea(ctx, newX_shape)
        self.newS_area = SectionArea(ctx, newState_shape)
        self.newU_area = SectionArea(ctx, newU_shape)
        self.cost_area = SectionArea(ctx, cost_shape)

        # argument area
        x_shape = (T + 1, dim_obs)
        state_shape = (T + 1, dim_state)
        u_shape = (T, dim_action)
        K_shape = (T, dim_action, dim_obs)
        self.X_area = SectionArea(ctx, x_shape)
        self.S_area = SectionArea(ctx, state_shape)
        self.U_area = SectionArea(ctx, u_shape)
        self.k_area = SectionArea(ctx, u_shape)
        self.K_area = SectionArea(ctx, K_shape)
        self.alpha_area = SectionArea(ctx, cost_shape)

        self.barrier_start = ctx.Barrier(parties=num_processes + 1)
        self.barrier_finish = ctx.Barrier(parties=num_processes + 1)
        self.list_proc = []

        pickled_env = cloudpickle.dumps(make_env)
        result_areas = (self.newX_area, self.newS_area, self.newU_area, self.cost_area)
        arg_areas = (self.X_area, self.S_area, self.U_area, self.k_area, self.K_area, self.alpha_area)

        for idx_process in range(num_processes):
            args = (
                pickled_env,
                arg_areas,
                result_areas,
                idx_process,
                num_processes,
                T,
                self.barrier_start,
                self.barrier_finish,
                debug,
            )

            proc = ctx.Process(target=forward_process, args=args, daemon=True)
            proc.start()
            self.list_proc.append(proc)

        self.barrier_start.wait()

        self.newX = self.newX_area.get_np()
        self.newS = self.newS_area.get_np()
        self.newU = self.newU_area.get_np()
        self.cost = self.cost_area.get_np()

        self.X = self.X_area.get_np()
        self.S = self.S_area.get_np()
        self.U = self.U_area.get_np()
        self.K = self.K_area.get_np()
        self.k = self.k_area.get_np()
        self.alpha = self.alpha_area.get_np()

    def forward(self, X, S, U, k, K, alpha_list):
        np.copyto(self.X, X)
        np.copyto(self.S, S)
        np.copyto(self.U, U)
        np.copyto(self.k, k)
        np.copyto(self.K, K)
        np.copyto(self.alpha, alpha_list)

        self.barrier_start.wait()
        self.barrier_finish.wait()

        idx_min = np.argmin(self.cost)

        return self.newX[idx_min], self.newS[idx_min], self.newU[idx_min], self.cost[idx_min]
