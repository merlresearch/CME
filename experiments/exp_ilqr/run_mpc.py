# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import logging
import os
import time

import cv2

from CythonEnv.CyMazeWrapper import CyThetaObsMaze
from CythonEnv.CyRender import CyRender
from experiments.simulation import get_initial_states_16_points
from ilqr.ilqr import MjIterativeLQR
from ilqr.logger import initialize_logger


def display(renderer, X):
    for x in X:
        obs = renderer.render(x)
        cv2.imshow("image", obs)
        cv2.waitKey(60)


def main(ring_idx, is_theta):
    logger = initialize_logger(logging_level=logging.DEBUG, save_log=False)

    dir_root = os.environ["MAZE_SIMULATOR_ROOT"]

    if is_theta:
        config_path = os.path.join(dir_root, "experimental_settings/sim2sim/config_theta_less_friction.txt")
    else:
        config_path = os.path.join(dir_root, "experimental_settings/sim2sim/config_xyz_less_friction.txt")

    T = 15

    def make_env():
        maze = CyThetaObsMaze(config_path, lam_control=10.0, lam_vel=0.01, force_theta_grad=False)

        if is_theta:
            maze.set_ring_idx(ring_idx)

        return maze

    initial_states = get_initial_states_16_points()

    lqr = MjIterativeLQR(make_env, timesteps=T, num_alpha=8, force_single_thread=False)
    renderer = CyRender(config_path)
    real_env = make_env()

    total_steps = []
    for initial_state in initial_states:
        real_env.reset()
        real_env.set_state_with_ring_and_theta(ring_idx, initial_state[2])

        if is_theta:
            renderer.set_ring_idx(ring_idx)

        lqr.initialize(real_env.get_state_vector(), policy="random")
        lqr.optimize(optimize_iterations=5)

        logger.info("initial trajectory : cost = {}".format(lqr.current_cost))
        observations = [lqr.current_obs_sequence]

        cur_x, _, _, _ = real_env.step(lqr.current_control_sequence[0])

        total_time = 0

        for i in range(600):
            t0 = time.time() * 1000

            next_action = lqr.update_s1(real_env.get_state_vector())
            cur_x, _, _, _ = real_env.step(next_action)

            t1 = time.time() * 1000
            total_time += t1 - t0
            logger.info("mpc step {:.1f} ms at {}/{}".format(t1 - t0, i, 600))

            logger.info("step {} trajectory : cost = {:.2f}".format(i + 1, lqr.current_cost))
            obs = renderer.render(real_env.get_state_vector())
            cv2.imshow("real_step", obs)
            cv2.waitKey(1)
            observations.append(lqr.current_obs_sequence)
            if real_env.get_ring_idx() == 0:
                break
        total_steps.append(i + 1)

    print(total_time, total_steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ring-idx", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--theta", action="store_true", default=False)
    args = parser.parse_args()
    main(args.ring_idx, args.theta)
