# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import logging
import os

import cv2

from CythonEnv.CyMazeWrapper import CyThetaObsMaze
from CythonEnv.CyRender import CyRender
from ilqr.ilqr import MjIterativeLQR
from ilqr.logger import initialize_logger
from ilqr.util import plot_thetas_theta_dots


def display(renderer, X):
    for x in X:
        obs = renderer.render(x)
        cv2.imshow("image", obs)
        cv2.waitKey(20)


def main(ring_idx, is_theta):
    logger = initialize_logger(logging_level=logging.DEBUG, save_log=False)

    dir_root = os.environ["MAZE_SIMULATOR_ROOT"]

    if not is_theta:
        config_path = os.path.join(dir_root, "experimental_settings/sim2sim/config_xyz_less_friction.txt")
    else:
        config_path = os.path.join(dir_root, "experimental_settings/sim2sim/config_theta_less_friction.txt")

    def make_env():
        maze = CyThetaObsMaze(config_path, lam_control=10.0, lam_vel=0.1, force_theta_grad=True)

        if is_theta:
            maze.set_ring_idx(ring_idx)
        return maze

    lqr = MjIterativeLQR(make_env, timesteps=30, num_alpha=8, force_single_thread=False)
    renderer = CyRender(config_path)

    if is_theta:
        renderer.set_ring_idx(ring_idx)

    lqr.initialize(policy="random")
    display(renderer, lqr.current_state_sequence)

    logger.info("initial trajectory : cost = {}".format(lqr.current_cost))
    states = [lqr.current_obs_sequence]

    for i in range(20):
        lqr.optimize()
        logger.info("step {} trajectory : cost = {}".format(i + 1, lqr.current_cost))
        display(renderer, lqr.current_state_sequence)
        states.append(lqr.current_obs_sequence)

    plot_thetas_theta_dots(states)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ring-idx", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--theta", action="store_true", default=False)

    args = parser.parse_args()
    main(args.ring_idx, args.theta)
