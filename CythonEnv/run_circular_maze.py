# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import os
import time

import cv2
import numpy as np

from CythonEnv.CyMaze import _CyMaze
from CythonEnv.CyMazeWrapper import CyThetaObsMaze
from CythonEnv.CyRender import CyRender


def main():
    config_name = "config.txt"
    config_theta_name = "config_theta.txt"
    # config_name = "config_theta.txt"
    # config_name = "config_nowall.txt"

    config_path = os.path.join(os.environ["MAZE_SIMULATOR_ROOT"], "Configuration", config_name)
    config_theta_path = os.path.join(os.environ["MAZE_SIMULATOR_ROOT"], "Configuration", config_theta_name)

    assert os.path.isfile(
        config_path
    ), "Cannot find {}. You should set MAZE_SIMULATOR_ROOT. See README.md for details.".format(config_path)

    maze = CyThetaObsMaze(config_path)
    maze_theta = CyThetaObsMaze(config_theta_path)
    render = CyRender(config_path)
    maze.reset()

    degree = 0.3
    fric_slide, fric_spin, fric_roll = 0.5, 0.0001, 0.00001
    fricloss = 0.00000001

    while True:
        k = cv2.waitKey(20)

        if k == ord("d"):
            xRot, yRot = 0.0, degree
        elif k == ord("a"):
            xRot, yRot = 0.0, -degree
        elif k == ord("s"):
            xRot, yRot = degree, 0.0
        elif k == ord("w"):
            xRot, yRot = -degree, 0.0
        elif k == ord("t"):
            maze.reset()
        elif k == ord("r"):
            obs = maze.get_obs()
            idx_ring = maze.get_ring_idx()
            maze_theta.set_state_vector_with_ring(idx_ring, obs)
        elif k == ord("q"):
            fric_slide = fric_slide / 10
            fric_spin = fric_spin / 10
            fric_roll = fric_roll / 10

            maze.set_fric(fric_slide, fric_spin, fric_roll)
            maze_theta.set_fric(fric_slide, fric_spin, fric_roll)
        elif k == ord("e"):
            fric_slide = fric_slide * 10
            fric_spin = fric_spin * 10
            fric_roll = fric_roll * 10
            maze.set_fric(fric_slide, fric_spin, fric_roll)
            maze_theta.set_fric(fric_slide, fric_spin, fric_roll)
        elif k == ord("g"):
            fricloss *= 10
            print("fricloss = {}".format(fricloss))
            maze.set_fricloss(fricloss)
        else:
            xRot, yRot = 0.0, 0.0

        xRot = math.radians(xRot)
        yRot = math.radians(yRot)
        action = np.array((xRot, yRot))

        obs, reward, done, _ = maze.step(np.array(action))
        obs2, reward, done, _ = maze_theta.step(action)

        image = render.render(maze.get_xyz_vec())
        cv2.imshow("xyz_maze", image)

        image2 = render.render(maze_theta.get_xyz_vec())
        cv2.imshow("theta_maze", image2)


if __name__ == "__main__":
    main()
