# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import csv
import logging
import math
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from CythonEnv.utils import frames_to_gif, get_sinusoidal


def make_log(episode_step, state, next_state, action):
    """
    :param episode_step (int): Current number of steps in an episode
    :param state (np.ndarray): MuJoCo state
    :param next_state (np.ndarray): MuJoCo next state
    :param action (np.ndarray): Action to the MAZE. Should be **ABSOLUTE** servo angle in **RADIAN**
    :return (list): Log consists of above four contents
    """
    assert isinstance(state, np.ndarray) and isinstance(next_state, np.ndarray) and isinstance(action, np.ndarray)
    ret = [episode_step]
    ret.extend(state.tolist())
    ret.extend(next_state.tolist())
    ret.extend(action.tolist())
    return ret


def get_transitions_from_path(transition_path):
    logger = logging.getLogger("maze_simulator")

    assert os.path.isfile(transition_path)
    transition_reader = csv.reader(open(transition_path, "r", newline="\n"))

    states, next_states, actions = None, None, None

    for row in transition_reader:
        state = np.array(row[1:5], dtype=np.float64)
        next_state = np.array(row[5:9], dtype=np.float64)
        action = np.array(row[9:11], dtype=np.float64)
        if states is None:
            states = state
            next_states = next_state
            actions = action
        else:
            states = np.vstack((states), state)
            next_states = np.vstack((next_states), next_state)
            actions = np.vstack((actions, action))

    if states is None:
        logger.error("Cannot find data.")
        raise ValueError

    return states, next_states, actions


def _get_reset_state(transition_path, episode_max_steps=200, real_setup=False):
    logger = logging.getLogger("maze_simulator")
    logger.info("Load state action pairs from {}".format(transition_path))

    reset_states = None
    transition_reader = csv.reader(open(transition_path, "r", newline="\n"))

    for row in transition_reader:
        reset_state = np.array([row[1:5]], dtype=np.float64)
        if reset_states is None:
            reset_states = reset_state
        else:
            reset_states = np.vstack((reset_states, reset_state))

    return reset_states


def get_command_line_args_for_collect_data(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-max-steps", type=int, default=30 * 100, help="Maximum steps in an episode"  # 100 seconds
    )
    parser.add_argument("--config-path", type=str, default=None, help="Maximum steps in an episode")
    parser.add_argument("--transition-path", type=str, default=None, help="Maximum steps in an episode")
    parser.add_argument("--show-process", action="store_true", help="Visualize movement")
    parser.add_argument("--save-img", action="store_true", help="Save image as PNG file")
    parser.add_argument("--save-movie", action="store_true", help="Save image as PNG file")
    parser.add_argument("--logging-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", help="Logging level")
    parser.add_argument("--reset-state", type=bool, default=False, help="Reset state to measure residual")
    parser.add_argument("--file-name", type=str, default=None, help="desired filename for data")
    return parser


def calculate_stepwise_error(oracle_future_states, stepwise_future_states):
    """
    Calculate stepwise errors
    :param np.ndarray(N, dim_state) oracle_future_states: Oracle consecutive transitions.
    :param np.ndarray(max_steps, N - (max_steps - 1), dim_state) stepwise_future_states:
        list of stepwise future_states. Each index(idx) corresponds to [idx] steps ahead future state.
        i.e. [1 step future, 2 step future, ...]
    :return np.ndarray(max_steps, ): Stepwise error array. Each index(idx) represents error at (idx) step future states.
    """
    max_steps, _, dim_state = stepwise_future_states.shape
    N, _ = oracle_future_states.shape
    stepwise_oracle_future_states = np.zeros_like(stepwise_future_states)
    for i in range(max_steps):
        stepwise_oracle_future_states[i, :, :] = oracle_future_states[i : N - (max_steps - 1) + i, :]
    return np.linalg.norm(stepwise_oracle_future_states - stepwise_future_states, axis=(1, 2), ord=2)


def save_simulations_on_two_maze(
    rel_actions_rad,
    env1,
    env2,
    initial_state,
    isShowResults=False,
    render=None,
    save_dir=None,
    prefix1=None,
    prefix2=None,
):
    """
    Show simulations with a same initial state and actions on two different maze envs.
    :param rel_actions_rad:
    :param env1:
    :param env2:
    :param initial_state:
    :param render:
    :return:
    """
    env1.reset()
    env2.reset()
    env1.set_state_vector(initial_state)
    env2.set_state_vector(initial_state)
    frames1 = []
    frames2 = []
    for action in rel_actions_rad:
        next_state1, _, _, _ = env1.step(action)
        next_state2, _, _, _ = env2.step(action)
        if render is not None:
            image_env1 = render.render(next_state1)
            image_env2 = render.render(next_state2)
            frames1.append(image_env1)
            frames2.append(image_env2)
            if isShowResults:
                cv2.imshow("image_env1", image_env1)
                cv2.imshow("image_env2", image_env2)
                cv2.waitKey(200)
    if save_dir is not None:
        frames_to_gif(frames1, prefix=prefix1, save_dir=save_dir, interval=1)
        frames_to_gif(frames2, prefix=prefix2, save_dir=save_dir, interval=1)
    if isShowResults:
        cv2.waitKey(200)


def plot_input_data(data, log_dir, length=None):
    if length is not None:
        data = data[:length]
    data = data.T

    timesteps = np.arange(data[0].shape[0])
    states = data[1:5]
    actions = data[9:11]

    fig, ax1 = plt.subplots()
    states = np.array(states)
    # Plot board angle
    ax1.plot(timesteps, np.rad2deg(states[0]), label="Board Angle X [deg]", color="b")
    ax1.plot(timesteps, np.rad2deg(states[1]), label="Board Angle Y [deg]", color="g")

    # Plot board angle
    ax1.plot(timesteps, np.rad2deg(actions[0]), label="Control Signal X [deg]", color="c")
    ax1.plot(timesteps, np.rad2deg(actions[1]), label="Control Signal Y [deg]", color="m")

    # Plot theta
    ax2 = ax1.twinx()
    ax2.plot(timesteps, np.rad2deg(states[2]), label="Theta [deg]", color="r")
    ax2.plot(timesteps, np.rad2deg(states[3]), label="Theta dot [deg/s]", color="y")

    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Action (Board Angle) [deg]")
    # ax1.set_ylim([-6, 6])
    ax2.set_ylabel("Theta [deg], Theta dot [deg/s]")
    fig.legend()
    fig.savefig(os.path.join(log_dir, "input_data.png"))


def safe_angular_convolve(data, filter):
    d = (len(filter) - 1) // 2
    data = np.pad(data, [d, d], "edge")
    sins = np.sin(data)
    coss = np.cos(data)

    sins = np.convolve(sins, filter, mode="valid")
    coss = np.convolve(coss, filter, mode="valid")

    c = np.arctan2(sins, coss)

    return c


def safe_convolve(data, filter=None, filter_width=5):
    if filter is None:
        n_moving_average = filter_width
        filter = np.ones(n_moving_average) / n_moving_average

    d = (len(filter) - 1) // 2
    data = np.pad(data, [d, d], "edge")
    c = np.convolve(data, filter, mode="valid")
    return c


def radian_error(gt_radians, estimated_radians):
    def diff_pi(dst, src):
        a = dst - src
        a = (a + math.pi) % (2 * math.pi) - math.pi
        return a

    error = np.mean(np.abs(diff_pi(gt_radians, estimated_radians)))
    return error


def copy_config_file_recursively(orig_config_path, dst_dir_path):
    assert os.path.isfile(orig_config_path), "{} not found".format(orig_config_path)
    assert os.path.isdir(dst_dir_path), "{} not found".format(dst_dir_path)
    _, config_filename = os.path.split(orig_config_path)
    count = 0
    while True:
        dst_file_path = os.path.join(dst_dir_path, config_filename + "_copy_" + str(count))
        if not os.path.isfile(dst_file_path):
            shutil.copyfile(orig_config_path, dst_file_path)
            return dst_file_path
        count += 1


def diff_rad(theta_rad_1, theta_rad_2):
    theta_diff = theta_rad_1 - theta_rad_2
    theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi
    return theta_diff
