# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np

NP_DTYPE = np.float64


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def plot_state(states, filename):
    if isinstance(states, list):
        states = np.array(states)
    assert states.ndim == 2

    fig, ax1 = plt.subplots()

    next_states = np.copy(states[1:])
    states = states[:-1]
    actions = next_states - states

    timesteps = np.arange(states.shape[0])

    ax1.plot(timesteps, np.rad2deg(states.T[0]), label="Absolute Board Angle X [deg]", color="b")
    ax1.plot(timesteps, np.rad2deg(states.T[1]), label="Absolute Board Angle Y [deg]", color="g")
    ax1.plot(timesteps, np.rad2deg(actions.T[0]), label="Relative Board Angle X [deg]", color="b", linestyle="dotted")
    ax1.plot(timesteps, np.rad2deg(actions.T[1]), label="Relative Board Angle Y [deg]", color="g", linestyle="dotted")

    ax2 = ax1.twinx()
    ax2.plot(timesteps, np.rad2deg(states.T[2]), label="theta [deg]", color="r")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Action (Board Angle) [deg]")
    ax1.set_ylim([-6, 6])
    ax2.set_ylabel("Theta (Boll Location) [deg]")
    fig.legend()
    fig.savefig(filename)

    plt.close()


def plot_thetas_theta_dots(states):
    if isinstance(states, list):
        states = np.array(states)
    assert states.ndim == 3

    fig = plt.figure(figsize=(12, 4))

    timesteps = np.arange(states[0].shape[0])

    titles = ["Board Angle X [deg]", "Board Angle Y [deg]", "Ball Angle X [deg]", "Ball Ang. Vel. [deg/sec]"]

    for title_idx, title in enumerate(titles):
        ax = fig.add_subplot(1, 4, title_idx + 1)
        for idx, state in enumerate(states):
            ax.plot(timesteps, np.rad2deg(state[:, title_idx]), label="{}".format(idx))
        ax.set_title(title, fontsize=13)

    plt.tight_layout()
    plt.show()
    plt.close()
