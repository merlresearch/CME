# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import numpy as np


def get_initial_states_16_points(dim_state=4):
    """
    Give scenarios with initial states with theta = np.pi*N/8 + np.pi/16 (N=-8, ..., 7).
    """
    assert dim_state == 4, "Currently only supports Theta Dynamics."

    initial_states = np.zeros((16, dim_state), dtype=np.float64)
    initial_thetas = np.array([x * np.pi / 8.0 + np.pi / 16.0 for x in range(-8, 8)])
    initial_states[:, 2] = initial_thetas
    return initial_states


def get_initial_states_n_points_limited(dim_state=4, n=9):
    """
    Give scenarios with initial states 8 different parts from -40~40 deg region
    """
    assert dim_state == 4, "Currently only supports Theta Dynamics."

    initial_states = np.zeros((n, dim_state), dtype=np.float64)
    initial_thetas = np.linspace(-2.0 * np.pi / 9.0, 2 * np.pi / 9, n)
    initial_states[:, 2] = initial_thetas
    return initial_states
