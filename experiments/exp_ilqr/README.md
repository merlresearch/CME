<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# iLQR Experiments

## Installation

Before starting, please install maze_simulator by following [docs/install_maze_simulator.md](../../docs/install_maze_simulator.md).

## Experiments

### Run iLQR to solve CME

```bash
$ source ./install/setup_env.sh
$ python experiments/exp_ilqr/run_ilqr.py
```

### Run MPC to solve CME

```bash
$ source ./install/setup_env.sh
$ python experiments/exp_ilqr/run_mpc.py
```
