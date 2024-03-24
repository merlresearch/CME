<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Circular MAZE Environment

## Features

This repository contains scripts to control Circular MAZE Environment (CME) from python.

## Installation

All installation process is done on the following system configuration:

```bash
$ cat /etc/os-release
NAME="Ubuntu"
VERSION="18.04.1 LTS (Bionic Beaver)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 18.04.1 LTS"
VERSION_ID="18.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=bionic
UBUNTU_CODENAME=bionic
$ conda --v
conda 4.6.11
```

Please install CME by following [this document](./install_maze_simulator.md).

## Examples

### Control MAZE simulator from keyboard

Set environment variables by the following command.

```bash
$ source ./install/setup_env.sh
```

Run an example script as:

```bash
$ python experiments/exp_ilqr/run_mpc.py
```

### Experiments

For iLQR on simulator experiments, see [experiments/exp_ilqr](experiments/exp_ilqr/README.md).

For RL on simulator experiments, see [experiments/rl](experiments/rl/README.md).

## Citation

If you use the software, please cite the following ([TR2021-032](https://merl.com/publications/docs/TR2021-032.pdf)):

```BibTeX
    @article{Ota2021mar,
    author = {Ota, Kei and Jha, Devesh K. and Romeres, Diego and van Baar, Jeroen and Smith, Kevin and Semistsu, Takayuki and Oiki, Tomoaki and Sullivan, Alan and Nikovski, Daniel N. and Tenenbaum, Joshua B.},
    title = {Data-Efficient Learning for Complex and Real-Time Physical Problem Solving using Augmented Simulation},
    journal = {IEEE Robotics and Automation Letters},
    year = 2021,
    volume = 6,
    number = 2,
    month = mar,
    doi = {10.1109/LRA.2021.3068887},
    url = {https://www.merl.com/publications/TR2021-032}
    }
```

## Related Links

https://www.youtube.com/watch?v=xaxNCXBovpc&feature=youtu.be

## Contact

Please contact Devesh Jha at jha@merl.com

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below::

```
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

The file `CythonEnv/utils.py` was adapted with modifications. The original license file is included in [LICENSES/MIT.txt](LICENSES/MIT.txt).

```
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2017 Preferred Networks, Inc.

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
```
