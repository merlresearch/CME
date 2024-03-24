# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse
import os

from experiments.task_manager import run_in_concurrent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-experiments", type=int, default=5)
    parser.add_argument("-n", "--concurrent", dest="concurrent", type=int, required=True)
    parser.add_argument("--dry", dest="dry_run", default=False, action="store_true")
    parser.add_argument("--max-steps", type=int, default=int(1e6))
    parser.add_argument("--episode-max-steps", type=int, default=500)
    parser.add_argument("--test-interval", type=int, default=10000)
    parser.add_argument("--logdir", default="results/rl")
    parser.add_argument("--force", action="store_true")
    # Experiments options
    parser.add_argument("--reset-to-random-ring", action="store_true")
    parser.add_argument("--input-state", action="store_true")
    parser.set_defaults(reset_to_random_ring=True)
    args = parser.parse_args()

    concurrent = args.concurrent
    dry_run = args.dry_run
    logdir = args.logdir

    common_args = ["--max-steps", str(args.max_steps)]
    common_args += ["--episode-max-steps", str(args.episode_max_steps)]
    common_args += ["--test-interval", str(args.test_interval)]
    if args.reset_to_random_ring:
        common_args += ["--reset-to-random-ring"]
    if args.input_state:
        common_args += ["--input-state"]

    list_args = []
    for policy in ["sac"]:
        filename = os.path.join("experiments/rl/run_{}.py".format(policy))
        assert os.path.exists(filename)
        rl_logdir = os.path.join(logdir, policy)
        os.makedirs(rl_logdir, exist_ok=True)
        cmd = ["python", filename]
        cmd += common_args
        cmd += ["--logdir", rl_logdir]
        for _ in range(args.n_experiments):
            list_args.append(cmd.copy())

    if dry_run:
        for c in list_args:
            print(" ".join(c))
    else:
        run_in_concurrent(list_args, concurrent, logdir=logdir, use_gpu=True)


if __name__ == "__main__":
    main()
