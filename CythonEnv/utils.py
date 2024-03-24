# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2017 Preferred Networks, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

standard_library.install_aliases()  # NOQA

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import tempfile
from logging import FileHandler, Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def is_return_code_zero(args):
    """
    Return true if the given command's return code is zero.
    All the messages to stdout or stderr are suppressed.
    forked from https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/is_return_code_zero.py
    """
    with open(os.devnull, "wb") as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


def is_under_git_control():
    """
    Return true iff the current directory is under git control.
    """
    return is_return_code_zero(["git", "rev-parse"])


def prepare_output_dir(
    args, user_specified_dir=None, argv=None, time_format="%Y%m%dT%H%M%S.%f", suffix="", preferred_filename=None
):
    """
    Prepare a directory for outputting training results.
    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:
        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables
    Additionally, if the current directory is under git control, the following
    information is saved:
        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff`

    :param args (dict or argparse.Namespace): Arguments to save
    :param user_specified_dir (str or None): If str is specified, the output
        directory is created under that path. If not specified, it is
        created as a new temporary directory instead.
    :param argv (list or None): The list of command line arguments passed to a
        script. If not specified, sys.argv is used instead.
    :param time_format (str): Format used to represent the current datetime. The
    :param default format is the basic format of ISO 8601.
    :return: Path of the output directory created by this function (str).
    """
    if suffix is not "":
        suffix = "_" + suffix
    time_str = datetime.datetime.now().strftime(time_format) + suffix
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError("{} is not a directory".format(user_specified_dir))
        if preferred_filename is None:
            outdir = os.path.join(user_specified_dir, time_str)
        else:
            outdir = os.path.join(user_specified_dir, preferred_filename)
        if os.path.exists(outdir):
            raise RuntimeError("{} exists".format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)

    # Save all the arguments
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, "environ.txt"), "w") as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    if is_under_git_control():
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, "git-head.txt"), "wb") as f:
            f.write(subprocess.check_output("git rev-parse HEAD".split()))

        # Save `git status`
        with open(os.path.join(outdir, "git-status.txt"), "wb") as f:
            f.write(subprocess.check_output("git status".split()))

        # Save `git log`
        with open(os.path.join(outdir, "git-log.txt"), "wb") as f:
            f.write(subprocess.check_output("git log".split()))

        # Save `git diff`
        with open(os.path.join(outdir, "git-diff.txt"), "wb") as f:
            f.write(subprocess.check_output("git diff".split()))

    return outdir


def initialize_logger(logging_level=logging.INFO, output_dir="logs/"):
    logger = getLogger("maze_simulator")
    logger.setLevel(logging_level)

    handler_format = Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s", "%H:%M:%S"
    )  # If you want to show date, add %Y-%m-%d
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(handler_format)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_handler = FileHandler(output_dir + "/" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f") + ".log", "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(handler_format)

    if len(logger.handlers) == 0:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    else:
        # Overwrite logging setting
        assert len(logger.handlers) == 2
        logger.handlers[0] = stream_handler
        logger.handlers[1] = file_handler

    return logger


def frames_to_gif(frames, prefix, save_dir, interval=50, fps=30):
    """
    Convert frames to gif file
    """
    assert len(frames) > 0
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    # TODO: interval should be 1000 / fps ?
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)
    output_path = "{}/{}.gif".format(save_dir, prefix)
    anim.save(output_path, writer="imagemagick", fps=fps)
    print("Generated a movie in ", output_path)


def get_sinusoidal(ampl_vec, freq_vec, sim_length, fps=30.0):
    """
    Sum of sinusoid with amplitude, frequences defined by ampl_vec and freq_vec.
    (frequences is expressed in rad/sec)
    """
    f_t = lambda steps: np.sum(ampl_vec * np.sin((freq_vec * steps * (1 / fps))))
    t = np.arange(0, sim_length).reshape(sim_length, 1)
    trj = np.apply_along_axis(f_t, 1, t)
    return trj


if __name__ == "__main__":
    n_sines = 5
    max_freq = 10
    max_amp = 1

    ampl_vec = np.random.rand(1, n_sines) * max_amp
    freq_vec = np.linspace(1, max_freq, num=n_sines)
    sim_length = 100
    f = get_sinusoidal(ampl_vec, freq_vec, sim_length)
    print(f)

    plt.figure()
    plt.plot(f)
    plt.show()
