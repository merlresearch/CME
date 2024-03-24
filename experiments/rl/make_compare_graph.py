# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import glob
import logging
import os
import re
import struct

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing.event_accumulator import DEFAULT_SIZE_GUIDANCE, TENSORS, EventAccumulator


def get_score_trajectory_from_tfsummary(path_file, summary_name="Common/average_test_return"):
    assert os.path.exists(path_file)

    size_guidance = DEFAULT_SIZE_GUIDANCE.copy()
    size_guidance[TENSORS] = 0

    accumulator = EventAccumulator(path_file, size_guidance=size_guidance)
    accumulator.Reload()

    eval_loss = accumulator.Tensors(summary_name)

    steps = [tmp.step for tmp in eval_loss]
    scores = [struct.unpack("f", tmp.tensor_proto.tensor_content)[0] for tmp in eval_loss]

    return steps, scores


def aggregate_actual_scores(list_file, summary_name="Common/average_test_return"):
    list_trials = []

    for cur_file in list_file:
        try:
            steps, scores = get_score_trajectory_from_tfsummary(cur_file, summary_name)
        except KeyError:
            continue

        list_trials.append((steps, scores))

    return list_trials


def average_actual_scores(list_scores, sampling_interval=10000, max_steps=0, allow_interpolate=True):
    """

    :return: (step_axis, mean_scores, error_values, all_scores)

    step_axis : x-axis values
    mean_scores : average score along step_axis
    error_values : standard deviation along step_axis
    all_scores: all scores including average and standard deviation
    """
    logger = logging.getLogger(__name__)

    if max_steps == 0:
        for steps, scores in list_scores:
            if np.max(steps) > max_steps:
                max_steps = np.max(steps)

    min_steps = 1000000000
    for steps, scores in list_scores:
        if np.min(steps) < min_steps:
            min_steps = np.min(steps)
    min_steps = max(min_steps, 10000)

    new_list_scores = []

    for steps, scores in list_scores:
        if np.max(steps) >= max_steps:
            new_steps = []
            new_scores = []
            for cur_step, cur_score in zip(steps, scores):
                if cur_step > max_steps:
                    break

                new_steps.append(cur_step)
                new_scores.append(cur_score)

            steps = np.array(new_steps)
            scores = np.array(new_scores)

            new_list_scores.append((steps, scores))
        else:
            logger.info("skip experiment length {}".format(np.max(steps)))

    list_scores = new_list_scores

    if len(list_scores) == 0:
        logger.warning("skip incompleted experiment")
        return None

    mean_scores = []
    errors = []
    all_scores = []
    # step_axis = np.arange(15000, max_steps + 1, step=sampling_interval)
    step_axis = np.arange(min_steps, max_steps + 1, step=sampling_interval)

    if allow_interpolate:
        list_functions = []

        for steps, scores in list_scores:
            list_functions.append(interp1d(steps, scores))

        for cur_step in step_axis:
            values = [f(cur_step) for f in list_functions]
            mean_value = np.mean(values)
            mean_scores.append(mean_value)
            all_scores.append(values)
    else:
        list_dicts = []

        for steps, scores in list_scores:
            cur_dict = {}
            for cur_step, cur_score in zip(steps, scores):
                cur_dict[cur_step] = cur_score
            list_dicts.append(cur_dict)

        for cur_step in step_axis:
            values = [d[cur_step] for d in list_dicts]
            all_scores.append(values)
            mean_value = np.mean(values)
            error_value = np.std(values)
            mean_scores.append(mean_value)
            errors.append(error_value)

    mean_scores = np.array(mean_scores)
    error_values = np.array(errors)
    all_scores = np.array(all_scores).transpose()

    return step_axis, mean_scores, error_values, all_scores


def enumerate_flat(root):
    # key=> method_name, valud => list_file
    dict_methods = {}
    list_file = glob.glob("{}/*/*/*tfevents*".format(root), recursive=True)

    for cur_file in list_file:
        parent_name = os.path.relpath(cur_file, start=root).split("/")[0]

        if parent_name not in dict_methods:
            dict_methods[parent_name] = []

        dict_methods[parent_name].append(cur_file)

    print(dict_methods.keys())
    return dict_methods


def main():
    logging.basicConfig(
        datefmt="%d/%Y %I:%M:%S",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="dir_log", type=str, default="results/rl")
    parser.add_argument("--prefix", type=str, default="result")
    parser.add_argument("--max_steps", default=0, type=int)
    parser.add_argument("--legend", action="store_true", default=False)
    args = parser.parse_args()

    max_steps = args.max_steps

    dict_methods = enumerate_flat(args.dir_log)
    method_order = sorted(dict_methods.keys())

    figsize = (8, 5)
    random_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    font_size = 16

    summary_names = ["Common/average_test_return", "Common/average_test_steps"]
    for summary_name in summary_names:
        fig_return, ax_return = plt.subplots(figsize=figsize, dpi=300)
        for idx, method in enumerate(method_order):
            list_file = dict_methods[method]
            logger.info("analyze {} has {} seeds".format(method, len(list_file)))

            list_scores = aggregate_actual_scores(list_file, summary_name)

            cur_steps, cur_mean_scores, cur_errrors_scores, _ = average_actual_scores(
                list_scores=list_scores, max_steps=max_steps, allow_interpolate=False
            )

            if summary_name == "Common/average_test_steps" and np.max(cur_mean_scores) > 5000:
                cur_mean_scores /= 16
                cur_errrors_scores /= 16

            method_name = method.replace("_", "/")
            color = random_colors[idx % len(random_colors)]
            line_style = "-"

            cur_steps = cur_steps / 1000000.0

            ax_return.plot(
                cur_steps, cur_mean_scores, color=color, label=method_name, linewidth=1.0, linestyle=line_style
            )
            ax_return.fill_between(
                cur_steps,
                cur_mean_scores - cur_errrors_scores,
                cur_mean_scores + cur_errrors_scores,
                facecolor=color,
                alpha=0.2,
            )

            ax_return.set_xlabel("million steps", fontsize=font_size)
            ylabel = (
                "average steps needed to reach goal"
                if summary_name == "Common/average_test_steps"
                else "average return"
            )
            ax_return.set_ylabel(ylabel, fontsize=font_size)

            plt.tick_params(labelsize=font_size)
            ax_return.grid(which="major", color="black", linestyle="-", alpha=0.15)
            ax_return.grid(which="minor", color="black", linestyle="-", alpha=0.15)

            if args.legend:
                # legend = plt.legend(loc='lower right', borderaxespad=0, fontsize=15)
                legend = plt.legend(fontsize=font_size, framealpha=1.0)
                fig_return = legend.figure
                fig_return.canvas.draw()
                bbox = legend.get_window_extent().transformed(fig_return.dpi_scale_trans.inverted())
                fig_return.savefig("legend.png", dpi="figure", bbox_inches=bbox)

            plt.tight_layout()
            if args.prefix is not None:
                fig_return.savefig(args.prefix + "_" + summary_name.split("/")[-1] + ".png", figsize=figsize, dpi=300)
            else:
                plt.show()
        plt.close()


if __name__ == "__main__":
    main()
