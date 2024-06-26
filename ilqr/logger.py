# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import datetime
import logging
import os
from logging import FileHandler, Formatter, StreamHandler, getLogger


def initialize_logger(
    logging_level=logging.INFO, output_dir="results/", filename=None, logger_name="maze_controller", save_log=True
):
    logger = getLogger(logger_name)
    logger.setLevel(logging_level)

    handler_format = Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s", "%H:%M:%S"
    )  # If you want to show date, add %Y-%m-%d
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(handler_format)

    if save_log:
        if filename is not None:
            filename = filename
        else:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            filename = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f") + ".log")

        file_handler = FileHandler(filename, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(handler_format)

    if len(logger.handlers) == 0:
        logger.addHandler(stream_handler)
        if save_log:
            logger.addHandler(file_handler)
    else:
        # Overwrite logging setting
        logger.handlers[0] = stream_handler
        if save_log:
            logger.handlers[1] = file_handler

    logger.propagate = False

    return logger
