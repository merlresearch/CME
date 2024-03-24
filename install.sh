#!/bin/sh
# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

export MUJOCO_ROOT=~/.mujoco/mujoco200

rm -rf build install
mkdir -p build
cd $_
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PWD}/../install \
    ..
make -j8 install
cd ..
