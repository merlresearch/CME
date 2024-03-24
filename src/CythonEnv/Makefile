# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

.PHONY:clean build


BUILD_DIR=../../cmake-build-debug

build:
	python setup.py build_ext --build-lib ${BUILD_DIR} --user-lib ${BUILD_DIR}
	cd ${BUILD_DIR} && make install
clean:
	rm -rf ${BUILD_DIR}/Cy*.so
