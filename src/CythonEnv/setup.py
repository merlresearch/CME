# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os
import sys
from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

if "--user-lib" in sys.argv:
    idx_lib = sys.argv.index("--user-lib")
    DIR_BUILD = sys.argv[idx_lib + 1]
    path_lib_common = os.path.join(DIR_BUILD, "libcommon_ops.so")

    if not os.path.exists(path_lib_common):
        raise ValueError("{} is not found in directory {}".format(path_lib_common, DIR_BUILD))

    del sys.argv[idx_lib + 1]
    sys.argv.remove("--user-lib")
else:
    raise ValueError("you must specify --user-lib where there are libcommon_ops and librender_ops")

compile_args = ["-std=c++11", "-pthread", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1"]
srcs = ""

assert os.path.isdir(
    DIR_BUILD
), "Cannot find {}. You should set MAZE_SIMULATOR_ROOT. See README.md for details.".format(DIR_BUILD)

DIR_INCLUDE = os.path.join(os.path.dirname(__file__), "..")

ext_env = Extension(
    "CyMaze",
    ["CyMaze.pyx"],
    extra_compile_args=compile_args,
    include_dirs=[DIR_INCLUDE, numpy.get_include()],
    library_dirs=[DIR_BUILD],
    libraries=["common_ops"],
)

ext_render = Extension(
    "CyRender",
    ["CyRender.pyx"],
    extra_compile_args=compile_args,
    include_dirs=[DIR_INCLUDE, numpy.get_include()],
    library_dirs=[DIR_BUILD],
    libraries=["render_ops"],
)

setup(ext_modules=cythonize([ext_env, ext_render], compiler_directives={"language_level": "3"}))
