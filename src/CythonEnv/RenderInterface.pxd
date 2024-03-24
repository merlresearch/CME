# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "RenderInterface.h":
    cdef cppclass RenderInterface:
        RenderInterface(string path_config)
        int getHeight()
        int getWidth()
        void setRingIdx(int ringIdx);
        void render(const vector[double]& state, unsigned char* image)
