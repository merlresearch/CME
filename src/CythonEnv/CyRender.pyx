# distutils: language = c++

# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os

import numpy as np

cimport numpy as np
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libcpp cimport bool
from RenderInterface cimport RenderInterface


cdef class CyRender:
    cdef RenderInterface* c_render
    cdef int height, width
    cdef unsigned char *c_frame

    def __cinit__(self, path_config):
        if not os.path.exists(path_config):
            raise ValueError("invalid path : ", path_config)

        self.c_render = new RenderInterface(bytes(path_config, encoding="ascii"))
        self.height = self.c_render.getHeight()
        self.width = self.c_render.getWidth()
        self.c_frame = <unsigned char*>PyMem_Malloc(self.height * self.width * 4)

    def size(self):
        # return the image size
        return (self.height, self.width)

    def set_ring_idx(self, ring_idx):
        self.c_render.setRingIdx(ring_idx)

    def render(self, np.ndarray[np.float64_t, ndim=1] state):
        self.c_render.render(state, self.c_frame)
        cdef unsigned char [:,:,:] arr = <unsigned char[:self.height, :self.width, :4]> self.c_frame
        return np.array(arr)

    def __dealloc__(self):
        PyMem_Free(self.c_frame)
        del self.c_render
