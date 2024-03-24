# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "CyInterface.h" nogil:
    cdef cppclass CyInterface:
        CyInterface(string path_config) except +
        vector[double] reset()
        void step(double rotX, double rotY, double* out_state, double* out_reward, int* out_done)
        double costState();
        void setState(vector[double] state)
        void setRingState(int ringIdx, double theta)
        int dimState()
        void getState(double* out_state)
        void setRingIdx(int ringIdx) except +
        int getRingIdx()
        vector[double] getThetaVector()
        vector[double] getXYZVector()
        void setBallFric(double fricSlide, double fricSpin, double fricRoll);
        void setFricloss(double fricLoss);

        vector[double] Lx()
        vector[double] Lxx()

        vector[double] LxTheta()
        vector[double] LxxTheta()
        vector[double] LxThetaOnTheta()
        vector[double] LxxThetaOnTheta()

        void FxFu(double xRot, double yRot, double* fx_out, double* fu_out)
        void FxFuTheta(double xRot, double yRot, double* fx_out, double* fu_out)
        void FxFuThetaOnTheta(double xRot, double yRot, double* fx_out, double* fu_out)
