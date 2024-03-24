// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef BIMGAME_CYINTERFACE_H
#define BIMGAME_CYINTERFACE_H

#include <string>
#include <vector>

class PhysicsEngine;
class Environment;

class CyInterface {
public:
    CyInterface(std::string path_config);
    ~CyInterface();

    int dimState();
    std::vector<double> reset();
    void step(double rotX, double rotY, double* out_state, double* out_reward, int* out_done);

    std::vector<double> Lx();
    std::vector<double> Lxx();

    std::vector<double> LxTheta();
    std::vector<double> LxxTheta();
    std::vector<double> LxThetaOnTheta();
    std::vector<double> LxxThetaOnTheta();

    /*fx_out shape[dim_state, dim_state], fu_out shape[dim_state, 2]*/
    void FxFu(double xRot, double yRot, double* fx_out, double* fu_out);
    /*returns derivatives with theta representation in xyz representation*/
    void FxFuTheta(double xRot, double yRot, double* fx_out, double* fu_out);
    void FxFuThetaOnTheta(double xRot, double yRot, double* fx_out, double* fu_out);

    void setBallFric(double fricSlide, double fricSpin, double fricRoll);
    void setFricloss(double fricLoss);

    // return Theta representation
    std::vector<double> getThetaVector();
    std::vector<double> getXYZVector();

    void getState(double* out_state);
    double costState();
    void setState(const std::vector<double>& state);
    void setRingState(int ringIdx, double theta);
    void setRingIdx(int ringIdx);
    int getRingIdx();
private:
    PhysicsEngine* physics;
    Environment* environment;
    bool isThetaInternal;
    bool mHandleBallRotation;
};


#endif //BIMGAME_CYINTERFACE_H
