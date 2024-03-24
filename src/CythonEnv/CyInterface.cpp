// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>
#include <chrono>

#include "CyInterface.h"
#include "../common/Environment.h"
#include "../common/SinglePhysicsEngine.h"
#include "../common/CompositePhysicsEngine.h"

using namespace std;

CyInterface::CyInterface(std::string path_config) : physics(NULL), environment(NULL) {
    std::map<std::string, std::string> configMap;
    HelperFunctions::readConfig(path_config.c_str(), configMap);

    bool useAdaptiveWall = false;
    if (configMap.find("AdaptiveWall") != configMap.end() && configMap["AdaptiveWall"] == "Yes") {
        useAdaptiveWall = true;
    }

    if (useAdaptiveWall) {
        physics = CompositePhysicsEngine::makeFromConfig(configMap);
    } else{
        physics = SinglePhysicsEngine::makeFromConfig(configMap);
    }

    if (configMap["Dynamics"] == "Theta") {
        isThetaInternal = true;
    } else {
        isThetaInternal = false;
    }
    if(configMap["HandleBallRotation"] == "Yes") {
        mHandleBallRotation = true;
    } else if (configMap["HandleBallRotation"] == "No") {
        mHandleBallRotation = false;
    } else {
        throw std::runtime_error("invalid HandleBallRotation");
    }

    environment = Environment::makeFromConfig(configMap, physics);
}

CyInterface::~CyInterface() {
    if (physics != NULL) {
        delete physics;
    }
    if (environment != NULL) {
        delete environment;
    }
}

void CyInterface::setRingIdx(int ringIdx) {
    ThetaDynamics* dynamics = dynamic_cast<ThetaDynamics*> (environment->getDynamics());

    if (dynamics != NULL){
        dynamics->setRingIdx(ringIdx);
    } else {
        throw std::runtime_error{"current dynamic doesn't support setRingIdx."};
    }
}

int CyInterface::getRingIdx() {
    //return environment->getDynamics()->getRingIdx(environment->getState());
    Vector3 ballPos = environment->getState()->ballPos(0);
    return physics->getModelInfo()->getRingIdx(ballPos);
}

/**Utility Functions***/
std::vector<double> CyInterface::getThetaVector() {
    vector<double> vec (ThetaDynamics::getThetaVecSize(), 0);
    GameState* state = environment->getState();
    ThetaDynamics::makeVec(state, vec.data(), physics->getModelInfo());

    return vec;
}

std::vector<double> CyInterface::getXYZVector() {
    int numBalls = 1;
    vector<double> vec (XYZDynamics::calcVecSizeWithBall(numBalls, mHandleBallRotation), 0);
    GameState* state = environment->getState();
    XYZDynamics::makeVec(state, vec.data(), numBalls, mHandleBallRotation);

    return vec;
}


void CyInterface::setState(const std::vector<double>& state) {
    GameState* gameState = environment->getState();
    environment->getDynamics()->setVec(gameState, state.data());
}

void CyInterface::setRingState(int ringIdx, double theta) {
    GameState* gameState = environment->getState();
    environment->getDynamics()->setRingState(gameState, ringIdx, theta, *physics->getModelInfo());
}


void CyInterface::setBallFric(double fricSlide, double fricSpin, double fricRoll){
    physics->setBallFric(fricSlide, fricSpin, fricRoll);
}

void CyInterface::setFricloss(double fricLoss) {
    physics->setFricloss(fricLoss);
}


/*
 * - the board rotation X
 * - the board rotation Y
 * - theta on the board surface
 * - the velocity of theta on the board surface
 * TODO implement under common
 * */
void CyInterface::getState(double* out_state) {
    GameState* state = environment->getState();
    environment->getDynamics()->toVec(state, out_state);
}

std::vector<double> CyInterface::reset() {
    int dimVec = environment->getDynamics()->getVecSize();
    std::vector<double> out(dimVec, 0);

    environment->reset();
    GameState* lastState = environment->getState();
    environment->getDynamics()->toVec(lastState, out.data());

    return out;
}

int CyInterface::dimState() {
    return environment->getDynamics()->getVecSize();
}


void CyInterface::step(double rotX, double rotY, double* out_state, double* out_reward, int* out_done) {
    using namespace std;
    typedef std::chrono::high_resolution_clock Clock;

    auto t1 = Clock::now();
    environment->doContinuousAction(rotX, rotY);
    auto t2 = Clock::now();
    // cout << "continuous :" << chrono::duration_cast<chrono::microseconds>(t2-t1).count() << " us" << endl;

    GameState* lastState = environment->getState();
    //lastState->ToVectorFast(out_state, this->environment->NumBalls(), this->environment->HandleBallRotation());
    environment->getDynamics()->toVec(lastState, out_state);
    *out_reward = environment->lastReward();
    *out_done = environment->terminalState();
}

double CyInterface::costState() {
    GameState* lastState = environment->getState();
    return environment->getDynamics()->cost(lastState);
}

std::vector<double> CyInterface::Lx() {
    using namespace std;
    vector<double> lx (environment->getDynamics()->getVecSize(), 0);

    GameState* lastState = environment->getState();
    environment->getDynamics()->Lx(lastState, lx.data());
    return lx;
}

std::vector<double> CyInterface::Lxx(){
    int dimSize = environment->getDynamics()->getVecSize();
    vector<double> lxx (dimSize*dimSize, 0);

    GameState* lastState = environment->getState();

    environment->getDynamics()->Lxx(lastState, lxx.data());
    return lxx;
}


/*fx_out shape[dim_state, dim_state], fu_out shape[dim_state, 2]*/
void CyInterface::FxFu(double xRot, double yRot, double* fx_out, double* fu_out) {
    environment->getDynamics()->fx_fu(environment, physics, xRot, yRot, fx_out, fu_out);
}


void CyInterface::FxFuTheta(double xRot, double yRot, double* fx_out, double* fu_out) {
    if (isThetaInternal) {
        environment->getDynamics()->fx_fu(environment, physics, xRot, yRot, fx_out, fu_out);
    }
    else {
        XYZDynamics* dynamics = environment->getXYZDynamics();
        dynamics->fx_fu_theta(environment, physics, xRot, yRot, fx_out, fu_out);
    }
}

std::vector<double> CyInterface::LxTheta() {
    using namespace std;

    if (isThetaInternal) {
        return Lx();
    }
    else {
        XYZDynamics* dynamics = environment->getXYZDynamics();
        vector<double> lx (4, 0);
        GameState* lastState = environment->getState();
        dynamics->Lx_theta(lastState, lx.data());
        return lx;
    }
}

/*
 * Return gradients on "Theta" representation
 * */
std::vector<double> CyInterface::LxThetaOnTheta() {
    using namespace std;

    ThetaDynamics* dynamics = environment->getThetaDynamics();
    vector<double> lx (dynamics->getVecSize(), 0);

    GameState* lastState = environment->getState();
    dynamics->Lx(lastState, lx.data());
    return lx;
}


std::vector<double> CyInterface::LxxTheta() {
    if (isThetaInternal) {
         return Lxx();
    }
    else {
        XYZDynamics* dynamics = dynamic_cast<XYZDynamics*> (environment->getDynamics());

        if (dynamics != NULL){
            int dimSize = 4;
            vector<double> lxx (dimSize*dimSize, 0);
            GameState* lastState = environment->getState();

            dynamics->Lxx_theta(lastState, lxx.data());
            return lxx;
        } else {
            throw std::runtime_error{"current dynamic doesn't support setRingIdx."};
        }
    }

}

std::vector<double> CyInterface::LxxThetaOnTheta() {
    ThetaDynamics* dynamics = environment->getThetaDynamics();
    int dimSize = dynamics->getVecSize();
    vector<double> lxx (dimSize*dimSize, 0);

    GameState* lastState = environment->getState();

    dynamics->Lxx(lastState, lxx.data());
    return lxx;
}


void CyInterface::FxFuThetaOnTheta(double xRot, double yRot, double* fx_out, double* fu_out) {
    ThetaDynamics* dynamics = environment->getThetaDynamics();
    dynamics->fx_fu(environment, physics, xRot, yRot, fx_out, fu_out);

}
