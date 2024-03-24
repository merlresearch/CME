// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "stdafx.h"
#include "Environment.h"
#include "SinglePhysicsEngine.h"
#include "Dynamics.h"
#include "HelperFunctions.h"
#include <random>
#include <chrono>

using Eigen::MatrixXd;


Environment::Environment (PhysicsEngine* engine, XYZDynamics* xyzDynamics, ThetaDynamics* thetaDynamics, DynamicsType typeDynamics,
        int numBalls, RewardMode rewardMode, bool handleBallRotation, Num rotateInc, float envFPS, float simFPS)
{
    mMJCPtr = engine;

    mNumBalls = numBalls;
    mRotateInc = rotateInc;
    mEnvFps = envFPS;
    mSimulationFps = simFPS;
    mRewardMode = rewardMode;
    mHandleBallRotation = handleBallRotation;
    mTerminal = 0;
    mReward = 0;
    mGameState.reset(mMJCPtr->createNewGameState());

    mXYZDynamics.reset(xyzDynamics);
    mThetaDynamics.reset(thetaDynamics);

    if (typeDynamics == DynamicsType::XYZ_DYNAMICS) {
        mDefaultDynamics = xyzDynamics;
    } else if (typeDynamics == DynamicsType::THETA_DYNAMICS){
        mDefaultDynamics = thetaDynamics;
    } else{
        throw std::runtime_error("invalid dynamics type");
    }

}

Environment* Environment::makeFromConfig(ConfigMap config_map, PhysicsEngine* physics){
    std::string::size_type sz;            // alias of size_t

    float envFps = std::stof(config_map["fps"], &sz);
    float simulationFps = std::stof(config_map["simulation_fps"], &sz);
    Num rotateInc = std::stof(config_map["Rotation_Increment"], &sz);
    RewardMode rewardMode;

    if (config_map["Reward"] == "RingSparse") {
        rewardMode = RewardMode::RING_SPARSE;
        throw std::runtime_error("RingSparse is not implemented.");
    } else if (config_map["Reward"] == "DistanceDense") {
        rewardMode = RewardMode::DISTANCE_DENSE;
    } else if (config_map["Reward"] == "ThetaDistance") {
        rewardMode = RewardMode::THETA_DISTANCE;
    } else {
        std::cerr << "invalid Reward Mode" << std::endl;
        throw std::runtime_error("invalid reward mode.");
    }

    bool handleBallRotation = false;
    if(config_map["HandleBallRotation"] == "Yes") {
        handleBallRotation = true;
    } else if (config_map["HandleBallRotation"] == "No") {
        handleBallRotation = false;
    } else {
        throw std::runtime_error("invalid HandleBallRotation");
    }

    int numBalls = std::stoi(config_map["Nr_Of_Balls"]);

    ThetaDynamics* thetaDynamics = new ThetaDynamics(physics->getModelInfo(), 4, handleBallRotation, rewardMode);
    XYZDynamics* xyzDynamics = new XYZDynamics(physics->getModelInfo(), numBalls, handleBallRotation, rewardMode);
    DynamicsType  typeDynamics;

    if (config_map["Dynamics"] == "XYZ") {
        typeDynamics = DynamicsType ::XYZ_DYNAMICS;
    } else if (config_map["Dynamics"] == "Theta") {
        typeDynamics = DynamicsType ::THETA_DYNAMICS;
    } else {
        throw std::runtime_error("invalid Dynamics");
    }

    Environment* mediator = new Environment(physics, xyzDynamics, thetaDynamics, typeDynamics,
            numBalls, rewardMode, handleBallRotation, rotateInc, envFps, simulationFps);
    return mediator;
}


Environment::~Environment ()
{
}

void Environment::reset() {
    mMJCPtr->resetState (mGameState.get());
}

void Environment::doDiscreteAction(BoardRotation rot) {

    Num incRad = 2*M_PI * (mRotateInc / 360);
    // rotate centered xaxis
    Num xRot = 0.0f;
    // rotate centered yaxis
    Num yRot = 0.0f;

    //Based on the number, decide the axis of rotation
    if (rot == RIGHT)
    {
        yRot = incRad;
    }
    else if (rot == LEFT)
    {
        yRot = -incRad;
    }
    else if (rot == UP)
    {
        xRot = -incRad;
    }
    else if (rot == DOWN)
    {
        xRot = incRad;
    }

    doContinuousAction(xRot, yRot);
}

void Environment::doContinuousAction(Num xRot, Num yRot)
{
    using namespace std;
    typedef std::chrono::high_resolution_clock Clock;

    mTerminal = 0;
    mReward = 0;

    ModelInfo* modelInfo = mMJCPtr->getModelInfo();
    float next_frame_time = 1.f / mEnvFps;
    float compute_time = 1.f / mSimulationFps;
    int numRotation = std::round(next_frame_time / compute_time);
    // we rotate mRotateInc in one environment frame.
    Num unitRotX = xRot / numRotation;
    Num unitRotY = yRot / numRotation;
    compute_time = next_frame_time / numRotation;

    Num currentRotX = this->mGameState->BoardRotX();
    Num currentRotY = this->mGameState->BoardRotY();

    // for debug
    float total_time=0.0f;

    auto t1 = Clock::now();

    Num initialCost = mDefaultDynamics->cost(mGameState.get());

    auto t2 = Clock::now();
    auto d1 = chrono::duration_cast<chrono::microseconds>(t2-t1).count();

    bool success = true;

    for (int i = 0; i < numRotation; i++) {
        float cur_compute_time = compute_time;
        total_time += cur_compute_time;

        currentRotX += unitRotX;
        currentRotY += unitRotY;
        mMJCPtr->setBoardRotation(mGameState.get(), currentRotX, currentRotY);

        //mMJCPtr->incrementBoardRotationWithLimit(mGameState.get(), unitRotX, unitRotY);
        success = mMJCPtr->computePhysics(mGameState.get(), cur_compute_time);

        if(!success) {
            break;
        }
    }
    auto t3 = Clock::now();
    auto d2 = chrono::duration_cast<chrono::microseconds>(t3-t2).count();

    Num finalCost = mDefaultDynamics->cost(mGameState.get());
    mReward = initialCost - finalCost;

    TerminalState termState = mGameState->getTerminateState(*modelInfo);

    // compute is diverged.
    if (!success || termState == TerminalState::IS_INVALID) {
        this->mTerminal = 2;
    } else{
        this->mTerminal = termState;
    }

    auto t4 = Clock::now();
    auto d3 = chrono::duration_cast<chrono::microseconds>(t4-t3).count();
    //cout << "make first state view :" << d1 << " us" << endl;
    //cout << "step :" << d2 << " us" << endl;
    //cout << "calc reward :" << d3 << " us" << endl;
}

void Environment::simulateForDerivative(GameState* state, Num xRot, Num yRot, PhysicsEngine* physics, mjtNum* warmstart) {
    // save solver options
    float next_frame_time = 1.f / mEnvFps;
    float compute_time = 1.f / mSimulationFps;
    int numRotation = std::round(next_frame_time / compute_time);
    // we rotate mRotateInc in one environment frame.
    Num unitRotX = xRot / numRotation;
    Num unitRotY = yRot / numRotation;
    compute_time = next_frame_time / numRotation;

    Num currentRotX = state->BoardRotX();
    Num currentRotY = state->BoardRotY();

    // iterationによるぶれを防止する
    int niter = 5;
    mju_copy(state->getMjData()->qacc_warmstart, warmstart, physics->getMjNv());

    for (int i = 0; i < numRotation; i++) {
        currentRotX += unitRotX;
        currentRotY += unitRotY;

        //physics->setBoardRotation(state, currentRotX, currentRotY);
        // do unlimited rotation
        state->setBoardRotation(*physics->getModelInfo(), currentRotX, currentRotY);
        physics->computePhysicsForDerivative(state, compute_time);
    }

}
