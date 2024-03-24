// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef MEDIATOR_H
#define MEDIATOR_H

#include <string>
#include <boost/bimap.hpp>
#include <boost/bimap/unordered_set_of.hpp>
#include <boost/bimap/vector_of.hpp>
#include <boost/program_options.hpp>
#include <boost/config.hpp>
#include <boost/optional.hpp>
#include <boost/none.hpp>
#include <boost/foreach.hpp>
#include <boost/assign/list_inserter.hpp>
#include <boost/thread.hpp>

#include <Eigen/Dense>

#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <memory>


#include "GameState.h"
#include "Dynamics.h"
#include "type_config.h"

class PhysicsEngine;

enum BoardRotation{
    NOP,
    RIGHT,
    LEFT,
    UP,
    DOWN
};

enum DynamicsType {
    XYZ_DYNAMICS,
    THETA_DYNAMICS
};


using ConfigMap = std::map<std::string, std::string>;

/*
 * Environment class
 * This has a state to represent the physical state of the simulation.
 * This also has a reward and terminal state for the last action.
 *
 * Although GameState is similar to this class, it represents physical state only.
 *
 * We shouldn't operate an instance of this class with multiple threads.
 * */
class Environment
{
public:
	Environment (PhysicsEngine* engine, XYZDynamics* xyzDynamics, ThetaDynamics* thetaDynamics, DynamicsType typeDynamics,
	        int numBalls, RewardMode rewardMode, bool handleBallRotation, Num rotateInc, float envFPS, float simFPS);
	~Environment ();

    static Environment* makeFromConfig(ConfigMap config_map, PhysicsEngine* physics);

	void doDiscreteAction(BoardRotation rot);
    void doContinuousAction(Num xRot, Num yRot);
	void reset();

	// access
	int terminalState () { return mTerminal; }
	GameState* getState() {return mGameState.get();}
	Num lastReward() {return mReward;}

	float mEnvFps;
	float mSimulationFps;

	RewardMode getRewardMode() {
	    return mRewardMode;
	};
	int NumBalls() {
	    return mNumBalls;
	}

    void simulateForDerivative(GameState* state, Num xRot, Num yRot, PhysicsEngine* physics, mjtNum* warmstart);
    bool HandleBallRotation() {return mHandleBallRotation;}
    Dynamics* getDynamics(){return mDefaultDynamics;}
    XYZDynamics* getXYZDynamics(){return mXYZDynamics.get();}
    ThetaDynamics* getThetaDynamics(){return mThetaDynamics.get();}

private:
	PhysicsEngine* mMJCPtr;
	std::unique_ptr<GameState> mGameState;
    Dynamics* mDefaultDynamics;
    std::unique_ptr<ThetaDynamics> mThetaDynamics;
    std::unique_ptr<XYZDynamics> mXYZDynamics;

    RewardMode mRewardMode;
    bool mHandleBallRotation;
    Num mRotateInc;
	int   mTerminal;
	int mNumBalls;
	Num mReward;
};

#endif
