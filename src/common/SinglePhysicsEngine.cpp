// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "stdafx.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include "mujoco.h"
#include "SinglePhysicsEngine.h"
#include "WallRemover.h"
#include "HelperFunctions.h"
#include "GameState.h"
#include <chrono>
#include <boost/algorithm/string/split.hpp>


const double MARBLE_RADIUS = 0.00635;
const double MARBLE_DIAMETER = 2.0 * MARBLE_RADIUS;


SinglePhysicsEngine::SinglePhysicsEngine(
        int num_balls, bool randomStart, Num* friction, Num friction_loss,
        Num tolerance, Num max_rotation, bool forceAngularVelZero,
        Num xAxisAngleRad, Num yAxisAngleRad, const std::string& xmlPath,
        const WallConfig* wallConfig, const std::string& mjkey)
	: m_mjcModel(0), mNrBalls(0)
{
	mRandomStart=randomStart;
    std::uint32_t seed = std::random_device()();
    mRandomEngine = std::mt19937{seed};

	mNrBalls = num_balls;
	mXMLfile = xmlPath;

    mXAxisAngleRad = xAxisAngleRad;
    mYAxisAngleRad = yAxisAngleRad;

    mFriction[0] = friction[0];
    mFriction[1] = friction[1];
    mFriction[2] = friction[2];
    mFrictionLoss = friction_loss;
    mTorelance = tolerance;
    mForceAngularVelZero = forceAngularVelZero;
    maxRotRad = max_rotation;

    if( mjVERSION_HEADER!=mj_version() ) {
        mju_error("Headers and library have different versions");
    }

    // activate MuJoCo license
    int result = mj_activate(mjkey.c_str());
    if (result!=1) {
        std::cerr << "Could not activate MuJoCo key, make sure path is known." << std::endl;
        throw std::logic_error("invalid MuJoCo key path");
    }

    createMJCStructures(wallConfig);
}


SinglePhysicsEngine* SinglePhysicsEngine::makeFromConfig(ConfigMap configMap)
{
    std::string::size_type sz;            // alias of size_t
    //std::cout << " In mediator ... " << std::endl;
    Num friction_loss = std::stod(configMap["FrictionLoss"]);

    std::vector<std::string> results;
    boost::algorithm::split(results, configMap["Friction"], [](char c){return c == ',';});

    Num friction[3];
    for (unsigned int i=0; i < 3; ++i)
        friction[i] = std::stod(results[i]);

    Num max_rotation = std::stod(configMap["Max_Rotation"],&sz);

    std::string xml_file = HelperFunctions::STRExtendWithModelDirpath(configMap["MujocoModel"]);
    std::string mjkey = configMap["MujocoKey"];

    std::string mazeWallsFile = "";
    WallConfig wallConfig;

    if (configMap.find("MazeWallsFile") != configMap.end())
    {
        mazeWallsFile = configMap["MazeWallsFile"];
        wallConfig = WallConfig::makeWallConfig(mazeWallsFile);
    }

    int numBalls = std::stoi(configMap["Nr_Of_Balls"]);

    bool handleBallRotation = false;
    if(configMap["HandleBallRotation"] == "Yes") {
        handleBallRotation = true;
    } else if (configMap["HandleBallRotation"] == "No") {
        handleBallRotation = false;
    } else {
        throw std::runtime_error("invalid HandleBallRotation");
    }

    Num tolerance = 0.00000001;
    if (configMap["FixSolverIter"] == "Yes") {
        tolerance = 0;
    }

    Num xAxisAngleRad = degToRad(std::stod(configMap["XAxisAngleDeg"], &sz));
    Num yAxisAngleRad = degToRad(std::stod(configMap["YAxisAngleDeg"], &sz));

    bool forceAngularVelZero = !handleBallRotation;
    bool randomStart = false;
    if (configMap["Random_Start"] == "Yes") {
        randomStart = true;
    } else {
        randomStart = false;
    }

    SinglePhysicsEngine* MJCEngine = new SinglePhysicsEngine(
        numBalls, randomStart, friction, friction_loss, tolerance,
        max_rotation, forceAngularVelZero, xAxisAngleRad, yAxisAngleRad,
        xml_file, &wallConfig, mjkey);

    return MJCEngine;
}


SinglePhysicsEngine::~SinglePhysicsEngine()
{
}


GameState* SinglePhysicsEngine::createNewGameState(){
    mjData* newData = mj_makeData(m_mjcModel);
    GameState* newState = new GameState{*this->mModelInfo.get(), newData, 0, 0};
    resetState(newState);

    return newState;
}

/*
 * set states of marbles directry to initialize positions randomly.
 * To set positions correctly, see http://www.mujoco.org/book/index.html#Clarifications (floating objects section)
 *
 * */
void SinglePhysicsEngine::resetState(GameState* state)
{
    using namespace std;

    mjData* mjcData = state->getMjData();
    mj_resetData(m_mjcModel, mjcData);

	for (unsigned int i = 0; i < mNrBalls; ++i)
	{
		Num x, y;
		if (mRandomStart)
		{
			bool picked_valid_loc = false;
			while (!picked_valid_loc)
			{
			    Vector2 pos = mModelInfo->sampleRandomBallPos(mModelInfo->getOuterRingIdx(), mRandomEngine);
			    x = pos[0];
			    y = pos[1];

				// check to make sure we are not colliding with another ball
				if (i > 0)
					picked_valid_loc = !(checkBallCollisions (state, i,x,y));
				else
					picked_valid_loc = true;
			}
		}
		else
		{
			//pickStartLoc (i, x, y);
			throw std::logic_error("not implemented error.");
		}

		// Now update the ball position in MuJoCo data
		int marbleId = mModelInfo->getMarbleJointId(m_mjcModel, i);
        int jointId = m_mjcModel->jnt_qposadr[marbleId];
        mjcData->qpos[jointId] = x;
        mjcData->qpos[jointId+1] = y;
	}

	state->setBoardRotation(*mModelInfo, 0, 0);

    mj_forward(m_mjcModel, mjcData);
}

bool SinglePhysicsEngine::checkBallCollisions (GameState* state, unsigned int ball_idx, Num x, Num y)
{
	bool result = false;
	mjData* mjcData = state->getMjData();

	for (unsigned int i = 0; i < ball_idx; ++i)
	{
        int marbleId = mModelInfo->getMarbleJointId(m_mjcModel, i);
        int jointId = m_mjcModel->jnt_qposadr[marbleId];
		Num d = x - mjcData->qpos[jointId];
		Num dist = d * d;
		d = y - mjcData->qpos[jointId+1];
		dist += (d * d);

		if (sqrt(dist) <= (MARBLE_DIAMETER + 0.003))
		{
			result = true;
			break;
		}
	}

	return result;
}


//Compute physics using mj_step
bool SinglePhysicsEngine::computePhysics(GameState* state, float timesteps)
{
    // TODO use mutex
    if(timesteps != m_mjcModel->opt.timestep) {
        m_mjcModel->opt.timestep = timesteps;
    }
	mjData* mjcData = state->getMjData();

    if (mForceAngularVelZero) {
        // This assumes that qvel is composed of ball velocities only.
        for(int i = 0; i < this->mNrBalls; i++) {
            int jointId = this->mModelInfo->getMarbleJointId(m_mjcModel, i);
            int qposId = this->m_mjcModel->jnt_qposadr[jointId];
            int qvelId = this->m_mjcModel->jnt_dofadr[jointId];
            mjcData->qpos[qposId+3] = 1;
            mjcData->qpos[qposId+4] = 0;
            mjcData->qpos[qposId+5] = 0;
            mjcData->qpos[qposId+6] = 0;
            mjcData->qvel[qvelId+3] = 0;
            mjcData->qvel[qvelId+4] = 0;
            mjcData->qvel[qvelId+5] = 0;
        }
    }

    double start_time = mjcData->time;

    // run mj_step and count
    mj_step(m_mjcModel, mjcData);

    //std::cout << "nefc: " << mjcData->nefc << std::endl;

    double end_time = mjcData->time;

    if (end_time < start_time) {
        std::cerr << "============detect auto reset==============-" << std::endl;
        return false;
    }

    return true;
}

bool SinglePhysicsEngine::computePhysicsForDerivative(GameState* state, float timeStep) {
    int save_iterations = m_mjcModel->opt.iterations;
    mjtNum save_tolerance = m_mjcModel->opt.tolerance;
    m_mjcModel->opt.iterations = solverIterationForDerivative;
    m_mjcModel->opt.tolerance = 0;

    this->computePhysics(state, timeStep);

    m_mjcModel->opt.iterations = save_iterations;
    m_mjcModel->opt.tolerance = save_tolerance;
}

void SinglePhysicsEngine::calcWarmStart(GameState* state, double* out_warmstart) {
    int nwarmup = solverWarmupForDerivative;

    mjData* data = state->getMjData();
    mj_forward(m_mjcModel, data);
    for (int i = 0; i < nwarmup; i++) {
        mj_forwardSkip(m_mjcModel, data, mjSTAGE_VEL, 1);
    }
    mju_copy(out_warmstart, data->qacc_warmstart, getMjNv());

}



//Create model and data mujoco structures form XML file
void SinglePhysicsEngine::createMJCStructures(const WallConfig* wallConfig)
{
	// make sure one source is given
	const char* filename = mXMLfile.c_str();
    if( !filename)
        return;

    char error[1000] = "Could not load binary file";
    mjModel* mnew;
    mnew = loadXMLWithWallLimitation(filename, wallConfig);

    if( !mnew )
    {
        throw std::runtime_error("failed to load mujoco XML.");
    }

    m_mjcModel = mnew;
    mModelInfo.reset(new ModelInfo{mXAxisAngleRad, mYAxisAngleRad});
	setPhysicsParameters ();

	m_pert.select = 0;
	m_pert.active = 0;
	mjv_defaultPerturb (&m_pert);
}

void SinglePhysicsEngine::setPhysicsParameters ()
{
    m_mjcModel->opt.o_solref[0]	= 0.002;

    for (unsigned int i = 0; i < m_mjcModel->nv; ++i)
    {
        m_mjcModel->dof_frictionloss[i] = mFrictionLoss;
    }

    int c_body_id = -1;
    int c_geom_type = -1;
    for (unsigned int i = 0; i < m_mjcModel->ngeom; ++i)
    {
        if (c_body_id != m_mjcModel->geom_bodyid[i])
        {
            c_body_id = m_mjcModel->geom_bodyid[i];
            c_geom_type = -1;
        }
        else
        {
            if (c_geom_type != m_mjcModel->geom_type[i])
            {
                c_geom_type = m_mjcModel->geom_type[i];
            }
        }

        m_mjcModel->geom_friction[i*3 + 0] = mFriction[0];
        m_mjcModel->geom_friction[i*3 + 1] = mFriction[1];
        m_mjcModel->geom_friction[i*3 + 2] = mFriction[2];
    }
    m_mjcModel->opt.tolerance = mTorelance;
}

void SinglePhysicsEngine::setBoardRotation(GameState *state, Num xRot, Num yRot)
{
    if (xRot > maxRotRad) {
        xRot = maxRotRad;
    } else if(xRot < -maxRotRad) {
        xRot = -maxRotRad;
    }
    if (yRot > maxRotRad) {
        yRot = maxRotRad;
    } else if(yRot < -maxRotRad) {
        yRot = -maxRotRad;
    }

    state->setBoardRotation(*mModelInfo, xRot, yRot);
}


void SinglePhysicsEngine::setBallFric(Num fricSlide, Num fricSpin, Num fricRoll) {
    // TODO handle multiple balls
    // TODO set ball firc only
    //int ballBodyId = this->mModelInfo->getMarbleBodyId(m_mjcModel, 0);
    //setFric(ballBodyId, fricSlide, fricSpin, fricRoll);
    setAllFric(fricSpin, fricSpin, fricRoll);
}

void SinglePhysicsEngine::setWallFric(Num fricSlide, Num fricSpin, Num fricRoll) {
    /*
     * 簡略化のために全体で同じfrictionを共有する
     * */
    throw std::runtime_error("don't call this method.");
    //int bodyId = this->mModelInfo->getWallBodyId(m_mjcModel);
    //setFric(bodyId, fricSlide, fricSpin, fricRoll);
}

void SinglePhysicsEngine::setAllFric(Num fricSlide, Num fricSpin, Num fricRoll) {

    for (unsigned int i = 0; i < m_mjcModel->ngeom; ++i)
    {
        m_mjcModel->geom_friction[i*3 + 0] = fricSlide;
        m_mjcModel->geom_friction[i*3 + 1] = fricSpin;
        m_mjcModel->geom_friction[i*3 + 2] = fricRoll;
    }

}

void SinglePhysicsEngine::setFricloss (Num fricLoss)
{
    for (unsigned int i = 0; i < m_mjcModel->nv; ++i)
    {
        m_mjcModel->dof_frictionloss[i] = fricLoss;
    }
}

void SinglePhysicsEngine::setFric(int bodyId, Num fricSlide, Num fricSpin, Num fricRoll) {
    int geomCount = 0;

    for (unsigned int i = 0; i < m_mjcModel->ngeom; ++i)
    {
        //m_mjcModel->geom_condim[i] = 1;
        if (bodyId != m_mjcModel->geom_bodyid[i])
        {
            continue;
        }
        geomCount += 1;
        m_mjcModel->geom_friction[i*3 + 0] = fricSlide;
        m_mjcModel->geom_friction[i*3 + 1] = fricSpin;
        m_mjcModel->geom_friction[i*3 + 2] = fricRoll;
    }

}
