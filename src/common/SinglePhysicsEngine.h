// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PHYSICS_H
#define PHYSICS_H

#pragma once

#include "stdafx.h"
#include "mujoco.h"
#include "glfw3.h"
#include <Eigen/Dense>


#include <map>
#include <random>
#include <memory>
#include <vector>

#include "ModelInfo.h"
#include "GameState.h"
#include "type_config.h"

#define _CHANGE_MODEL_AFTER_LOAD 0
#define _SELECT_FROM_XML_BEFORE_LOAD 1

#if _CHANGE_MODEL_AFTER_LOAD & _SELECT_FROM_XML_BEFORE_LOAD
#error "It the end, there can be only one!!!"
#endif

using ConfigMap = std::map<std::string, std::string>;
class WallConfig;


// Physics engine interface
class PhysicsEngine
{
public:
    virtual ~PhysicsEngine(){};
    virtual GameState* createNewGameState()=0;
    virtual void setBallFric(Num fricSlide, Num fricSpin, Num fricRoll)=0;
    virtual void setWallFric(Num fricSlide, Num fricSpin, Num fricRoll)=0;
    virtual void setFricloss (Num fricLoss)=0;
    virtual void calcWarmStart(GameState* state, double* out_warmstart)=0;
    virtual bool computePhysics(GameState* state, float timeStep)=0;
    virtual bool computePhysicsForDerivative(GameState* state, float timeStep)=0;

    // properties
    virtual int jnt_qposadr(int ballIdx)=0;
    virtual int jnt_dofadr(int ballIdx)=0;
    virtual int getMjNv()=0;
    virtual int getMjNq()=0;
    virtual ModelInfo* getModelInfo()=0;

    // common processes
    virtual void setBoardRotation(GameState *state, Num xRot, Num yRot)=0;
    virtual bool checkBallCollisions(GameState* state, unsigned int, Num x, Num y)=0;
    virtual void resetState(GameState* state)=0;
};


///Physics engine subclasses Engine
class SinglePhysicsEngine : public PhysicsEngine
{
public:
	SinglePhysicsEngine(
        int num_balls, bool randomStart, Num* friction, Num friction_loss,
        Num tolerance, Num max_rotation, bool forceAngularVelZero,
        Num xAxisAngleRad, Num yAxisAngleRad, const std::string& xmlPath,
        const WallConfig* wallConfig, const std::string& mjkey);
	virtual ~SinglePhysicsEngine();
    static SinglePhysicsEngine* makeFromConfig(ConfigMap configMap);
    virtual GameState* createNewGameState();

    virtual void setBallFric(Num fricSlide, Num fricSpin, Num fricRoll);
    virtual void setWallFric(Num fricSlide, Num fricSpin, Num fricRoll);
    virtual void setFricloss (Num fricLoss);
    virtual void calcWarmStart(GameState* state, double* out_warmstart);
    virtual bool computePhysics(GameState* state, float timeStep);
    virtual bool computePhysicsForDerivative(GameState* state, float timeStep);

    // properties
    virtual int jnt_qposadr(int ballIdx) {
        int marbleId = mModelInfo->getMarbleJointId(m_mjcModel, 0);
        return m_mjcModel->jnt_qposadr[marbleId];
    }

    virtual int jnt_dofadr(int ballIdx){
        int marbleId = mModelInfo->getMarbleJointId(m_mjcModel, 0);
        return m_mjcModel->jnt_dofadr[marbleId];
    }

    virtual int getMjNv () {return m_mjcModel->nv;}
    virtual int getMjNq () {return m_mjcModel->nq;}
    virtual ModelInfo* getModelInfo() {return mModelInfo.get();}

    // common processes
    virtual void setBoardRotation(GameState *state, Num xRot, Num yRot);
    virtual bool checkBallCollisions(GameState* state, unsigned int, Num x, Num y);
    virtual void resetState(GameState* state);

protected:
    mjModel* m_mjcModel;

private:
    void setFric(int bodyId, Num fricSlide, Num fricSpin, Num fricRoll);
    void setAllFric(Num fricSlide, Num fricSpin, Num fricRoll);
    void setPhysicsParameters ();
    void createMJCStructures(const WallConfig* wallConfig);

    std::string mXMLfile;
    bool mRandomStart;

	unsigned int mNrBalls;
    Num mFriction[3];
    Num mFrictionLoss;
    Num mTorelance;
    Num maxRotRad;
    bool mForceAngularVelZero;
    Num mXAxisAngleRad, mYAxisAngleRad;

	//OpenGL rendering
	mjvPerturb m_pert;

	std::mt19937 mRandomEngine;
	std::unique_ptr<ModelInfo> mModelInfo;

	const int solverIterationForDerivative =  5;
    const int solverWarmupForDerivative =  5;
};


#endif
