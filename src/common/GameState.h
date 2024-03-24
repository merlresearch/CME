// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef BIMGAME_GAMESTATE_H
#define BIMGAME_GAMESTATE_H


#include <vector>
#include <Eigen/Dense>
#include "mujoco.h"

#include "HelperFunctions.h"
#include "type_config.h"
#include "ModelInfo.h"

enum TerminalState {
    IS_RUNNING,
    IS_DONE,
    IS_INVALID
};

class ModelInfo;
class PhysicsEngine;


// This is the minimum snapshot of gamestate
// (mjData*) is too heavy to load and store.
struct GameStateCache {
    Num rotX, rotY;
    Quaternion boardQuat;
    double qpos[7]; // (x,y,z)
    double qvel[6]; // (dx,dy,dz)
    double warmstart[6]; //nv
};

class GameState{
public:
    mjData* getMjData()const {return mMjData;}
    Num BoardRotX() const {return mBoardRotX;}
    Num BoardRotY() const {return mBoardRotY;}

    GameState(const ModelInfo& modelInfo, mjData* data, Num boardRotX, Num boardRotY){
        mMjData = data;
        this->setBoardRotation(modelInfo, boardRotX, boardRotY);
    }

    Quaternion boardQuaternion(const ModelInfo& modelInfo) {
        return modelInfo.rotToQuat(mBoardRotX, mBoardRotY);
    }
    // This returns (0, 0, 0)
    Vector3 boardPos() {
        return Vector3{mMjData->mocap_pos[0], mMjData->mocap_pos[1], mMjData->mocap_pos[2]};
    }


    void setBoardRotation(const ModelInfo& modelInfo, Num rotX, Num rotY){
        mBoardRotX = rotX;
        mBoardRotY = rotY;
        Quaternion quat = modelInfo.rotToQuat(rotX, rotY);

        mMjData->mocap_quat[0] = quat.w();
        mMjData->mocap_quat[1] = quat.x();
        mMjData->mocap_quat[2] = quat.y();
        mMjData->mocap_quat[3] = quat.z();
    }

    Vector3 ballPos(int idx) const {
        // TODO refer mjModel index
        return Vector3{mMjData->qpos[idx*7], mMjData->qpos[idx*7+1], mMjData->qpos[idx*7+2]};
    }

    Vector3 ballVel(int idx) const {
        // TODO refer mjModel index
        return Vector3{mMjData->qvel[idx*6], mMjData->qvel[idx*6+1], mMjData->qvel[idx*6+2]};
    }


    void incrementBoardRotation(const ModelInfo& modelInfo, Num incX, Num incY){
        setBoardRotation(modelInfo, mBoardRotX+incX, mBoardRotY+incY);
    }

    void incrementBoardRotationWithBallRotate(const ModelInfo& modelInfo, Num incX, Num incY){
        Num preX = mBoardRotX;
        Num preY = mBoardRotY;
        Num nextX = mBoardRotX + incX;
        Num nextY = mBoardRotY + incY;

        Quaternion preQuat = modelInfo.rotToQuat(preX, preY);
        Quaternion nextQuat = modelInfo.rotToQuat(nextX, nextY);
        Quaternion  diff = nextQuat * preQuat.inverse();
        Vector3 ballPos = this->ballPos(0);
        Vector3 ballVel = this->ballVel(0);
        Vector3 rotatedBall = diff.toRotationMatrix() * ballPos;
        Vector3 rotatedVel = diff.toRotationMatrix() * ballVel;

        this->mMjData->qpos[0] = rotatedBall.x();
        this->mMjData->qpos[1] = rotatedBall.y();
        this->mMjData->qpos[2] = rotatedBall.z();
        this->mMjData->qvel[0] = rotatedVel.x();
        this->mMjData->qvel[1] = rotatedVel.y();
        this->mMjData->qvel[2] = rotatedVel.z();

        setBoardRotation(modelInfo, mBoardRotX+incX, mBoardRotY+incY);
    }


    GameState* copy(const ModelInfo& modelInfo, PhysicsEngine* physics);

    /*save state to mjData (mj_makeData and mj_copy is tooooooooooooooo slow.)*/
    void saveState(GameStateCache& cache) {
        // now quaternion and angular velocity are dropped
        mju_copy(cache.qpos, this->mMjData->qpos, 7);
        mju_copy(cache.qvel, this->mMjData->qvel, 6);
        mju_copy(cache.warmstart, this->mMjData->qacc_warmstart, 6);
        cache.rotX = this->mBoardRotX;
        cache.rotY = this->mBoardRotY;

        cache.boardQuat.w() = mMjData->mocap_quat[0];
        cache.boardQuat.x() = mMjData->mocap_quat[1];
        cache.boardQuat.y() = mMjData->mocap_quat[2];
        cache.boardQuat.z() = mMjData->mocap_quat[3];
    }
    void loadState(GameStateCache& cache) {
        mju_copy(this->mMjData->qpos, cache.qpos, 7);
        mju_copy(this->mMjData->qvel, cache.qvel, 6);
        mju_copy(this->mMjData->qacc_warmstart, cache.warmstart, 6);
        this->mBoardRotX = cache.rotX;
        this->mBoardRotY = cache.rotY;

        mMjData->mocap_quat[0] = cache.boardQuat.w();
        mMjData->mocap_quat[1] = cache.boardQuat.x();
        mMjData->mocap_quat[2] = cache.boardQuat.y();
        mMjData->mocap_quat[3] = cache.boardQuat.z();
    }


    ~GameState() {
        mj_deleteData(mMjData);
    }

    TerminalState getTerminateState(const ModelInfo& modelInfo);

private:
    // this member is owned by this class
    mjData* mMjData;

    // board rotation around inertial system axes
    Num mBoardRotX, mBoardRotY;
};


#endif //BIMGAME_GAMESTATE_H
