// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

#include "GameState.h"
#include "Dynamics.h"
#include "ModelInfo.h"
#include "SinglePhysicsEngine.h"

TerminalState GameState::getTerminateState(const ModelInfo& modelInfo) {
    // TODO handle multiple balls
    Quaternion  quat = modelInfo.rotToQuat(mBoardRotX, mBoardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 pos {this->mMjData->qpos[0], this->mMjData->qpos[1], this->mMjData->qpos[2]};
    Vector3 proj = R.transpose() * pos;

    Num z = proj.z() - modelInfo.ballCenterPtOnBoard.z();

    if (abs(z) > modelInfo.marbleRadius * 2)
    {
        //std::cout << ballPos_proj.x() << ", " << ballPos_proj.y() << ", " << ballPos_proj.z() << std::endl;
        std::cout << "Ball is off the maze plane!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "Z axis" << z << std::endl;
        return TerminalState::IS_INVALID;
    }

    int ringIdx = modelInfo.getRingIdx(Vector2{pos.x(),pos.y()});
    if (ringIdx == 0)
        return TerminalState ::IS_DONE;

    return TerminalState ::IS_RUNNING;
}


/*
 * DON'T use this method, if you concern about performance.
 * mj_makeData is very slow function.
 * */
GameState* GameState::copy(const ModelInfo& modelInfo, PhysicsEngine* physics) {
    GameState* state = physics->createNewGameState();
    mju_copy(state->getMjData()->qpos, this->mMjData->qpos, physics->getMjNq());
    mju_copy(state->getMjData()->qvel, this->mMjData->qvel, physics->getMjNv());
    return state;
}
