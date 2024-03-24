// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Dynamics.h"
#include "ModelInfo.h"
#include "HelperFunctions.h"

#include "GameState.h"
#include "SinglePhysicsEngine.h"


/*
 * returns the cost ranged in [0, 1].
 * The cost is calculated from the distance along circle between the ball and the nearest hole.
 * */
Num ballCostFromTheta(Num boardRotX, Num boardRotY, Vector3 ballPos, const ModelInfo& modelInfo){
    Quaternion  quat = modelInfo.rotToQuat(boardRotX, boardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 proj_pos = R.transpose() * ballPos;
    Num theta = std::atan2(proj_pos.y(), proj_pos.x());
    int ringIdx = modelInfo.getRingIdx(ballPos);

    return ballCostFromTheta(ringIdx, theta, modelInfo);
}

Num ballCostFromTheta(int ring, Num theta, const ModelInfo& modelInfo) {
    if (ring == 0) {
        return 0.0;
    }

    if (theta < 0) {
        theta += 2*M_PI;
    }

    Num minAngle = 10.0f;
    const std::vector<Num>& curHoleRadius = modelInfo.listHoleRadius[ring - 1];

    for (auto curRad : curHoleRadius) {
        Num distRad = std::abs(curRad - theta);

        if (distRad < minAngle) {
            minAngle = distRad;
        }

        distRad = std::abs((2*M_PI + curRad) - theta);
        if (distRad < minAngle) {
            minAngle = distRad;
        }

    }

    // (0, 1)
    Num costRad = 0.0;
    if (ring == 1) {
        costRad = minAngle / (2*M_PI);
    } else {
        costRad = minAngle / (M_PI_2/2);
    }

    return costRad;
}



Num ballCostFromThetaAndDistance(Num boardRotX, Num boardRotY, Vector3 ballPos, const ModelInfo& modelInfo) {
    Quaternion boardRot = modelInfo.rotToQuat(boardRotX, boardRotY);

    const Matrix3 R = boardRot.toRotationMatrix();
    Vector3 pos = R.transpose() * ballPos - modelInfo.ballCenterPtOnBoard;
    // supress small error
    pos.z() = 0.0;

    int ring = modelInfo.getRingIdx(Vector2{pos.x(), pos.y()});

    if (ring == 0)  {
        return 0.0;
    }

    Num cost = ring;
    double dist = pos.norm();

    Num norm = modelInfo.innerRadius[ring] - modelInfo.innerRadius[ring-1];

    // (0, 1)
    Num dist_cost = (dist - modelInfo.innerRadius[ring-1]) / norm;

    // [-pi, pi]
    Num angle = std::atan2(pos.y(), pos.x());
    if (angle < 0) {
        angle = 2*M_PI + angle;
    }

    Num minAngle = 10.0f;
    Num costRad = ballCostFromTheta(ring, angle, modelInfo);

    // dist
    Num totalCost = cost + 0.5 * dist_cost + 0.5 * costRad;

    return totalCost ;
}


void Dynamics::setRingState(GameState* state, int ringIdx, Num theta, const ModelInfo& modelInfo) {
    Num r = modelInfo.centerRadius[ringIdx];

    mjData* mjState = state->getMjData();
    Num boardRotX = 0;
    Num boardRotY = 0;

    Num proj_x = r * std::cos(theta);
    Num proj_y = r * std::sin(theta);
    Num proj_z = -0.00365;
    Vector3 proj {proj_x, proj_y, proj_z};


    mjState->qpos[0] = proj_x;
    mjState->qpos[1] = proj_y;
    mjState->qpos[2] = proj_z;

    mjState->qpos[3] = 1;
    mjState->qpos[4] = 0;
    mjState->qpos[5] = 0;
    mjState->qpos[6] = 0;

    mjState->qvel[0] = 0;
    mjState->qvel[1] = 0;
    mjState->qvel[2] = 0;
    mjState->qvel[3] = 0;
    mjState->qvel[4] = 0;
    mjState->qvel[5] = 0;

    state->setBoardRotation(modelInfo, boardRotX, boardRotY);

}
