// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Dynamics.h"

#include <mujoco.h>

#include "GameState.h"
#include "ModelInfo.h"
#include "Environment.h"
#include "SinglePhysicsEngine.h"
#include "HelperFunctions.h"


ThetaDynamics::ThetaDynamics(const ModelInfo* info, int ringIdx, bool handleBallRotation, RewardMode rewardMode) : modelInfo(info){
    this->ringIdx = ringIdx;
    this->handleBallRotation = handleBallRotation;

    switch(rewardMode) {
        case RewardMode ::RING_SPARSE:
            throw std::runtime_error("sparse reward is not implemented.");
        default:
            this->rewardMode = rewardMode;
    }
}

ThetaDynamics::~ThetaDynamics(){}


int ThetaDynamics::getVecSize() {
    return getThetaVecSize();
}

int ThetaDynamics::getThetaVecSize() {
    return 4;
}


/*
 *
convert to the feature vector which is composed of
    - board_x
    - board_y
    - ball_theta
    - ball_theta_vel

 TODO handle multple balls
 * */
void ThetaDynamics::toVec(GameState* state, Num* out){
    makeVec(state, out, this->modelInfo);
}

/*
 * stateless toVec for CyInterface
 * */
void ThetaDynamics::makeVec(GameState* state, Num* out, const ModelInfo* modelInfo){
    Num boardRotX = state->BoardRotX();
    Num boardRotY = state->BoardRotY();
    mjData* mjState = state->getMjData();

    Quaternion  quat = modelInfo->rotToQuat(boardRotX, boardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 pos {mjState->qpos[0], mjState->qpos[1], mjState->qpos[2]};
    Vector3 vel {mjState->qvel[0], mjState->qvel[1], mjState->qvel[2]};
    Vector3 proj_pos = R.transpose() * pos;
    Vector3 proj_vel = R.transpose() * vel;


    // [-pi, pi]
    Num angle = std::atan2(proj_pos.y(), proj_pos.x());
    Num r = std::sqrt(proj_pos.x()*proj_pos.x() + proj_pos.y()*proj_pos.y());

    Num angleVel = (-proj_vel.x() * std::sin(angle) + proj_vel.y() * std::cos(angle)) / r;
    Num energyBasedVel = std::sqrt(proj_vel.x()*proj_vel.x() + proj_vel.y()*proj_vel.y()) / r;

    if (angleVel < 0) {
        energyBasedVel = -energyBasedVel;
    }

    out[0] = boardRotX;
    out[1] = boardRotY;
    out[2] = angle;
    //out[3] = energyBasedVel;
    out[3] = angleVel;
}

/*
 *
 * param rotateDistance : ball moving distance
 * param ballTheta : angle of ball position (not quaternion)
 * */

void calculateBallAngularVelocity(Num res[3], Num rotateDistance, Num ballTheta) {
    Num radius = 0.00635;
    Num angularVel = -rotateDistance / radius;

    //Quaternion base(1.0, 0.0, 0.0, 0.0);
    // compute quaternion velocity from velocity
    //std::cout << angularVel << std::endl;

    Num dt =  1;
    Quaternion rotQuat (cos(angularVel/2), cos(ballTheta)*sin(angularVel/2), sin(ballTheta)*sin(angularVel/2), 0);
    double mjquat[4];
    mjquat[0] = rotQuat.w();
    mjquat[1] = rotQuat.x();
    mjquat[2] = rotQuat.y();
    mjquat[3] = rotQuat.z();

    mju_quat2Vel(res, mjquat, dt);

}

/*
 *
convert to the feature vector which is composed of
    - board_x
    - board_y
    - ball_theta
    - ball_theta_vel

 the distance from the ball to the center is passed by an argument.
 TODO handle multple balls
 * */
void ThetaDynamics::setVec(GameState* state, const Num* in){
    Num r = modelInfo->centerRadius[ringIdx];

    mjData* mjState = state->getMjData();
    Num boardRotX = in[0];
    Num boardRotY = in[1];
    Num theta = in[2];
    Num thetaVel = in[3];

    vec2state(in, ringIdx, mjState->qpos, mjState->qvel, modelInfo);

    mjState->qpos[3] = 1;
    mjState->qpos[4] = 0;
    mjState->qpos[5] = 0;
    mjState->qpos[6] = 0;


    // TODO test this logic
    if (handleBallRotation) {
        Num rotateDistance = r * thetaVel; // + or -
        Num angleVel[3];
        calculateBallAngularVelocity(angleVel, rotateDistance, theta);
        mjState->qvel[3] = angleVel[0];
        mjState->qvel[4] = angleVel[1];
        mjState->qvel[5] = angleVel[2];
    } else {
        mjState->qvel[3] = 0;
        mjState->qvel[4] = 0;
        mjState->qvel[5] = 0;
    }

    state->setBoardRotation(*this->modelInfo, boardRotX, boardRotY);
}

void ThetaDynamics::vec2state(const Num* in, int ringIdx, Num* out_qpos, Num* out_qvel, const ModelInfo* modelInfo) {
    Num r = modelInfo->centerRadius[ringIdx];

    Num boardRotX = in[0];
    Num boardRotY = in[1];
    Num theta = in[2];
    Num thetaVel = in[3];

    Num proj_x = r * std::cos(theta);
    Num proj_y = r * std::sin(theta);
    Num proj_z = -0.00365;
    Vector3 proj {proj_x, proj_y, proj_z};

    const Num vr = 0;
    Num proj_vel_x = vr * std::cos(theta) - thetaVel * r * std::sin(theta);
    Num proj_vel_y = vr * std::sin(theta) + thetaVel * r * std::cos(theta);
    Vector3 proj_vel{proj_vel_x, proj_vel_y, 0};

    Quaternion  quat = modelInfo->rotToQuat(boardRotX, boardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 coord_pos = R * proj;
    Vector3 coord_vel = R * proj_vel;

    out_qpos[0] = coord_pos.x();
    out_qpos[1] = coord_pos.y();
    out_qpos[2] = coord_pos.z();

    out_qvel[0] = coord_vel.x();
    out_qvel[1] = coord_vel.y();
    out_qvel[2] = coord_vel.z();
}

void ThetaDynamics::setRingState(GameState* state, int ringIdx, Num theta, const ModelInfo& modelInfo) {
    Dynamics::setRingState(state, ringIdx, theta, modelInfo);
    this->ringIdx = ringIdx;
}



Num ThetaDynamics::cost(GameState* state) {
    Num boardRotX = state->BoardRotX();
    Num boardRotY = state->BoardRotY();
    auto d = state->getMjData();
    Vector3 pos {d->qpos[0], d->qpos[1], d->qpos[2]};
    return ballCostFromTheta(boardRotX, boardRotY, pos, *this->modelInfo);
}

void ThetaDynamics::Lx(GameState* state, Num* out) {
    int dimSize = this->getVecSize();
    Num vec[dimSize];
    this->toVec(state, vec);
    Lx_vec(vec, out);
}

void ThetaDynamics::Lx_vec(const Num* vec, Num* out) {
    // out = double[dimSize]
    Num eps = 1e-6;
    int dimSize = this->getVecSize();
    Num theta = vec[IDX_THETA];

    for (int i = 0; i < dimSize; i++) {
        out[i] = 0;
    }
    Num diff = ballCostFromTheta(ringIdx, theta+eps, *this->modelInfo) - ballCostFromTheta(ringIdx, theta-eps, *this->modelInfo);
    out[IDX_THETA] = diff / (2*eps);
}

void ThetaDynamics::Lxx(GameState* state, Num* out) {
    int dimSize = this->getVecSize();
    Num vec[dimSize];
    this->toVec(state, vec);
    Lxx_vec(vec, out);
}


void ThetaDynamics::Lxx_vec(const Num* vec, Num* out) {
    // Lxx is almost 0. So we should try to use zero matrix as the return value.
    Num eps = 1e-6;

    int dimSize = this->getVecSize();
    int totalSize = dimSize*dimSize;
    for (int i = 0; i < totalSize; i++) {
        out[i] = 0;
    }
    Num base_lx[dimSize], pos_lx[dimSize], work_vec[dimSize];
    for (int i = 0; i < totalSize; i++) {
        work_vec[i] = vec[i];
    }

    work_vec[IDX_THETA] = work_vec[IDX_THETA] - eps;
    Lx_vec(work_vec, base_lx);
    work_vec[IDX_THETA] = work_vec[IDX_THETA] + 2*eps;
    Lx_vec(work_vec, pos_lx);
    // out[2,2] is not zero
    out[IDX_THETA + IDX_THETA*dimSize] = (pos_lx[IDX_THETA] - base_lx[IDX_THETA]) / (2*eps);
}


/*
 *
    - board_x
    - board_y
    - ball_theta
    - ball_theta_vel
*
 * */
void ThetaDynamics::fx_fu(Environment* environment, PhysicsEngine* physics,
                          Num xRot, Num yRot,
                          Num* fx_out, Num* fu_out) {
    using namespace std;

    const double board_eps = 1e-4;
    const double ball_eps = 1e-4;
    int dimState = this->getVecSize();
    Num vec[dimState];

    GameState * state = environment->getState();
    GameStateCache initialCache;
    state->saveState(initialCache);

    this->toVec(state, vec);
    this->setVec(state, vec);

    int nv = physics->getMjNv();
    mjtNum warmstart[nv];
    physics->calcWarmStart(state, warmstart);


    // This element are cache for center state
    GameStateCache stateCache;
    state->saveState(stateCache);

    environment->simulateForDerivative(state, xRot, yRot, physics, warmstart);

    Num center_vec[dimState];
    this->toVec(state, center_vec);

    state->loadState(stateCache);

    // 要素ごとのperturb
    for (int idx_vec = 0; idx_vec < dimState; idx_vec++) {
        double eps = 0.0;

        if (idx_vec == 0 || idx_vec == 1) {
            eps = board_eps;
        } else {
            eps = ball_eps;
        }

        vec[idx_vec] += eps;
        this->setVec(state, vec);
        vec[idx_vec] -= eps;

        environment->simulateForDerivative(state, xRot, yRot, physics, warmstart);
        std::vector<Num> diff = differentiateState(center_vec, *state, eps);

        for (int i = 0; i < dimState; i++) {
            // row-majorで同じ行は出力が同じ
            fx_out[idx_vec + i*dimState] = diff[i];
        }
    }

    {
        double eps = board_eps;
        state->loadState(stateCache);
        environment->simulateForDerivative(state, xRot+eps, yRot, physics, warmstart);
        std::vector<Num> diff = differentiateState(center_vec, *state, eps);

        for (int i = 0; i < dimState; i++) {
            // row-majorで同じ行は出力が同じ
            fu_out[i*2] = diff[i];
        }
    }
    {
        double eps = board_eps;
        state->loadState(stateCache);
        environment->simulateForDerivative(state, xRot, yRot+eps, physics, warmstart);
        std::vector<Num> diff = differentiateState(center_vec, *state, eps);

        for (int i = 0; i < dimState; i++) {
            // row-majorで同じ行は出力が同じ
            fu_out[i*2+1] = diff[i];
        }
    }

    state->loadState(initialCache);
}

std::vector<Num> ThetaDynamics::differentiateState(Num* from_vec, GameState& to, Num dt) {
    int dimState = this->getVecSize();
    Num to_vec[dimState];
    this->toVec(&to, to_vec);

    std::vector<Num> diff(dimState);

    for (int i = 0; i < dimState; i++) {
        diff[i] = (to_vec[i] - from_vec[i]) / dt;
    }

    return diff;
}
