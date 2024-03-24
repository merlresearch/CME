#include "Dynamics.h"
// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "mujoco.h"
#include "ModelInfo.h"

#include "GameState.h"
#include "SinglePhysicsEngine.h"
#include "Environment.h"

XYZDynamics::XYZDynamics(const ModelInfo* info, int numBalls, bool handleBoardRotation, RewardMode rewardMode) : modelInfo(info){
    switch (rewardMode) {
        case RewardMode::RING_SPARSE:
            throw std::runtime_error("sparse error is not supported in this mode.");
        case RewardMode::DISTANCE_DENSE:
        case RewardMode::THETA_DISTANCE:
            this->rewardMode = rewardMode;
    }

    this->numBalls = numBalls;
    this->handleBallRotation = handleBoardRotation;
    this->thetaDynamics.reset(new ThetaDynamics{info, 3, handleBoardRotation, rewardMode});
}

int XYZDynamics::getVecSize() {
    return calcVecSizeWithBall(this->numBalls, this->handleBallRotation);
}

int XYZDynamics::calcVecSizeWithBall(int numBalls, bool handleBallRotation) {
    int dimension = 2;
    dimension += numBalls*6;

    if (handleBallRotation) {
        dimension += numBalls * 7;
    }
    return dimension;
}


void XYZDynamics::toVec(GameState* state, Num* vec) {
    makeVec(state, vec, this->numBalls, this->handleBallRotation);
}

void XYZDynamics::makeVec(GameState* state, Num* vec, int numBalls, bool handleBallRotation) {
    int offset = 0;
    vec[0] = state->BoardRotX();
    vec[1] = state->BoardRotY();
    offset += 2;

    mjData* d = state->getMjData();

    // TODO use bodyIdx
    for (int i = 0; i < numBalls; i++) {
        int qposIdx = i*7;
        int qvelIdx = i*6;

        if (handleBallRotation) {
            vec[offset+0] = d->qpos[qposIdx];
            vec[offset+1] = d->qpos[qposIdx+1];
            vec[offset+2] = d->qpos[qposIdx+2];
            vec[offset+3] = d->qpos[qposIdx+3];
            vec[offset+4] = d->qpos[qposIdx+4];
            vec[offset+5] = d->qpos[qposIdx+5];
            vec[offset+6] = d->qpos[qposIdx+6];

            offset += 7;

            vec[offset]   = d->qvel[qvelIdx];
            vec[offset+1] = d->qvel[qvelIdx+1];
            vec[offset+2] = d->qvel[qvelIdx+2];
            vec[offset+3] = d->qvel[qvelIdx+3];
            vec[offset+4] = d->qvel[qvelIdx+4];
            vec[offset+5] = d->qvel[qvelIdx+5];

            offset += 6;
        } else {
            vec[offset+0] = d->qpos[qposIdx];
            vec[offset+1] = d->qpos[qposIdx+1];
            vec[offset+2] = d->qpos[qposIdx+2];
            offset += 3;

            vec[offset]   = d->qvel[qvelIdx];
            vec[offset+1] = d->qvel[qvelIdx+1];
            vec[offset+2] = d->qvel[qvelIdx+2];
            offset += 3;
        }
    }
}


void XYZDynamics::setVec(GameState* state, const Num* vec){
    // mjData* newData = mj_makeData(info->mjModel_);
    mjData* newData = state->getMjData();
    mjtNum defBallQuat[4] = {1.0, 0.0, 0.0, 0.0};
    mjtNum defBallAngVel[3] = {0.0, 0.0, 0.0};

    int offset = 0;
    Num boardRotX = vec[0];
    Num boardRotY = vec[1];
    offset += 2;

    mjtNum boardPos[3] = {0.0, 0.0, 0.0};
    memcpy(newData->mocap_pos, boardPos, sizeof(mjtNum)*3);


    for (int i = 0; i < numBalls; i++) {
        if (handleBallRotation) {
            memcpy(newData->qpos + (i * 7), vec + offset, sizeof(mjtNum) * 7);
            offset += 7;
            memcpy(newData->qvel + (i * 6), vec + offset, sizeof(mjtNum) * 6);
            offset += 6;
        } else {
            memcpy(newData->qpos + (i * 7), vec + offset, sizeof(mjtNum) * 3);
            memcpy(newData->qpos + (i * 7) + 3, defBallQuat, sizeof(mjtNum) * 4);
            offset += 3;
            memcpy(newData->qvel + (i * 6), vec + offset, sizeof(mjtNum) * 3);
            memcpy(newData->qvel + (i * 6) + 3, defBallAngVel, sizeof(mjtNum) * 3);
            offset += 3;
        }
    }

    state->setBoardRotation(*this->modelInfo, boardRotX, boardRotY);
}

int XYZDynamics::getRingIdx(GameState* state) {
    Vector3 ballPos = state->ballPos(0);
    return this->modelInfo->getRingIdx(ballPos);
}

void XYZDynamics::Lx(GameState* state, Num* out) {
    // TODO (oiki) : support multiple balls
    if(numBalls != 1) {
        throw std::runtime_error("Lx supports only 1 ball.");
    }

    if (handleBallRotation) {
        throw std::runtime_error("Lx supports only handleBallRotation=False.");
    }

    std::vector<Num> lx {Lx_impl(state->BoardRotX(), state->BoardRotY(), state->ballPos(0))};

    for (int i = 0; i < lx.size(); i++) {
        out[i] = lx[i];
    }
}


std::vector<Num> XYZDynamics::Lx_impl(Num boardRotX, Num boardRotY, Vector3 ballPos) {
    const double eps = 1e-6;
    std::vector<Num> out(8, 0);

    // for Board Rotation (2 degrees)
    double posCost = ballCost(boardRotX+eps, boardRotY, ballPos, *modelInfo);
    double negCost = ballCost(boardRotX-eps, boardRotY, ballPos, *modelInfo);
    out[0] = (posCost - negCost) / (2*eps);
    posCost = ballCost(boardRotX, boardRotY+eps, ballPos, *modelInfo);
    negCost = ballCost(boardRotX, boardRotY-eps, ballPos, *modelInfo);
    out[1] = (posCost - negCost) / (2*eps);

    // for Ball Position
    for (int idx_axis = 0; idx_axis < 3; idx_axis++) {
        Vector3 curPos = ballPos;

        // pos vec
        curPos[idx_axis] += eps;
        double posCost = ballCost(boardRotX, boardRotY, curPos, *modelInfo);

        // neg vec
        curPos[idx_axis] -= 2*eps;
        double negCost = ballCost(boardRotX, boardRotY, curPos, *modelInfo);
        out[2+idx_axis] = (posCost - negCost) / (2*eps);
    }

    return out;
}

void XYZDynamics::Lx_theta(GameState* state, Num* out) {
    Vector3 ballPos = state->ballPos(0);
    int ringIdx = modelInfo->getRingIdx(ballPos);
    this->thetaDynamics->setRingIdx(ringIdx);

    this->thetaDynamics->Lx(state, out);
}

void XYZDynamics::Lxx_theta(GameState* state, Num* out) {
    Vector3 ballPos = state->ballPos(0);
    int ringIdx = modelInfo->getRingIdx(ballPos);
    this->thetaDynamics->setRingIdx(ringIdx);

    this->thetaDynamics->Lxx(state, out);
}

void XYZDynamics::Lxx(GameState* state, Num* out) {
    if(numBalls != 1) {
        throw std::runtime_error("Lxx supports only 1 ball.");
    }

    if (handleBallRotation) {
        throw std::runtime_error("Lxx supports only handleBallRotation=False.");
    }


    const double eps = 1e-6;
    int col_size = 8;

    for (int i = 0; i < col_size; i++) {
        for (int j = 0; j < col_size; j++) {
            out[j+i*col_size] = 0;
        }
    }

    Vector3 ballPos = state->ballPos(0);
    Num boardRotX = state->BoardRotX();
    Num boardRotY = state->BoardRotY();

    std::vector<Num> posLx = Lx_impl(boardRotX+eps, boardRotY, ballPos);
    std::vector<Num> negLx = Lx_impl(boardRotX-eps, boardRotY, ballPos);

    for (int col=0; col < col_size; col++) {
        out[0*col_size + col] = (posLx[col] - negLx[col]) / (2*eps);
    }

    posLx = Lx_impl(boardRotX, boardRotY+eps, ballPos);
    negLx = Lx_impl(boardRotX, boardRotY-eps, ballPos);

    for (int col=0; col < col_size; col++) {
        out[1*col_size + col] = (posLx[col] - negLx[col]) / (2*eps);
    }

    for (int idx_axis = 0; idx_axis < 3; idx_axis++) {
        Vector3 curPos = ballPos;

        // pos vec
        curPos[idx_axis] += eps;
        posLx = Lx_impl(boardRotX, boardRotY, curPos);

        // neg vec
        curPos[idx_axis] -= 2*eps;
        negLx = Lx_impl(boardRotX, boardRotY, curPos);

        for (int col=0; col < col_size; col++) {
            out[(2+idx_axis)*col_size + col] = (posLx[col] - negLx[col]) / (2*eps);
        }
    }
}

Num XYZDynamics::ballCost(Num boardRotX, Num boardRotY, Vector3 ballPos, const ModelInfo& modelInfo){
    Num cost;

    switch (rewardMode){
        case DISTANCE_DENSE:
            cost = ballCostFromThetaAndDistance(boardRotX, boardRotY, ballPos, modelInfo);
            break;
        case THETA_DISTANCE:
            cost = ballCostFromTheta(boardRotX, boardRotY, ballPos, modelInfo);
            break;
        case RING_SPARSE:
            throw std::runtime_error("sparse reward is not implemented.");
        default:
            throw std::runtime_error("unknown reward.");
    }

    return cost;
}


Num XYZDynamics::cost(GameState* state) {
    Num cost = 0.0f;
    mjData* mjState = state->getMjData();

    for (int i = 0; i < this->numBalls; i++) {
        Vector3 pos {mjState->qpos[i*3], mjState->qpos[i*3+1], mjState->qpos[i*3+2]};

        Num curCost = ballCost(
                state->BoardRotX(),
                state->BoardRotY(),
                pos,
                *this->modelInfo);
        cost += curCost;
    }

    return cost;
}

void XYZDynamics::fx_fu(Environment* environment, PhysicsEngine* physics,
        Num xRot, Num yRot,
        Num* fx_out, Num* fu_out) {
    using namespace std;

    if (handleBallRotation) {
        throw std::runtime_error("fx_out fu_out is not supported in handleBallRotation=True.");
    }

    int nwarmup = 3;
    const double board_eps = 1e-3;
    const double ball_eps = 1e-5;

    GameState * state = environment->getState();

    // calculate warmstart
    ModelInfo *modelInfo = physics->getModelInfo();
    int nv = physics->getMjNv();
    mjtNum warmstart[nv];
    physics->calcWarmStart(state, warmstart);

    int dim_fx = 8;
    int jointId = physics->jnt_qposadr(0);
    int velId = physics->jnt_dofadr(0);

    unique_ptr<GameState> centerState {state->copy(*this->modelInfo, physics)};
    environment->simulateForDerivative(centerState.get(), xRot, yRot, physics, warmstart);

    for (int idx_vec = 0; idx_vec < dim_fx; idx_vec++) {
        double eps = 0.0;
        unique_ptr<GameState> posState {state->copy(*this->modelInfo, physics)};

        if (idx_vec == 0) {
            eps = board_eps;
            posState->incrementBoardRotation(*this->modelInfo, eps, 0);
        } else if (idx_vec == 1){
            eps = board_eps;
            posState->incrementBoardRotation(*this->modelInfo, 0, eps);
        } else if (idx_vec < 5){ //2,3,4
            eps = ball_eps;
            int relIdx = idx_vec - 2;
            posState->getMjData()->qpos[jointId + relIdx] += eps;
        } else if (idx_vec < 8) { //5,6,7
            eps = ball_eps;
            int relIdx = idx_vec - 5;
            posState->getMjData()->qvel[velId + relIdx] += eps;
        }

        environment->simulateForDerivative(posState.get(), xRot, yRot, physics, warmstart);
        std::vector<Num> diff = differentiateState(physics, *centerState, *posState, eps);

        for (int i = 0; i < dim_fx; i++) {
            fx_out[idx_vec + i*dim_fx] = diff[i];
        }

    }

    {
        double eps = board_eps;
        unique_ptr<GameState> posState{state->copy(*this->modelInfo, physics)};
        environment->simulateForDerivative(posState.get(), xRot+eps, yRot, physics, warmstart);
        std::vector<Num> diff = differentiateState(physics, *centerState, *posState, eps);


        for (int i = 0; i < dim_fx; i++) {
            fu_out[i*2] = diff[i];
        }
    }
    {
        double eps = board_eps;
        unique_ptr<GameState> posState{state->copy(*this->modelInfo, physics)};
        environment->simulateForDerivative(posState.get(), xRot, yRot+eps, physics, warmstart);
        std::vector<Num> diff = differentiateState(physics, *centerState, *posState, eps);

        for (int i = 0; i < dim_fx; i++) {
            fu_out[i*2+1] = diff[i];
        }
    }
}

void pos2angle(Vector3 pos, Num boardRotX, Num boardRotY, const ModelInfo* modelInfo,
        Num* angle, Num* r, Num* projected_z){
    Quaternion  quat = modelInfo->rotToQuat(boardRotX, boardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 proj_pos = R.transpose() * pos;
    *angle = std::atan2(proj_pos.y(), proj_pos.x());
    *r = std::sqrt(proj_pos.y()*proj_pos.y() + proj_pos.x()*proj_pos.x());
    *projected_z = proj_pos.z();
}

Vector3 angle2pos(Num r, Num angle, Num projected_z, Num boardRotX, Num boardRotY, const ModelInfo* modelInfo){
    Num proj_x = r * std::cos(angle);
    Num proj_y = r * std::sin(angle);
    Num proj_z = projected_z;
    Vector3 proj {proj_x, proj_y, proj_z};

    Quaternion  quat = modelInfo->rotToQuat(boardRotX, boardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 coord_pos = R * proj;
    return coord_pos;
}


Num vel2angVel(Num angle, Num r, Vector3 vel, Num boardRotX, Num boardRotY, const ModelInfo* modelInfo){
    Quaternion  quat = modelInfo->rotToQuat(boardRotX, boardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 proj_vel = R.transpose() * vel;
    Num angleVel = (-proj_vel.x() * std::sin(angle) + proj_vel.y() * std::cos(angle)) / r;

    return angleVel;
}

Vector3 getVelEPS(Num eps, Num angle, Num r, Num boardRotX, Num boardRotY, const ModelInfo* modelInfo){
    Num proj_vel_x = -std::sin(angle) * r * eps;
    Num proj_vel_y = std::cos(angle) * r * eps;
    Vector3 proj_vel {proj_vel_x, proj_vel_y, 0};

    Quaternion  quat = modelInfo->rotToQuat(boardRotX, boardRotY);
    Matrix3 R = quat.toRotationMatrix();

    Vector3 coord_vel = R * proj_vel;
    return coord_vel;
}



/*
 *
    - board_x
    - board_y
    - ball_theta
    - ball_theta_vel
*
 * */
void XYZDynamics::fx_fu_theta(Environment* environment, PhysicsEngine* physics,
                          Num xRot, Num yRot,
                          Num* fx_out, Num* fu_out) {
    using namespace std;

    const double board_eps = 1e-4;
    const double ball_eps = 1e-4;
    int dimState = 4;

    GameState * state = environment->getState();
    GameStateCache initialCache;
    state->saveState(initialCache);

    Vector3 ballPos = state->ballPos(0);
    Vector3 ballVel = state->ballVel(0);

    Num init_angle, init_r, init_z;
    pos2angle(ballPos, state->BoardRotX(), state->BoardRotY(), modelInfo, &init_angle, &init_r, &init_z);

    int nv = physics->getMjNv();
    mjtNum warmstart[nv];
    physics->calcWarmStart(state, warmstart);

    // This element are cache for center state
    GameStateCache stateCache;
    state->saveState(stateCache);

    environment->simulateForDerivative(state, xRot, yRot, physics, warmstart);

    Num center_angle, center_r, center_z;
    pos2angle(state->ballPos(0), state->BoardRotX(), state->BoardRotY(), modelInfo, &center_angle, &center_r, &center_z);
    Num center_vec[dimState];
    ThetaDynamics::makeVec(state, center_vec, modelInfo);

    for (int idx_vec = 0; idx_vec < dimState; idx_vec++) {
        double eps = 0.0;
        state->loadState(stateCache);

        if (idx_vec == 0) {
            eps = board_eps;
            state->incrementBoardRotation(*this->modelInfo, eps, 0);
        } else if (idx_vec == 1){
            eps = board_eps;
            state->incrementBoardRotation(*this->modelInfo, 0, eps);
        } else if (idx_vec == 2){
            eps = ball_eps;
            Num new_angle = init_angle + eps;
            Vector3 eps_pos = angle2pos(init_r, new_angle, init_z, state->BoardRotX(), state->BoardRotY(), modelInfo);

            state->getMjData()->qpos[0] = eps_pos.x();
            state->getMjData()->qpos[1] = eps_pos.y();
            state->getMjData()->qpos[2] = eps_pos.z();
        } else if (idx_vec == 3) {
            eps = ball_eps;

            Vector3 vel_eps = getVelEPS(eps, init_angle, init_r, state->BoardRotX(), state->BoardRotY(), modelInfo);
            Vector3 new_vel = ballVel + vel_eps;

            state->getMjData()->qvel[0] = new_vel.x();
            state->getMjData()->qvel[1] = new_vel.y();
            state->getMjData()->qvel[2] = new_vel.z();
        }

        environment->simulateForDerivative(state, xRot, yRot, physics, warmstart);

        Num to_vec[dimState];
        ThetaDynamics::makeVec(state, to_vec, modelInfo);

        for (int i = 0; i < dimState; i++) {
            fx_out[idx_vec + i*dimState] = (to_vec[i] - center_vec[i]) / eps;
        }
    }

    {
        double eps = board_eps;
        state->loadState(stateCache);
        environment->simulateForDerivative(state, xRot+eps, yRot, physics, warmstart);

        Num to_vec[dimState];
        ThetaDynamics::makeVec(state, to_vec, modelInfo);

        for (int i = 0; i < dimState; i++) {
            fu_out[i*2] = (to_vec[i] - center_vec[i]) / eps;
        }
    }
    {
        double eps = board_eps;
        state->loadState(stateCache);
        environment->simulateForDerivative(state, xRot, yRot+eps, physics, warmstart);

        Num to_vec[dimState];
        ThetaDynamics::makeVec(state, to_vec, modelInfo);

        for (int i = 0; i < dimState; i++) {
            fu_out[i*2+1] = (to_vec[i] - center_vec[i]) / eps;
        }
    }

    state->loadState(initialCache);
}


std::vector<Num> XYZDynamics::differentiateState(PhysicsEngine* physics, const GameState& from, const GameState& to, Num dt) {
    std::vector<Num> diff(8, 0);
    int qposId = physics->jnt_qposadr(0);
    int qvelId = physics->jnt_dofadr(0);

    diff[0] = (to.BoardRotX() - from.BoardRotX()) / dt;
    diff[1] = (to.BoardRotY() - from.BoardRotY()) / dt;

    for (int i = 2; i< 5; i++) {
        int relIdx = i - 2;
        diff[i] = (to.getMjData()->qpos[qposId+relIdx] - from.getMjData()->qpos[qposId+relIdx]) / dt;
    }

    for (int i = 5; i< 8; i++) {
        int relIdx = i - 5;
        diff[i] = (to.getMjData()->qvel[qvelId+relIdx] - from.getMjData()->qvel[qvelId+relIdx]) / dt;
    }

    return diff;
}
