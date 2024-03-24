// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef BIMGAME_DYNAMICS_H
#define BIMGAME_DYNAMICS_H


#include "mujoco.h"
#include <vector>
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
#include <iostream>
#include <fstream>
#include <iterator>
#include <memory>
#include "type_config.h"
#include "stdlib.h"
#include "GameState.h"


class ModelInfo;
class GameState;
class PhysicsEngine;
class Environment;

enum RewardMode {
    RING_SPARSE,
    DISTANCE_DENSE,
    THETA_DISTANCE
};


/* common implementations to calculate dynamcis*/
// cost function with respect to angle
Num ballCostFromTheta(int ring, Num theta, const ModelInfo& modelInfo);
Num ballCostFromTheta(Num boardRotX, Num boardRotY, Vector3 ballPos, const ModelInfo& modelInfo);
Num ballCostFromThetaAndDistance(Num boardRotX, Num boardRotY, Vector3 ballPos, const ModelInfo& modelInfo);

/*
 * calculate cost function and derivatives
 * "vec" means the vector representation of state.
 * */
class Dynamics {
public:
    virtual Num cost(GameState* state) = 0;
    virtual void Lx(GameState* state, Num* out) = 0;
    virtual void Lxx(GameState* state, Num* out) = 0;

    virtual int getVecSize() = 0;
    virtual void toVec(GameState* state, Num* out) = 0;
    virtual void setVec(GameState* state, const Num* in) = 0;
    virtual int getRingIdx(GameState* state) = 0;

    /*
     * set state with ring index and theta
     * */
    virtual void setRingState(GameState* state, int ringIdx, Num theta, const ModelInfo& modelInfo);

    virtual void fx_fu(Environment* environment, PhysicsEngine* physics,
                       Num xRot, Num yRot,
                       Num* fx_out, Num* fu_out) = 0;


    virtual ~Dynamics(){}
};


/*
 * This assumes the state which are composed of 4 elements
 *
 - boardRotX
 - boardRotY
 - theta
 - thetaVel
 * */
class ThetaDynamics : public Dynamics{
public:
    ThetaDynamics(const ModelInfo* info, int ringIdx, bool handleBallRotation, RewardMode rewardMode);

    // state management
    virtual int getVecSize();
    virtual void toVec(GameState* state, Num* out);
    virtual void setVec(GameState* state, const Num* in);
    virtual void setRingState(GameState* state, int ringIdx, Num theta, const ModelInfo& modelInfo) override;

    // cost functions
    virtual Num cost(GameState* state);
    virtual void Lx(GameState* state, Num* out);
    virtual void Lxx(GameState* state, Num* out);

    virtual void fx_fu(Environment* environment, PhysicsEngine* physics,
                       Num xRot, Num yRot,
                       Num* fx_out, Num* fu_out);

    // specific implementations
    void setRingIdx(int ringIdx) {this->ringIdx = ringIdx;}
    int getRingIdx(GameState* state) { return ringIdx; }

    static void makeVec(GameState* state, Num* out, const ModelInfo* modelInfo);

    /*
     *
     * @param out_qpos: double[3] output qpos values (this doesn't contain quaternion)
     * @param out_qvel: double[3] output qvel values (this doesn't contain quaternion)
     * */
    static void vec2state(const Num* in, int ringIdx, Num* out_qpos, Num* out_qvel, const ModelInfo* modelInfo);
    static int getThetaVecSize();

    virtual ~ThetaDynamics();
private:
    virtual void Lx_vec(const Num* vec, Num* out);
    virtual void Lxx_vec(const Num* vec, Num* out);
    virtual std::vector<Num> differentiateState(Num* from_vec, GameState& to, Num dt);

    const int IDX_THETA=2;
    int ringIdx;
    // This members is not owned by this class
    const ModelInfo* modelInfo;
    bool handleBallRotation;
    RewardMode rewardMode;
};

/*
 * This assumes the states which are composed of 8 or 15 elements (1 ball)
 *
 *
 * */
class XYZDynamics : public Dynamics{
public:
    XYZDynamics(const ModelInfo* info, int numBalls, bool handleBoardRotation, RewardMode rewardMode);

    // cost functions
    virtual Num cost(GameState* state);
    virtual void Lx(GameState* state, Num* out);
    virtual void Lxx(GameState* state, Num* out);


    virtual int getVecSize();
    virtual void toVec(GameState* state, Num* out);
    static void makeVec(GameState* state, Num* vec, int numBalls, bool handleBallRotation);
    virtual void setVec(GameState* state, const Num* in);
    virtual int getRingIdx(GameState* state);

    virtual void fx_fu(Environment* environment, PhysicsEngine* physics,
                       Num xRot, Num yRot,
                       Num* fx_out, Num* fu_out);

    virtual void fx_fu_theta(Environment* environment, PhysicsEngine* physics,
                             Num xRot, Num yRot,
                             Num* fx_out, Num* fu_out);

    void Lxx_theta(GameState* state, Num* out);
    void Lx_theta(GameState* state, Num* out);

    static int calcVecSizeWithBall(int numBalls, bool handleBallRotation);
    virtual ~XYZDynamics(){}
private:
    Num ballCost(Num boardRotX, Num boardRotY, Vector3 ballPos, const ModelInfo& modelInfo);
    std::vector<Num> Lx_impl(Num boardRotX, Num boardRotY, Vector3 ballPos);
    std::vector<Num> differentiateState(PhysicsEngine* physics, const GameState& from, const GameState& to, Num dt);

    bool handleBallRotation;
    int numBalls;
    const ModelInfo* modelInfo;
    RewardMode rewardMode;

    // theta dynamics used to calculate fx_fu_theta, Lx_theta, ...
    std::unique_ptr<ThetaDynamics> thetaDynamics;
};

#endif //BIMGAME_DYNAMICS_H
