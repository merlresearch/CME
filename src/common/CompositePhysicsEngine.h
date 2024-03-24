// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef MAZESIMULATOR_COMPOSITEPHYSICSENGINE_H
#define MAZESIMULATOR_COMPOSITEPHYSICSENGINE_H

#include <vector>
#include <memory>

#include "SinglePhysicsEngine.h"

/*
 * This class handles multiple PhysicsEngine instances to improve performance.
 * The physics engine which is used for simulation is selected by the ball position.
 *
 * All models must have the same qpos, qvel and mocap.
 * */

class CompositePhysicsEngine : public PhysicsEngine {
public:
    static CompositePhysicsEngine* makeFromConfig(ConfigMap configMap);
    CompositePhysicsEngine(std::vector<SinglePhysicsEngine*> engines);
    virtual ~CompositePhysicsEngine(){}

    virtual void setBallFric(Num fricSlide, Num fricSpin, Num fricRoll);
    virtual void setWallFric(Num fricSlide, Num fricSpin, Num fricRoll);
    virtual void setFricloss (Num fricLoss);
    virtual void calcWarmStart(GameState* state, double* out_warmstart);
    virtual bool computePhysics(GameState* state, float timeStep);
    virtual bool computePhysicsForDerivative(GameState* state, float timeStep);

    /**************trivial methods****************/
    // properties
    virtual int jnt_qposadr(int ballIdx);
    virtual int jnt_dofadr(int ballIdx);
    virtual int getMjNv();
    virtual int getMjNq ();
    virtual ModelInfo* getModelInfo();

    // common processes
    virtual GameState* createNewGameState();
    virtual void setBoardRotation(GameState *state, Num xRot, Num yRot);
    virtual bool checkBallCollisions(GameState* state, unsigned int ballIdx, Num x, Num y);
    virtual void resetState(GameState* state);


private:
    // returns the engine doing processes which don't depend on a game state.
    SinglePhysicsEngine* getDefaultEngine();
    SinglePhysicsEngine* getCurrentEngine(GameState* state);
    // convert general GameState to physics engine specific GameState
    // this returns the pointer to any state in states member.
    void convertPhysicsState(GameState *dst, GameState *src);

    // index corresponds to ringIdx
    std::vector<std::unique_ptr<SinglePhysicsEngine> > engines;
    // states for each physics engine
    std::vector<std::unique_ptr<GameState> > states;
    // get engine index from GameState
    int stateToIdx(const GameState* state);

};


#endif //MAZESIMULATOR_COMPOSITEPHYSICSENGINE_H
