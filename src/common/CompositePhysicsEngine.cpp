// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "CompositePhysicsEngine.h"

#include "WallRemover.h"
#include <boost/algorithm/string/split.hpp>


/*
 * TODO reuse codes in PhysicsEngine
 * */
SinglePhysicsEngine* makeSinglePhysics(ConfigMap configMap, WallConfig& wallConfig)
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


SinglePhysicsEngine* CompositePhysicsEngine::getDefaultEngine(){
    return engines[4].get();
}

CompositePhysicsEngine::CompositePhysicsEngine(std::vector<SinglePhysicsEngine*> engines) {
    for (int i = 0; i < engines.size(); i++) {
        this->engines.push_back(std::unique_ptr<SinglePhysicsEngine>{engines[i]});
        this->states.push_back(std::unique_ptr<GameState>{engines[i]->createNewGameState()});
    }
}

CompositePhysicsEngine* CompositePhysicsEngine::makeFromConfig(ConfigMap configMap) {
    std::vector<SinglePhysicsEngine*> engines;
    WallConfig wallConfig;

    for (int i = 0; i < 5 ; i++) {
        wallConfig.activateOnlySingleRing(i);
        SinglePhysicsEngine *engine = makeSinglePhysics(configMap, wallConfig);
        engines.push_back(engine);
    }

    auto physics = new CompositePhysicsEngine{
        engines
    };

    return physics;
}

int CompositePhysicsEngine::stateToIdx(const GameState* state) {
    auto ballPos = state->ballPos(0);
    int ringIdx = getModelInfo()->getRingIdx(ballPos);
    return ringIdx;
}

SinglePhysicsEngine* CompositePhysicsEngine::getCurrentEngine(GameState *state) {
    int engineIdx = this->stateToIdx(state);
    return engines[engineIdx].get();
}

void CompositePhysicsEngine::convertPhysicsState(GameState *dst, GameState *src) {
    GameStateCache cache;
    src->saveState(cache);
    dst->loadState(cache);
}


/**********************composite functions********************************/
void CompositePhysicsEngine::setBallFric(Num fricSlide, Num fricSpin, Num fricRoll){
    for (int i = 0; i < engines.size(); i++) {
        engines[i]->setBallFric(fricSlide, fricSpin, fricRoll);
    }
}

void CompositePhysicsEngine::setWallFric(Num fricSlide, Num fricSpin, Num fricRoll){
    for (int i = 0; i < engines.size(); i++) {
        engines[i]->setBallFric(fricSlide, fricSpin, fricRoll);
    }
}


void CompositePhysicsEngine::setFricloss(Num fricLoss){
    for (int i = 0; i < engines.size(); i++) {
        engines[i]->setFricloss(fricLoss);
    }
}


void CompositePhysicsEngine::calcWarmStart(GameState* state, double* out_warmstart) {
    int engineIdx = this->stateToIdx(state);
    PhysicsEngine* engine = this->engines[engineIdx].get();
    GameState* engineState = this->states[engineIdx].get();

    convertPhysicsState(engineState, state);

    engine->calcWarmStart(engineState, out_warmstart);

    convertPhysicsState(state, engineState);
}

bool CompositePhysicsEngine::computePhysics(GameState *state, float timeStep) {
    int engineIdx = this->stateToIdx(state);
    PhysicsEngine* engine = this->engines[engineIdx].get();
    GameState* engineState = this->states[engineIdx].get();

    convertPhysicsState(engineState, state);
    bool result = engine->computePhysics(engineState, timeStep);
    convertPhysicsState(state, engineState);

    return result;
}

bool CompositePhysicsEngine::computePhysicsForDerivative(GameState* state, float timeStep) {
    int engineIdx = this->stateToIdx(state);
    PhysicsEngine* engine = this->engines[engineIdx].get();
    GameState* engineState = this->states[engineIdx].get();

    convertPhysicsState(engineState, state);
    bool result = engine->computePhysicsForDerivative(engineState, timeStep);
    convertPhysicsState(state, engineState);

    return result;
}


/**********************trivial methods************************************/
int CompositePhysicsEngine::jnt_qposadr(int ballIdx){
    return getDefaultEngine()->jnt_qposadr(ballIdx);
}

int CompositePhysicsEngine::jnt_dofadr(int ballIdx) {
    return getDefaultEngine()->jnt_dofadr(ballIdx);
}

int CompositePhysicsEngine::getMjNv() {
    return getDefaultEngine()->getMjNv();
}

int CompositePhysicsEngine::getMjNq() {
    return getDefaultEngine()->getMjNq();
}

ModelInfo* CompositePhysicsEngine::getModelInfo() {
    return getDefaultEngine()->getModelInfo();
}

void CompositePhysicsEngine::setBoardRotation(GameState *state, Num xRot, Num yRot) {
    getDefaultEngine()->setBoardRotation(state, xRot, yRot);
}

bool CompositePhysicsEngine::checkBallCollisions(GameState *state, unsigned int ballIdx, Num x, Num y) {
    return getDefaultEngine()->checkBallCollisions(state, ballIdx, x, y);
}

void CompositePhysicsEngine::resetState(GameState *state) {
    getDefaultEngine()->resetState(state);
}

GameState* CompositePhysicsEngine::createNewGameState() {
    getDefaultEngine()->createNewGameState();
}
