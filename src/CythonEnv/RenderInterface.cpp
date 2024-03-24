// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "RenderInterface.h"
#include "../render/RenderingEngine.h"
#include "../common/SinglePhysicsEngine.h"
#include "../common/Environment.h"

using namespace std;

RenderInterface::RenderInterface(std::string path_config) : physics(NULL), renderer(NULL), viewEnv(NULL){
    std::map<std::string, std::string> configMap;
    HelperFunctions::readConfig(path_config.c_str(), configMap);

    if(configMap["HandleBallRotation"] == "Yes") {
        hasBallRotation = true;
    } else if (configMap["HandleBallRotation"] == "No") {
        hasBallRotation = false;
    } else {
        throw std::runtime_error("invalid HandleBallRotation");
    }

    physics = SinglePhysicsEngine::makeFromConfig(configMap);
    renderer = RenderingEngine::makeFromConfig(configMap);
    viewEnv = Environment::makeFromConfig(configMap, physics);
    ringIdx = 4;
}

void RenderInterface::setRingIdx(int ringIdx) {
    ThetaDynamics* dynamics = dynamic_cast<ThetaDynamics*> (viewEnv->getDynamics());

    if (dynamics != NULL){
        dynamics->setRingIdx(ringIdx);
    }
}

RenderInterface::~RenderInterface() {
    if (physics != NULL) {
        delete physics;
    }
    if (renderer != NULL) {
        delete renderer;
    }
    if (viewEnv != NULL) {
        delete viewEnv;
    }
}

int RenderInterface::getHeight() {
    return renderer->RenderHeight();
}

int RenderInterface::getWidth() {
    return renderer->RenderWidth();
}


void RenderInterface::render(const std::vector<double>& state, unsigned char* image){
    using namespace std;
    GameState* gameState = viewEnv->getState();
    viewEnv->getDynamics()->setVec(gameState, state.data());

    renderer->renderScene(gameState, image);
}
