// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef BIMGAME_RENDERINTERFACE_H
#define BIMGAME_RENDERINTERFACE_H

#include <vector>
#include <string>

class SinglePhysicsEngine;
class RenderingEngine;
class Environment;

class RenderInterface {
public:
    RenderInterface(std::string path_config);
    ~RenderInterface();

    int getHeight();
    int getWidth();
    void setRingIdx(int ringIdx);
    void render(const std::vector<double>& state, unsigned char* image);
private:
    SinglePhysicsEngine* physics;
    RenderingEngine* renderer;
    Environment* viewEnv;
    bool hasBallRotation;
    int ringIdx;
};



#endif //BIMGAME_RENDERINTERFACE_H
