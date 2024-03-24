// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef MAZESIMULATOR_WALLREMOVER_H
#define MAZESIMULATOR_WALLREMOVER_H


#include <string>
#include <stdexcept>
#include "mujoco.h"

struct WallConfig {
    const int NUM_WALLS=5;
    // index corresponds to innerRadius in ModelInfo.cpp
    bool useWalls[5];

    WallConfig() {
        for (int i = 0; i < NUM_WALLS; i++) {
            useWalls[i] = true;
        }
    }

    WallConfig(const WallConfig& src) {
        for (int i = 0; i < NUM_WALLS; i++) {
            useWalls[i] = src.useWalls[i];
        }
    }

    WallConfig& operator=(const WallConfig& src) {
        for (int i = 0; i < NUM_WALLS; i++) {
            useWalls[i] = src.useWalls[i];
        }
        return *this;
    }

    // index corresponds to centerRadius in ModelInfo.cpp
    void activateOnlySingleRing(int ringIdx) {
        if (ringIdx >= NUM_WALLS) {
            throw std::runtime_error("invalid ring idx");
        }

        for (int i = 0; i < NUM_WALLS; i++) {
            useWalls[i] = false;
        }

        useWalls[ringIdx] = true;
        if (ringIdx != 0)
            useWalls[ringIdx-1] = true;
    }

    static WallConfig makeWallConfig(const std::string &maze_walls_file);
};



mjModel* loadXMLWithWallLimitation(const char *filename, const WallConfig* wallConfig);



#endif //MAZESIMULATOR_WALLREMOVER_H
