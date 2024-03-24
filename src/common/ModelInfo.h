// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef BIMGAME_MODELINFO_H
#define BIMGAME_MODELINFO_H

#include <random>
#include <vector>
#include <Eigen/Dense>

#include <mujoco.h>
#include "type_config.h"


/*
 * Model Information
 * This structs manages the model information, which cannot be represented in the MuJoCo file.
 *
 * All values of these members are decide at the construction timing.
 * You MUST NOT modify these members.
 * */
struct ModelInfo {
    // the number of rings. ring0 is the center point.
    int numRings;

    // the inner radius of each ring.
    // the length is [numRings]
    Num* innerRadius;

    // the center radius of each ring.
    // the length is [numRings]
    Num* centerRadius;

    // gates radians [numRings]
    std::vector<std::vector<Num> > listHoleRadius;

    // the names of marbles in the MuJoCo file.
    std::string* marbleNames;

    // the names of marble joints in the MuJoCo file.
    std::string* marbleJointNames;

    // the rotation axis
    // the radian from ordinal x axis.
    Num xRotAxis, yRotAxis;

    Num maxRadius = 0.09635;
    const Num marbleRadius = 0.00635;

    // the position where the ball is located at the center of the board.
    Vector3 ballCenterPtOnBoard {0., 0., -0.010 + 0.00635};

    template <typename URBG>
    Vector2 sampleRandomBallPos (int idxRing, URBG& randomGenerator) const {
        //Set seed to current time, ensuring each run generates different random numbers
        srand ((unsigned int) time(NULL));


        if (idxRing == -1) {
            // a ball doesn't start from the center.
            std::uniform_int_distribution<int> dist(1, numRings-1);
            idxRing = dist(randomGenerator);
        }

        std::uniform_real_distribution<Num> dist(0, 2*M_PI);
        Num angle_rad = dist(randomGenerator);
        Num distance = centerRadius[idxRing];

        Num x = distance * cos(angle_rad);
        Num y = distance * sin(angle_rad);

        return Vector2{x, y};
    }

    int getOuterRingIdx() const{
        return numRings-1;
    }

    Quaternion rotToQuat(Num rotX, Num rotY) const;

    // get mujoco Id of the ball
    int getMarbleJointId(mjModel *m_mjcModel, int idx);
    std::string getMarbleName(int idx);

    // construct ModelInfo
    /**
     * vMjModel : mjModel, which is copied in this constructor
     * ***/
    ModelInfo(Num xRotAxis, Num yRotAxis);

    virtual ~ModelInfo();
    int getRingIdx(const Vector2& pos) const;
    int getRingIdx(const Vector3& pos) const;

    int getMarbleBodyId(mjModel *m_mjcModel, int idx);
    int getWallBodyId(mjModel *m_mjcModel);

    ModelInfo& operator=(const ModelInfo&) = delete;
    ModelInfo(const ModelInfo&) = delete;

};

#endif //BIMGAME_MODELINFO_H
