// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

//
// Created by saket on 19/08/30.
//
#include <gtest/gtest.h>

#include <cmath>
#include "../common/type_config.h"
#include "../common/ModelInfo.h"
#include "../common/GameState.h"
#include "../common/Dynamics.h"


TEST(Reward, BallStateCost)
{
    using namespace std;
    ModelInfo info{0.0, 0.0};

    Num boardRotX=0.0;
    Num boardRotY=0.0;

    // centerRadius[4] = 0.08879;
    Num centerRadius = info.centerRadius[4];

    Vector3 ballPos {0, centerRadius, 0};
    Vector3 nearHole {centerRadius * cos(M_PI/4), centerRadius * sin(M_PI/4), 0};

    Num nearCost = ballCostFromThetaAndDistance(boardRotX, boardRotY, nearHole, info);
    Num topCost = ballCostFromThetaAndDistance(boardRotX, boardRotY, ballPos, info);

    ASSERT_LT(nearCost, topCost);
}
