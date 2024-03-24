// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "mujoco.h"

#include "ModelInfo.h"


ModelInfo::ModelInfo(Num xRotAxis, Num yRotAxis)
{
    this->xRotAxis = xRotAxis;
    this->yRotAxis = yRotAxis;

    this->numRings = 5;
    this->centerRadius = new Num[numRings];
    this->innerRadius = new Num[numRings];

    //Setting the size of the channels
    centerRadius[4] = 0.08879;
    centerRadius[3] = 0.06848;
    centerRadius[2] = 0.04817;
    centerRadius[1] = 0.02786;
    centerRadius[0] = 0.0;

    innerRadius[4] = 0.09895; // dummy to calculate cost
    innerRadius[3] = 0.07864;
    innerRadius[2] = 0.05833;
    innerRadius[1] = 0.03802;
    innerRadius[0] = 0.01771;

    std::vector<Num> gate0_rads = {M_PI};
    std::vector<Num> gate1_rads = {M_PI/4, M_PI/4+M_PI_2, M_PI/4+M_PI_2*2, M_PI/4+M_PI_2*3};
    std::vector<Num> gate2_rads = {0, M_PI_2, M_PI_2*2, M_PI_2*3};
    std::vector<Num> gate3_rads = {M_PI/4, M_PI/4+M_PI_2, M_PI/4+M_PI_2*2, M_PI/4+M_PI_2*3};
    listHoleRadius.push_back(gate0_rads);
    listHoleRadius.push_back(gate1_rads);
    listHoleRadius.push_back(gate2_rads);
    listHoleRadius.push_back(gate3_rads);


    // Center point of the ball if it was placed in the center of the board
    // In MuJoCo the bottom of the board has a center of (0,0,-0.015)
    // The thickness of the bottom is 0.01, so the top of the bottom is at z = -0.010
    // The radius of the ball is approx. 0.00625 (6.25mm)
    // Hence the center of the ball when placed on the bottom is at z = -0.00365
    ballCenterPtOnBoard = Vector3(0, 0, -0.00365);

    marbleNames = new std::string[4] {
        std::string{"MarbleBody1"}, std::string("MarbleBody2"),
        std::string("MarbleBody3"), std::string("MarbleBody4")};

    marbleJointNames = new std::string[4] {
            std::string{"MarbleJoint1"}, std::string("MarbleJoint2"),
            std::string("MarbleJoint3"), std::string("MarbleJoint4")};

}

std::string ModelInfo::getMarbleName(int idx) {
    return marbleNames[idx];
}

int ModelInfo::getMarbleJointId(mjModel *m_mjcModel, int idx) {
    int marbleId = mj_name2id(m_mjcModel, mjOBJ_JOINT, marbleJointNames[idx].c_str ());
    return marbleId;
}

int ModelInfo::getMarbleBodyId(mjModel *m_mjcModel, int idx) {
    int marbleId = mj_name2id(m_mjcModel, mjOBJ_BODY, marbleNames[idx].c_str ());
    return marbleId;
}

int ModelInfo::getWallBodyId(mjModel *m_mjcModel) {
    // TODO use cache
    int marbleId = mj_name2id(m_mjcModel, mjOBJ_BODY, "WallBody");
    return marbleId;
}


/*
 * returns the ring index where the position is contained
 * */
int ModelInfo::getRingIdx(const Vector2& pos) const {
    double dist = pos.norm();

    int ringIdx;

    if (dist >= innerRadius[3]) //larger than centerline distance of wall between channels 3 and 4
    {
        ringIdx = 4;
    }
    else if (dist >= innerRadius[2]) //larger than centerline distance of wall between channels 2 and 3
    {
        ringIdx = 3;
    }
    else if (dist >= innerRadius[1]) //larger than centerline distance of wall between channels 1 and 2
    {
        ringIdx = 2;
    }
    else if (dist >= innerRadius[0]) //larger than centerline distance of wall between channels 0 and 1
    {
        ringIdx = 1;
    }
    else
    {
        ringIdx = 0;
    }

    return ringIdx;
}

/*
 * calculate Quaternion represents rotation around inertia axes.
 * order Rotation X => Rotation Y
 * */
Quaternion ModelInfo::rotToQuat(Num rotX, Num rotY) const {
    using namespace std;

    Quaternion quatX {cos(rotX/2), cos(xRotAxis)*sin(rotX/2), sin(xRotAxis)*sin(rotX/2), 0};
    Quaternion quatY {0.0,cos(yRotAxis), sin(yRotAxis), 0.0};

    quatY = quatX.conjugate() * quatY * quatX;
    Quaternion  rotatedY {cos(rotY/2), sin(rotY/2) * quatY.x(), sin(rotY/2) * quatY.y(), sin(rotY/2) * quatY.z()};

    return quatX * rotatedY;
}



int ModelInfo::getRingIdx(const Vector3& pos) const {
    double dist = pos.norm();

    int ringIdx;

    if (dist >= innerRadius[3]) //larger than centerline distance of wall between channels 3 and 4
    {
        ringIdx = 4;
    }
    else if (dist >= innerRadius[2]) //larger than centerline distance of wall between channels 2 and 3
    {
        ringIdx = 3;
    }
    else if (dist >= innerRadius[1]) //larger than centerline distance of wall between channels 1 and 2
    {
        ringIdx = 2;
    }
    else if (dist >= innerRadius[0]) //larger than centerline distance of wall between channels 0 and 1
    {
        ringIdx = 1;
    }
    else
    {
        ringIdx = 0;
    }

    return ringIdx;
}



ModelInfo::~ModelInfo() {
    delete[] innerRadius;
    delete[] centerRadius;
    delete[] marbleNames;
}
