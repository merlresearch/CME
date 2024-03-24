// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef BIMGAME_TYPE_CONFIG_H
#define BIMGAME_TYPE_CONFIG_H

#include <Eigen/Dense>

/*
 * define inner types
 * */
using Vector3 = Eigen::Vector3d;
using Vector2 = Eigen::Vector2d;
using Quaternion = Eigen::Quaterniond;
using Num = double;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;
using AngleAxis = Eigen::AngleAxisd;


#endif //BIMGAME_TYPE_CONFIG_H
