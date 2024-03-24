// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H


#include <string>
#include <iostream>
#include <map>
#include <fstream>
#include <boost/filesystem.hpp>
#include <Eigen/Geometry>
#include <chrono>

#include "type_config.h"

using ConfigMap = std::map<std::string, std::string>;

class HelperFunctions
{
public:
	static std::string STRExtendWithModelDirpath (const std::string&);
	static void GetCurrentWorkingDir (std::string&);
	static bool readConfig(const char*, std::map<std::string, std::string>&);
};

class StopWatch {
public:
    StopWatch(std::string name) : name(name){
        typedef std::chrono::high_resolution_clock Clock;
        start_time = Clock::now();
    }
    void start() {
        typedef std::chrono::high_resolution_clock Clock;
        start_time = Clock::now();
    }

    void stop() {
        typedef std::chrono::high_resolution_clock Clock;
        finish_time = Clock::now();
    }

    void print() {
        auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(finish_time - start_time).count();
        std::cout << name << " : " << d2 << "us" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::time_point<std::chrono::system_clock> finish_time;
    std::string name;
};


inline std::string HelperFunctions::STRExtendWithModelDirpath(const std::string& strPath)
{
	boost::filesystem::path path(strPath);
#if 0 // TMP-MOD
	if( path.is_absolute() )
		return path.string();

    boost::filesystem::path strCurrentPath(boost::filesystem::current_path());

    const char* szModelDirpath = std::getenv(ENV_VAR_MODEL_DIRPATH);
    if( NULL == szModelDirpath )
    {
    	szModelDirpath = strCurrentPath.c_str();
    }

    std::string strModelDirpath(szModelDirpath);
    boost::filesystem::path pathExtendedFilepath(strModelDirpath);
    pathExtendedFilepath /= path;
    return pathExtendedFilepath.string();
#else
		return path.string();
    #endif
}

inline void HelperFunctions::GetCurrentWorkingDir( std::string& current_working_dir )
{
  char buff[FILENAME_MAX];
  char* curr_wd = getcwd( buff, FILENAME_MAX );
  std::string cwd(buff);
  current_working_dir = cwd;
}

inline bool HelperFunctions::readConfig(const char* config_filename, std::map<std::string, std::string>& config_map)
{
	bool result = false;

	std::string config_file;
	config_file.assign (config_filename);

	std::ifstream is_file;
	is_file.open(config_file);
	if (is_file.is_open())
	{
		std::string line;
		while( std::getline(is_file, line) )
		{
			std::istringstream is_line(line);
			std::string key;
			if( std::getline(is_line, key, '='))
			{
				std::string value;
				if( std::getline(is_line, value) )
				{
					config_map[key] = value;
				}
			}
		}
		is_file.close();

		result = true;
	}

	return result;
}


/*
 * Translate quartenion systems
 * */
inline Quaternion QuatMJC2Ogre(Quaternion mjcQuat)
{
    Matrix4 permutationMatrix;
    permutationMatrix << 1, 0,0,0,
            0, 0,1,0,
            0,-1,0,0,
            0, 0,0,1;
    Matrix4 rotationMatrix;
    rotationMatrix << 0,0,0,0,
            0,0,0,0,
            0,0,0,0,
            0,0,0,1;

    Matrix4 intermediateResult1;
    Matrix4 InversePermMatrix;

    InversePermMatrix = permutationMatrix.inverse();
    mjcQuat = mjcQuat.normalized();
    //Convert the Ogre_quat to Rot_Mat_Ogre
    rotationMatrix.block<3,3>(0,0) = mjcQuat.toRotationMatrix();

    intermediateResult1 = permutationMatrix * rotationMatrix;
    //Multiply by the inverse of permutation matrix to convert back to Ogre coordinate system
    intermediateResult1 = intermediateResult1 * InversePermMatrix;
    //Throw away the scale information
    Quaternion out_quat;
    out_quat = intermediateResult1.block<3,3>(0,0);

    return out_quat;
}


/*
 * Translate coordinate system
 * Use this function to convert velocities, accelerations, and positions.
 * */
inline Vector3 CoordMJC2Ogre(Vector3 mjcCoord)
{
    Vector3 ogreCoord;
    ogreCoord[0] = mjcCoord[0];
    ogreCoord[1] = mjcCoord[2];
    ogreCoord[2] = -mjcCoord[1];
    return ogreCoord;
}

inline Num degToRad(Num deg) {
    return (deg / 180) * M_PI;
}

/*
 * calculate Quaternion represents rotation around inertia axes.
 * order Rotation X => Rotation Y
 * */
inline Quaternion rotToQuat(Num rotX, Num rotY) {
    using namespace std;

    Quaternion quatX {cos(rotX/2), 1*sin(rotX/2), 0, 0};
    Quaternion quatY {0.0, 0.0, 1.0, 0.0};

    quatY = quatX.conjugate() * quatY * quatX;
    Quaternion  rotatedY {cos(rotY/2), sin(rotY/2) * quatY.x(), sin(rotY/2) * quatY.y(), sin(rotY/2) * quatY.z()};

    return quatX * rotatedY;
}


#endif
