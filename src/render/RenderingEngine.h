// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef RENDERINGENGINE_H
#define RENDERINGENGINE_H

#include <string>
#include <memory>
#include <Eigen/Dense>

#include <OgreWindowEventUtilities.h>
#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreConfigFile.h>
#include <OgreMath.h>

#include "../common/type_config.h"

using ConfigMap = std::map<std::string, std::string>;

class GameState;
class Dynamics;
class ModelInfo;
class Environment;

///Rendering engine that subclasses abstract class Engine
class RenderingEngine:  public Ogre::WindowEventListener,
	public Ogre::FrameListener
{
public:
    RenderingEngine(int numBalls);

	~RenderingEngine();

	void setNrBalls (unsigned int nr_balls) { mNrBalls = nr_balls; }

    void initialize(
            const std::string& videoMode,
            const std::string& fullScreen,
            float cameraRollDegree,
            const std::string& resourcePath,
            const std::string& mazeWallsFile,
            const std::string& boardModelPath,
            const std::string& ballModelPath);

	///set up Ogre Resources
	void setupResources();
	void renderScene(GameState* state, unsigned char* RenderImgPtr);

	void setupRenderTexture();

	unsigned int getImageBufferSize() {
	    return mRenderWin_box_size;
	}

    unsigned int RenderHeight() {
        return mRenderWin_height;
    }
    unsigned int RenderWidth() {
        return mRenderWin_width;
    }

    static RenderingEngine* makeFromConfig(ConfigMap mConfig_map);

private:
    std::unique_ptr<ModelInfo> mModelInfo;
    std::unique_ptr<Environment> mEnvironment;

    bool createScene(
            float cameraRollDegree,
            const std::string& mazeWallsFile,
            const std::string& boardModelPath,
            const std::string& ballModelPath);

    bool manualInitialize(        const std::string& videoMode,
                                  const std::string& fullScreen,
                                  const std::string& desiredRenderer);
    void cameraSetUp(float cameraRollDegree);
    void lightSetUp(int number_of_lights);

    void saveRenderImage(unsigned char*);
    void updateBodyAttributes(GameState* newState);

	unsigned int mNrBalls;

    unsigned int mRenderWin_width;
    unsigned int mRenderWin_height;
    unsigned int mRenderWin_bpp;
    unsigned int mRenderWin_box_size;

	std::vector<std::string>	m_ball_entity_names;
	std::vector<std::string>	m_child_node_names;

	Ogre::Root*					mRoot;
	Ogre::String 				mResourcesCfg;
	Ogre::String 				mPluginsCfg;
	Ogre::SceneManager* 		mSceneMgr;
	Ogre::Camera* 				mCamera;
	Ogre::SceneNode** 			mOgreNodeBalls;
	Ogre::SceneNode* 			mOgreNodeBoard;
	Ogre::SceneNode* 			mOgrePlaneNode;
	Ogre::SceneNode*			mCameraNode;
	Ogre::Plane* 				mOgrePlane;
	Ogre::RenderWindow* 		mWindow;
	Ogre::RenderTexture* 		mRenderTexture;
    Ogre::LogManager*           mOgreLogger;

	Ogre::PixelFormat 			mPixelFormat;
	Ogre::TexturePtr 			mTexPtr;
};
#endif
