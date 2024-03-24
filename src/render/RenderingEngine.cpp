// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../common/stdafx.h"
#include "RenderingEngine.h"
#include "../common/HelperFunctions.h"
#include <stdio.h>
#include <iostream>
#include <string>

#include <OgreEntity.h>
#include <OgreMesh.h>
#include <OgreCamera.h>
#include <OgreFrustum.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreConfigFile.h>
#include <OgreException.h>
#include <OgreHardwarePixelBuffer.h>
#include <OgreMeshManager.h>
#include <OgreSubMesh.h>

#include <boost/pool/pool.hpp>
#include <boost/pool/object_pool.hpp>
#include <math.h>

#include <fstream>
#include <boost/algorithm/string/split.hpp>

#include "../common/GameState.h"
#include "../common/ModelInfo.h"
#include "../common/Dynamics.h"
#include "../common/Environment.h"
#include "../common/SinglePhysicsEngine.h"

#define DO_PRETTY_RENDERING 0
#define DO_GEODESIC_RENDERING 0

#define STATIC_BOARD 0 //1

static const Ogre::Quaternion default_rotation(1,0,0,0);

RenderingEngine::RenderingEngine(int numBalls)
	:mRoot(0),
    mResourcesCfg(Ogre::StringUtil::BLANK),
    mPluginsCfg(Ogre::StringUtil::BLANK),
    mSceneMgr(0),
    mCamera(0),
	mOgreNodeBoard(0),
	mOgrePlane(0),
	mOgrePlaneNode(0),
	mOgreLogger(nullptr),
	mCameraNode(0),
	mWindow(0),
	mNrBalls(numBalls)
{
	mOgreNodeBalls = NULL;

	m_ball_entity_names.push_back("Game Marble entity1");
	m_ball_entity_names.push_back("Game Marble entity2");
	m_ball_entity_names.push_back("Game Marble entity3");
	m_ball_entity_names.push_back("Game Marble entity4");

	m_child_node_names.push_back("Marble node1");
	m_child_node_names.push_back("Marble node2");
	m_child_node_names.push_back("Marble node3");
	m_child_node_names.push_back("Marble node4");
}

RenderingEngine* RenderingEngine::makeFromConfig(ConfigMap mConfig_map) {
    std::string::size_type sz;

    int numBalls = std::stoi(mConfig_map["Nr_Of_Balls"]);
    const std::string resourcePath = mConfig_map["Resource_Path"];
    const std::string boardModelPath = HelperFunctions::STRExtendWithModelDirpath(mConfig_map["OgreModel1"]);
    const std::string ballModelPath = HelperFunctions::STRExtendWithModelDirpath(mConfig_map["OgreModel2"]);
    std::string mazeWallsFile = "";

    /*
     * Now we removes this feature. Because the shown image is destructed!
     *
    if (mConfig_map.find("MazeWallsFile") != mConfig_map.end())
    {
        mazeWallsFile = mConfig_map["MazeWallsFile"];
    }
     */

    Num xAxisAngleRad = degToRad(std::stod(mConfig_map["XAxisAngleDeg"], &sz));
    Num yAxisAngleRad = degToRad(std::stod(mConfig_map["YAxisAngleDeg"], &sz));


    ModelInfo* modelInfo = new ModelInfo(xAxisAngleRad, yAxisAngleRad);
    std::unique_ptr<SinglePhysicsEngine> physics {SinglePhysicsEngine::makeFromConfig(mConfig_map)};
    Environment* environment = Environment::makeFromConfig(mConfig_map, physics.get());

    const std::string videoMode = mConfig_map["Video_Mode"];
    const std::string fullScreen = mConfig_map["Full_Screen"];
    float inPlaneRotation = std::stof(mConfig_map["In_Plane_Rotation"],&sz);

    RenderingEngine* renderer = new RenderingEngine(numBalls);
    renderer->mModelInfo.reset(modelInfo);
    renderer->mEnvironment.reset(environment);

    renderer->initialize(
            videoMode,
            fullScreen,
            inPlaneRotation,
            resourcePath,
            mazeWallsFile,
            boardModelPath,
            ballModelPath
    );
    return renderer;
}



RenderingEngine::~RenderingEngine()
{
	if (mOgrePlane) delete mOgrePlane;

	mRoot->endRenderingQueued();

	mRoot->destroyAllRenderQueueInvocationSequences();
	mRoot->removeFrameListener(this);

	//OgreNodeList.clear ();

	for (unsigned int i=0; i < mNrBalls; ++i) mOgreNodeBalls[i]->detachAllObjects ();
	mOgrePlaneNode->detachAllObjects ();
	mOgreNodeBoard->detachAllObjects ();

	mSceneMgr->destroyAllCameras ();
	mSceneMgr->destroyAllLights ();
	mSceneMgr->destroyAllEntities ();

	// this causes crash somehow, not sure why
	//mRoot->destroySceneManager(mSceneMgr);
	//if (mRoot) delete mRoot;

	if (mOgreNodeBalls) delete [] mOgreNodeBalls;

	if (mOgreLogger) {
	    delete mOgreLogger;
	    mOgreLogger = nullptr;
	}
}

//Set path to Resources.cfg file
void RenderingEngine::initialize(
        const std::string& videoMode,
        const std::string& fullScreen,
        float cameraRollDegree,
        const std::string& resourcePath,
        const std::string& mazeWallsFile,
        const std::string& boardModelPath,
        const std::string& ballModelPath)
{
    mResourcesCfg = resourcePath + "resources.cfg";
    mPluginsCfg = resourcePath + "plugins.cfg";
    //Ogre::LogManager::getSingleton().setLogDetail(static_cast<Ogre::LoggingLevel>(0));

    // suppress Ogre Log.
    // https://forums.ogre3d.org/viewtopic.php?t=66641
    mOgreLogger = new Ogre::LogManager();
    mOgreLogger->createLog("ogre_log.log", false, false, true);
    Ogre::String logFileName = "";
    Ogre::String configFileName = "";
    mRoot = new Ogre::Root(mPluginsCfg, configFileName, logFileName);
    // Ogre::LogManager::getSingleton().setLogDetail(static_cast<Ogre::LoggingLevel>(0));
    setupResources();

    if(manualInitialize(videoMode, fullScreen, "OpenGL")) {
        mWindow = mRoot->initialise(true);
        mWindow->setHidden(true);
        // https://stackoverflow.com/questions/2397160/how-to-get-a-windowless-application-in-ogre
        Ogre::WindowEventUtilities::messagePump();
    }

    createScene(cameraRollDegree, mazeWallsFile, boardModelPath, ballModelPath);
}


void RenderingEngine::setupResources(){
	Ogre::ConfigFile cf;
	cf.load(mResourcesCfg);

	// Go through all sections & settings in the file
    Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();

    Ogre::String secName, typeName, archName;
    while (seci.hasMoreElements())
    {
        secName = seci.peekNextKey();
        Ogre::ConfigFile::SettingsMultiMap *settings = seci.getNext();
        Ogre::ConfigFile::SettingsMultiMap::iterator i;
        for (i = settings->begin(); i != settings->end(); ++i)
        {
            typeName = i->first;
            archName = i->second;

#if OGRE_PLATFORM == OGRE_PLATFORM_APPLE
            // OS X does not set the working directory relative to the app.
            // In order to make things portable on OS X we need to provide
            // the loading with it's own bundle path location.
            if (!Ogre::StringUtil::startsWith(archName, "/", false)) // only adjust relative directories
                archName = Ogre::String(Ogre::macBundlePath() + "/" + archName);
#endif

            Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
                archName, typeName, secName);
        }
    }
}

void RenderingEngine::setupRenderTexture()
{
	unsigned int width, height, depth;
	int left, top;
	//Get render window parameters
	mWindow->getMetrics(width, height, depth, left, top);
	mTexPtr = Ogre::TextureManager::getSingleton().createManual(
		"MainRenderTarget",
		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
		Ogre::TEX_TYPE_2D,
		width,
		height,
		depth,
		0,
		Ogre::PF_R8G8B8,
		Ogre::TU_RENDERTARGET);

	mRenderTexture = mTexPtr->getBuffer()->getRenderTarget();
	Ogre::Viewport* vp = mRenderTexture->addViewport(mCamera);
	vp->setClearEveryFrame(true);
	vp->setBackgroundColour(Ogre::ColourValue::Black);
	vp->setOverlaysEnabled(false);
	vp->update();
	mPixelFormat = mRenderTexture->suggestPixelFormat();

    mCamera->setAspectRatio(
            Ogre::Real(vp->getActualWidth()) /
            Ogre::Real(vp->getActualHeight()));

}

void RenderingEngine::lightSetUp(int light_num)
{
	Ogre::Real l_x = 200;
	Ogre::Real l_y = 500; //340; //50;
	Ogre::Real light_range = 5625;

	Ogre::Light* Mainlight = mSceneMgr->createLight("MainLight");
	Mainlight->setType(Ogre::Light::LT_SPOTLIGHT);
	//Mainlight->setType(Ogre::Light::LT_DIRECTIONAL);
	Mainlight->setPosition(0, l_y, 0);
	Mainlight->setDiffuseColour(0.43, 0.43, 0.48);
	Mainlight->setSpecularColour(1.0, 1.0, 1.0);
	Mainlight->setDirection(Ogre::Vector3(0, -l_y, 0));
	Mainlight->setSpotlightRange(Ogre::Degree(200), Ogre::Degree(250), 10.5);
	Mainlight->setAttenuation(light_range, 1.0, 0.0, 0.0);

	int light_num_tmp = light_num - 1;

	//For additional 4 lights projecting light in the XZ plane
	if (light_num_tmp > 0)
	{
		Ogre::Light* Sidelight1 = mSceneMgr->createLight("Sidelight1");
		Sidelight1->setPosition(l_x, l_y, 0);
		Sidelight1->setType(Ogre::Light::LT_SPOTLIGHT);
		Sidelight1->setDiffuseColour(0.15, 0.15, 0.15);
		Sidelight1->setSpecularColour(0.0, 0.0, 0.0);
		Sidelight1->setDirection(Ogre::Vector3(-l_x, -l_y, 0));
		Sidelight1->setSpotlightRange(Ogre::Degree(50), Ogre::Degree(150), 10.5);
		Sidelight1->setAttenuation(light_range, 1.0, 0.0014, 0.000007);
		light_num_tmp--;
	}
	else return;

	if (light_num_tmp > 0)
	{
		Ogre::Light* Sidelight2 = mSceneMgr->createLight("Sidelight2");
		Sidelight2->setPosition(-l_x, l_y, 0);
		Sidelight2->setType(Ogre::Light::LT_SPOTLIGHT);
		Sidelight2->setDiffuseColour(0.15, 0.15, 0.15);
		Sidelight2->setSpecularColour(0.0, 0.0, 0.0);
		Sidelight2->setDirection(Ogre::Vector3(l_x, -l_y, 0));
		Sidelight2->setSpotlightRange(Ogre::Degree(50), Ogre::Degree(150), 10.5);
		Sidelight2->setAttenuation(light_range,1.0,0.0014,0.000007);
		light_num_tmp--;
	}
	else return;

	if (light_num_tmp > 0)
	{
		Ogre::Light* Sidelight3 = mSceneMgr->createLight("Sidelight3");
		Sidelight3->setPosition(0, l_y, l_x);
		Sidelight3->setType(Ogre::Light::LT_SPOTLIGHT);
		Sidelight3->setDiffuseColour(1.0, 1.0, 1.0);
		Sidelight3->setSpecularColour(0.0, 0.0, 0.0);
		Sidelight3->setDirection(Ogre::Vector3(0, -l_y, -l_x));
		Sidelight3->setSpotlightRange(Ogre::Degree(50), Ogre::Degree(150), 10.5);
		Sidelight3->setAttenuation(light_range,1.0,0.0014,0.000007);
		light_num_tmp--;
	}
	else return;

	if (light_num_tmp > 0)
	{
		Ogre::Light* Sidelight4 = mSceneMgr->createLight("Sidelight4");
		Sidelight4->setPosition(0, l_y, -l_x);
		Sidelight4->setType(Ogre::Light::LT_SPOTLIGHT);
		Sidelight4->setDiffuseColour(1.0, 0.0, 1.0);
		Sidelight4->setSpecularColour(0.0, 0.0, 0.0);
		Sidelight4->setDirection(Ogre::Vector3(0, -l_y, l_x));
		Sidelight4->setSpotlightRange(Ogre::Degree(50), Ogre::Degree(150), 10.5);
		Sidelight4->setAttenuation(light_range,1.0,0.0014,0.000007);
	}
}

void RenderingEngine::cameraSetUp(float cameraRollDegree)
{
	mCamera = mSceneMgr->createCamera("MainCam");
	mCameraNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("CamNode1", Ogre::Vector3(0, 0, 0));

#if DO_GEODESIC_RENDERING
	mCameraNode->pitch(Ogre::Degree(270));
	mCameraNode->roll(Ogre::Degree(m_mediator->mInPlaneRotation));
	mCameraNode->setPosition(Ogre::Vector3(0, 125, 0));
	mCameraNode->attachObject(mCamera);

	std::cout << "Clip distances (near, far): " << mCamera->getNearClipDistance () << ", " << mCamera->getFarClipDistance () << std::endl;
	mCamera->setNearClipDistance (Ogre::Real (50.0));
#else
#if DO_PRETTY_RENDERING
	// pretty rendering only
	mCamera->setPosition(Ogre::Vector3(100, 350, 100));
	mCamera->lookAt (Ogre::Vector3(0, 0, 0));
	mCameraNode->attachObject(mCamera);
#else
	mCameraNode->pitch(Ogre::Degree(270));
	mCameraNode->roll(Ogre::Degree(cameraRollDegree));
	mCameraNode->setPosition(Ogre::Vector3(0, 350, 0));
	mCameraNode->attachObject(mCamera);
#endif
#endif

	// Set up viewport
	/*
	Ogre::Viewport* vp = mWindow->addViewport(mCamera);
	vp->setBackgroundColour(Ogre::ColourValue(0, 0, 0));

	 */

#if DO_GEODESIC_RENDERING
	mCamera->setProjectionType (Ogre::PT_ORTHOGRAPHIC);

	mCamera->setFOVy (Ogre::Radian (Ogre::Degree(80.0).valueRadians ()));
	const Ogre::Radian& fov_rad = mCamera->getFOVy();
	std::cout << "Camera FOV y (degrees): " << fov_rad.valueDegrees () << std::endl;

	std::cout << "Viewport (w, h): " << vp->getActualWidth () << ", " << vp->getActualHeight () << std::endl;

	// mCamera->setOrthoWindow (
	// 	Ogre::Real(vp->getActualWidth()),
	// 	Ogre::Real(vp->getActualHeight()));
#endif
}



//Create Ogre nodes, entities for the scene
// check if user specified a maze wall configuration file
// if so, then select only the walls according to the configuration file
// otherwise load the full model
bool RenderingEngine::createScene(
        float cameraRollDegree,
        const std::string& mazeWallsFile,
        const std::string& boardModelPath,
        const std::string& ballModelPath)
{

	Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);
	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();
	mSceneMgr = mRoot->createSceneManager(Ogre::ST_GENERIC);

#if DO_GEODESIC_RENDERING
	mSceneMgr->setShadowTechnique(Ogre::SHADOWTYPE_NONE);
#else
	mSceneMgr->setShadowTechnique(Ogre::SHADOWTYPE_STENCIL_ADDITIVE);
#endif

	mSceneMgr->setAmbientLight(Ogre::ColourValue(0.45, 0.45, 0.45));
	mSceneMgr->setShadowColour(Ogre::ColourValue(0.01,0.01,0.01));
	mSceneMgr->setShadowFarDistance(625.0);

	//Set up camera
	cameraSetUp(cameraRollDegree);

	Ogre::Entity* ogreEntityBoard;

	std::map<std::string, std::string> walls_config_map;

	if (HelperFunctions::readConfig(mazeWallsFile.c_str(), walls_config_map))
	{
		//Load and set up meshes
		Ogre::Entity* ogreEntityBoard_tmp = mSceneMgr->createEntity("Game Board entity", boardModelPath);
		ogreEntityBoard_tmp->setCastShadows(true);

		// Get the associated mesh
		const Ogre::MeshPtr& mesh_ptr = ogreEntityBoard_tmp->getMesh ();

		// get iterator over the submeshes of the mesh
		Ogre::Mesh::SubMeshIterator submesh_iter = mesh_ptr->getSubMeshIterator ();

		// get the name map of the submeshes
		// NOTE: the order of the names of submeshes, does not necessarily match the order
		// of the submeshes in the mesh!!!! The name map has a second entry which is an index
		// that index corresponds to the index of the submesh in the mesh
		const Ogre::Mesh::SubMeshNameMap& submesh_namemap = mesh_ptr->getSubMeshNameMap ();

		Ogre::MeshPtr new_mesh = Ogre::MeshManager::getSingleton ().createManual ("Game Board assembly", "General");

		unsigned int idx = 0;

		while (submesh_iter.hasMoreElements ())
		{
			Ogre::SubMesh* submesh = submesh_iter.getNext ();

			// find the name, by checking the index of the submesh
			auto iter = submesh_namemap.begin();

			while (iter->second != idx) iter++;

			//std::cout << iter->first << ", " << iter->second << std::endl;

			// now check if the name of the submesh is enabled in the walls config file
			if (walls_config_map.find(iter->first) != walls_config_map.end ())
			{
				std::string setting = walls_config_map[iter->first];
				if (setting.compare("yes") == 0)
				{
					//std::cout << "Wall " << iter->first << " selected by user, generating submesh" << std::endl;

					Ogre::SubMesh* new_submesh = new_mesh->createSubMesh (iter->first);

					*new_submesh = *submesh;
				}
				else
				{
					//std::cout << "Wall " << iter->first << " not selected by user." << std::endl;
				}
			}

			idx++;
		}

		new_mesh->_setBounds (mesh_ptr->getBounds ());
		new_mesh->_setBoundingSphereRadius (mesh_ptr->getBoundingSphereRadius ());
		new_mesh->load ();

		ogreEntityBoard = mSceneMgr->createEntity("Game Board", "Game Board assembly");
		ogreEntityBoard->setCastShadows(true);

		mSceneMgr->destroyEntity (ogreEntityBoard_tmp);
	}
	else
	{
		// No walls config file was found, so we simply load the entire mesh + submeshes
		ogreEntityBoard = mSceneMgr->createEntity("Game Board entity", boardModelPath);
		ogreEntityBoard->setCastShadows(true);
	}


#if DO_GEODESIC_RENDERING
	const int plane_size = 1200;
#else
	const int plane_size = 200; //300;
#endif

	mOgrePlane = new Ogre::Plane(Ogre::Vector3::UNIT_Y, -10);//-0.157, 1, 0
	Ogre::MeshManager::getSingleton().createPlane(
		"PlaneMesh",
		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
		*mOgrePlane,
		plane_size, plane_size, 10, 10,
		true,
		1, 1, 1,
		Ogre::Vector3::UNIT_Z);
	Ogre::Entity* mPlaneEntity = mSceneMgr->createEntity("PlaneMesh");
	mPlaneEntity->setCastShadows(false);

	mOgreNodeBoard = mSceneMgr->getRootSceneNode()->createChildSceneNode("Board node");
	mOgrePlaneNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("Plane Node");

	//std::cout << "Nr of Balls to render: " << mNrBalls << std::endl;

	mOgreNodeBalls = new Ogre::SceneNode*[mNrBalls];

	for (unsigned int i = 0; i < mNrBalls; ++i)
	{
		Ogre::Entity* ogreEntityBall = mSceneMgr->createEntity(m_ball_entity_names[i], ballModelPath);
		ogreEntityBall->setCastShadows(true);
		// ogreEntityBall->setVisible(false);

		ogreEntityBall->setMaterialName("Ball/NoMaterial");

		mOgreNodeBalls[i] = mSceneMgr->getRootSceneNode()->createChildSceneNode(m_child_node_names[i]);

		mOgreNodeBalls[i]->setPosition(0,0,0);

#if DO_GEODESIC_RENDERING
		mOgreNodeBalls[i]->scale (Ogre::Real(2.0), Ogre::Real(2.0), Ogre::Real(2.0));
#else
		mOgreNodeBalls[i]->attachObject(ogreEntityBall);
#endif

		//OgreNodeList.push_back(mOgreNodeBalls[i]);
	}

	//Setting positions the same as the models in Mujoco
	mOgreNodeBoard->setPosition(0,0,0);
#if DO_GEODESIC_RENDERING
	mOgreNodeBoard->setScale(0.04f, 0.04f, 0.04f);
	//mOgrePlaneNode->scale (Ogre::Real(2.0), Ogre::Real(2.0), Ogre::Real(2.0));
#else
	std::string substr("GameBoard_FinalExport");

	// different member versions of find in the same order as above:
	std::size_t found = boardModelPath.find(substr);
	if (found!=std::string::npos)
	{
		// std::cout << "apply scaling for original board" << std::endl;
		mOgreNodeBoard->setScale(0.01f, 0.01f, 0.01f);
	}
	else
	{
		// std::cout << "apply scaling for modified board" << std::endl;
		mOgreNodeBoard->setScale(10.f, 10.f, 10.f);
	}
#endif
	mOgrePlaneNode->setPosition(0,0,0);


    mOgreNodeBoard->attachObject(ogreEntityBoard);
	mOgrePlaneNode->attachObject(mPlaneEntity);

#if DO_GEODESIC_RENDERING
	ogreEntityBoard->setMaterialName("MatteOldLaceBoardTmp");
#else
	ogreEntityBoard->setMaterialName("MatteOldLaceBoard2");
#endif

	//createCircularPlane ();

	//Adding texture to the ground plane
#if DO_GEODESIC_RENDERING
	mPlaneEntity->setMaterialName("MatteOldLaceGroundTmp");
#else
	mPlaneEntity->setMaterialName("MatteOldLaceGround");
#endif

	//Light
	int light_num = 1; //3;
	lightSetUp(light_num);

	Ogre::WindowEventUtilities::addWindowEventListener(mWindow, this);

    setupRenderTexture();

	//Render window parameters
	mRenderWin_width = mRenderTexture->getViewport(0)->getActualWidth();
	mRenderWin_height = mRenderTexture->getViewport(0)->getActualHeight();
	mRenderWin_bpp = static_cast<unsigned int>(Ogre::PixelUtil::getNumElemBytes(mPixelFormat));
	mRenderWin_box_size = mRenderWin_width * mRenderWin_height * mRenderWin_bpp;

#ifdef _DEBUG
	printf("render window size: %d, %d, %d\n", mRenderWin_width, mRenderWin_height, mRenderWin_bpp);
#endif
	mRoot->addFrameListener(this);
	return true;
}

//Manual initialization
bool RenderingEngine::manualInitialize(
        const std::string& videoMode,
        const std::string& fullScreen,
        const std::string& desiredRenderer)
{
    Ogre::RenderSystem *renderSystem;
    bool ok = false;
    const Ogre::RenderSystemList &renderers =
        Ogre::Root::getSingleton().getAvailableRenderers();

    // See if the list is empty (no renderers available)
    if(renderers.empty())
        return false;

    for(Ogre::RenderSystemList::const_iterator it = renderers.begin();
        it != renderers.end(); it++)
    {
        renderSystem = (*it);
        if(strstr(renderSystem->getName().c_str (), desiredRenderer.c_str ()))
        {
            ok = true;
            break;
        }
    }

    if(!ok) {
        // We still don't have a renderer; pick
        // up the first one from the list
        renderSystem = (*renderers.begin());
    }

    Ogre::Root::getSingleton().setRenderSystem(renderSystem);

    // Manually set some configuration options (optional)
	renderSystem->setConfigOption("Full Screen",fullScreen);
	renderSystem->setConfigOption("Video Mode", videoMode);

    //std::cout << "Ogre current rendering configuration: " << std::endl;

    return true;
}


//Start Ogre rendering pipeline
void RenderingEngine::renderScene(GameState* state, unsigned char* RenderImgPtr){
    updateBodyAttributes(state);
	mRoot->renderOneFrame();
	//Put image data in the pointer
	saveRenderImage(RenderImgPtr);
}

//Save the rendered image as a string and return pointer to it
void RenderingEngine::saveRenderImage(unsigned char* RenderImgPtr)
{
    //Create pixelbox with the dimensions of the render window
    Ogre::PixelBox pixelbox (mRenderWin_width, mRenderWin_height, 1, mPixelFormat, RenderImgPtr);
    mRenderTexture->copyContentsToMemory (pixelbox,Ogre::RenderTarget::FB_FRONT);
}


//Set ball and board positions using data from Mujoco
void RenderingEngine::updateBodyAttributes(GameState* newState)
{
	std::vector<Ogre::Vector3> dirVector;

	Ogre::Vector3 dirVec(0,0,0);

	//Get the update from the physics engine

	// synchronize ball positions
	for (int i = 0; i < mNrBalls; i++){
        Ogre::SceneNode* node = mOgreNodeBalls[i];
        Vector3 mjPos = newState->ballPos(i);
        Vector3 pos = CoordMJC2Ogre(mjPos);

        dirVec.x = 1000*pos[0];//x coordinate
        dirVec.y = 1000*pos[1];//y coordinate
        dirVec.z = 1000*pos[2];//z coordinate
        node->setPosition(dirVec);

        // ignore ball rotation.
        //Quaternion rot = QuatMJC2Ogre(state.rot);
        Ogre::Real fw = 1;
        Ogre::Real fx = 0;
        Ogre::Real fy = 0;
        Ogre::Real fz = 0;

        Ogre::Quaternion quat(fw, fx, fy, fz);
        node->setOrientation(quat);
	}

	// synchronize board position.
	Ogre::SceneNode* nodes[2] = {mOgreNodeBoard, mOgrePlaneNode};

    for (int i = 0; i < 2; i++) {
        Ogre::SceneNode* node = nodes[i];
        Quaternion  boardQuat = newState->boardQuaternion(*mModelInfo.get());

        Quaternion rot = QuatMJC2Ogre(boardQuat);
        Vector3 pos = CoordMJC2Ogre(newState->boardPos());

        dirVec.x = 1000 * pos[0];//x coordinate
        dirVec.y = 1000 * pos[1];//y coordinate
        dirVec.z = 1000 * pos[2];//z coordinate
        node->setPosition(dirVec);

        Ogre::Real fw = rot.w();
        Ogre::Real fx = rot.x();
        Ogre::Real fy = rot.y();
        Ogre::Real fz = rot.z();

        Ogre::Quaternion quat(fw, fx, fy, fz);
        node->setOrientation(quat);
    }

}
