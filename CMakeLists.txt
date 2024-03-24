# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)


project(MazeSimulator)

set(TEST_TARGET cpptest)
set(COMMON_LIB_TARGET common_ops)
set(RENDER_LIB_TARGET render_ops)

set(MUJOCO_HOME $ENV{HOME}/.mujoco/mujoco200 CACHE PATH "")
set(MUJOCO_INCLUDE_DIR ${MUJOCO_HOME}/include CACHE PATH "")
set(MUJOCO_LIBRARY ${MUJOCO_HOME}/bin/libmujoco200.so CACHE FILEPATH "")

find_package(Boost REQUIRED COMPONENTS filesystem system thread chrono)

set(OGRE_LIB "/usr/local/lib/OGRE" CACHE PATH "")
set(OGRE_INCLUDE "/usr/local/include/OGRE" CACHE PATH "")
set(OGRE_SHARE "/usr/local/share/OGRE" CACHE PATH "")
set(CMAKE_MODULE_PATH ${OGRE_LIB}/cmake )


# REF: https://forums.ogre3d.org/viewtopic.php?t=56723
FIND_PACKAGE(OGRE REQUIRED)

FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)

# REF: https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package(Eigen3 REQUIRED NO_MODULE)

file(GLOB COMMON_SOURCE_FILES "src/common/*.cpp")
file(GLOB COMMON_HEADER_FILES "src/common/*.h")


# REF: https://stackoverflow.com/questions/10851247/how-to-activate-c-11-in-cmake
# REF: https://github.com/facebook/Surround360/issues/3
set(CMAKE_VERBOSE_MAKEFILE  1)
SET(CMAKE_CXX_FLAGS "-std=c++11 -pthread -fPIC -D_GLIBCXX_USE_CXX11_ABI=1")


add_executable(${TEST_TARGET}
        src/cpptest/cpptest.cpp
        ${COMMON_SOURCE_FILES} ${COMMON_HEADER_FILES} src/common/type_config.h
        src/common/Dynamics.h src/common/Dynamics.cpp src/common/WallRemover.cpp src/common/CompositePhysicsEngine.h src/common/CompositePhysicsEngine.cpp)

target_link_libraries(
        ${TEST_TARGET}
        PUBLIC
        pthread
        libgtest.a
        libgtest_main.a
        ${Boost_LIBRARIES}
        ${MUJOCO_LIBRARY}
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
)

target_include_directories(
        ${TEST_TARGET}
        PRIVATE
        ${OPENGL_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${MUJOCO_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenGL_INCLUDE_DIR}
        ${BOOST_INCLUDE_DIRS}
)

add_library(
        ${COMMON_LIB_TARGET}
        SHARED
        ${COMMON_SOURCE_FILES} ${COMMON_HEADER_FILES} src/CythonEnv/CyInterface.cpp src/CythonEnv/CyInterface.h
        src/common/Dynamics.h src/common/Dynamics.cpp src/common/ThetaDynamics.cpp src/common/XYZDynamics.cpp src/common/WallRemover.cpp src/common/CompositePhysicsEngine.h src/common/CompositePhysicsEngine.cpp src/common/WallRemover.h)

target_include_directories(
        ${COMMON_LIB_TARGET}
        PUBLIC
        ${OPENGL_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${MUJOCO_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenGL_INCLUDE_DIR}
        ${BOOST_INCLUDE_DIRS}
)

target_link_libraries(
        ${COMMON_LIB_TARGET}
        PUBLIC
        ${Boost_LIBRARIES}
        ${MUJOCO_LIBRARY}
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES})


add_library(
        ${RENDER_LIB_TARGET}
        SHARED
        ${COMMON_SOURCE_FILES} ${COMMON_HEADER_FILES} src/render/RenderingEngine.cpp src/render/RenderingEngine.h src/common/type_config.h
        src/CythonEnv/RenderInterface.h src/CythonEnv/RenderInterface.cpp)

target_include_directories(
        ${RENDER_LIB_TARGET}
        PUBLIC
        ${OPENGL_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${MUJOCO_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenGL_INCLUDE_DIR}
        ${BOOST_INCLUDE_DIRS}
        ${OGRE_INCLUDE}
        ${OGRE_INCLUDE}/Overlay
)

target_link_libraries(
        ${RENDER_LIB_TARGET}
        PUBLIC
        ${Boost_LIBRARIES}
        ${MUJOCO_LIBRARY}
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${OGRE_LIBRARIES}
)


# CAUTION! This target doesn't built automatically after pyx files are changed.
add_custom_target(
        NAME ALL
        DEPENDS ${COMMON_LIB_TARGET} ${RENDER_LIB_TARGET}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/CythonEnv
        COMMAND python setup.py build --build-lib ${CMAKE_CURRENT_BINARY_DIR} --user-lib ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "build Cython artifacts.")

install(
    TARGETS ${RENDER_LIB_TARGET} ${COMMON_LIB_TARGET}
    LIBRARY DESTINATION "./"
    RUNTIME DESTINATION "./"
)

install(
        CODE "file( GLOB GeneratedCython \"${CMAKE_CURRENT_BINARY_DIR}/Cy*.so\" )"
        CODE "file( COPY \${GeneratedCython} DESTINATION \"${CMAKE_CURRENT_SOURCE_DIR}/CythonEnv\" )"
)

configure_file("setup_env.sh.in" "setup_env.sh")

install(
        FILES
        "${CMAKE_BINARY_DIR}/setup_env.sh"
        DESTINATION "./"
)

add_subdirectory(data)
add_subdirectory(experiments)
