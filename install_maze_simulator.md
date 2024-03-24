<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Installation of MAZE simulator

## Install python libraries

- Create virtual env using Anaconda

```bash
# Only python 3.6 is allowed because of compatibility to pyrealsense
$ conda create -n maze python=3.6 anaconda
$ conda activate maze
```

- Install libraries

```bash
$ conda install scipy scikit-learn pyqtgraph
$ pip install -r requirements.txt
```

## Install prequisites

- Install dependencies

```bash
$ sudo apt-get install cmake cmake-curses-gui
$ sudo apt-get install libxaw7-dev libxt-dev
$ sudo apt-get install freeglut3-dev
$ sudo apt-get install libfreetype6-dev libxrandr-dev
$ sudo apt-get install --no-install-recommends libboost-all-dev
$ sudo apt-get install zlib1g-dev libfreeimage-dev libois-dev libtinyxml-dev libzzip-dev libcppunit-dev libglew-dev libdevil-dev
$ sudo apt-get install libeigen3-dev
```

- Install googletest

```bash
$ wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz
$ tar xvf release-1.8.1.tar.gz
$ cd googletest-release-1.8.1/
$ mkdir build
$ cd build
$ cmake ..
$ sudo make install
```

## Install OGRE and MuJoCo

### OGRE

```bash
$ mkdir -p ~/workspace/env/ogre; cd $_
$ wget https://bitbucket.org/sinbad/ogre/get/v1-9-0RC2.tar.gz
$ tar zxf sinbad-ogre-4c20a40dae61.tar.gz
$ cd sinbad-ogre-4c20a40dae61/
$ mkdir build; cd $_
$ cmake \
-DFREETYPE_INCLUDE_DIR=/usr/include/freetype2 \
-DFREETYPE_FT2BUILD_INCLUDE_DIR=/usr/include/freetype2 \
-DOGRE_CONFIG_THREADS=0 \
..
$ sudo make -j8 install
```

- NOTE: There could be an issue related to `Ogre::ProgressiveMeshGenerator::addIndexDataImpl<unsigned short>()`, for
  instance:

```bash
$ sudo make -j8 install
...
../../lib/libOgreMain.so.1.9.0: undefined reference to `voidOgre::ProgressiveMeshGenerator::addIndexDataImpl<unsigned short>(unsignedshort*, unsigned short const*,std::vector<Ogre::ProgressiveMeshGenerator::PMVertex*,Ogre::STLAllocator<Ogre::ProgressiveMeshGenerator::PMVertex*,Ogre::CategorisedAllocPolicy<(Ogre::MemoryCategory)0> > >&, unsignedshort)'
collect2: error: ld returned 1 exit status
...
```

- A workaround is modifying `OgreMain/src/OgreProgressiveMeshGenerator.cpp` as follows. Then make again.

```cpp
$ emacs -nw ../OgreMain/src/OgreProgressiveMeshGenerator.cpp
# After the function body of ProgressiveMeshGenerator::addIndexDataImpl(),
# instantiate the template function:

template void ProgressiveMeshGenerator::addIndexDataImpl<unsigned short>(
  unsigned short* iPos,
  const unsigned short* iEnd,
  VertexLookupList& lookup,
  unsigned short submeshID);

template void ProgressiveMeshGenerator::addIndexDataImpl<unsigned int>(
  unsigned int* iPos,
  const unsigned int* iEnd,
  VertexLookupList& lookup,
  unsigned short submeshID);
```

### MuJoCo

```bash
$ mkdir -p ~/.mujoco/; cd $_
$ wget https://www.roboti.us/download/mujoco200_linux.zip
$ unzip mujoco200_linux.zip
$ mv mujoco200_linux mujoco200
$ cp /path/to/mjkey.txt ./mujoco200/bin/

# Check installation. You can skip followings:
$ export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin/:$LD_LIBRARY_PATH
$ cd mujoco200/bin
$ ./simulate
MuJoCo Pro version 2.00
```

## Build and run the simulator

### Build and configure

```bash
$ bash install.sh
```

Now the files and models will be installed to the `maze_simulator/install`.
