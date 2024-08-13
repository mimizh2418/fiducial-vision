#!/bin/bash

set -euo pipefail

# Install dependencies
sudo apt-get install --quiet -y --no-install-recommends \
  build-essential \
  cmake\
  gstreamer1.0-gl \
  gstreamer1.0-opencv \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-tools \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer1.0-0 \
  libgstreamer1.0-dev

OPENCV_VER="82"
PROJDIR=$(pwd)
TMPDIR=$(mktemp -d)

# Build and install OpenCV from source
cd "${TMPDIR}"
git clone --branch ${OPENCV_VER} --depth 1 --recurse-submodules --shallow-submodules https://github.com/opencv/opencv-python.git opencv-python-${OPENCV_VER}
cd opencv-python-${OPENCV_VER}
export ENABLE_CONTRIB=0
export ENABLE_HEADLESS=1
# Configure OpenCV build
export CMAKE_ARGS="-DWITH_GSTREAMER=ON -DBUILD_NEW_PYTHON_SUPPORT=ON -DBUILD_opencv_python3=ON -DHAVE_opencv_python3=ON"
export MAKEFLAGS="-j$(nproc)"
python3 -m pip wheel . --verbose --wheel-dir ${PROJDIR}/wheels/
