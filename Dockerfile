# syntax=docker/dockerfile:1

FROM --platform=$BUILDPLATFORM tonistiigi/xx AS xx

FROM --platform=$BUILDPLATFORM debian:bookworm-20240812 AS crossenv
COPY --from=xx / /
ARG TARGETPLATFORM

# Build build-python from source
RUN apt-get update && apt-get install -y \
    wget \
    buid-essential \
    cmake \
    libssl-dev \
    zlib1g-dev \
    libffi-dev \
    && wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz \
    && tar -zxvf Python-3.11.9.tgz \
    && cp -r Python-3.11.9 Python-3.11.9-host
WORKDIR /Python-3.11.9
RUN ./configure --prefix=/build-python \
    && make -j$(nproc) \
    && make install

# Build host-python from source
WORKDIR /Python-3.11.9-host
RUN xx-apt-get update && xx-apt-get install -y \
    xx-c-essentials \
    xx-cxx-essentials \
    libssl-dev \
    zlib1g-dev \
    libffi-dev \
    pkg-config \
    && apt-get update && apt-get install -y \
      gcc-$(xx-info) \
      g++-$(xx-info) \
      binutils-$(xx-info)
RUN PATH=/build-python/bin:$PATH ./configure \
    --prefix=/host-python \
    --host=$(xx-info) \
    --build=$(gcc -print-multiarch) \
    --without-ensurepip \
    --enable-shared \
    --enable-optimizations \
    --with-lto \
    ac_cv_buggy_getaddrinfo=no \
    ac_cv_file__dev_ptmx=yes \
    ac_cv_file__dev_ptc=no \
    && make -j$(nproc) \
    && make install

# Create virtual environment
WORKDIR /crossenv
RUN /build-python/bin/pip3 install crossenv
RUN /build-python/bin/python3 -m crossenv /host-python/bin/python3 venv

FROM crossenv AS opencv
WORKDIR /crossenv
ENV PATH="/crossenv/venv/bin:/crossenv/venv/cross/bin$PATH"
# Install GStreamer dependencies
RUN xx-apt-get update && xx-apt-get install -y \
    gstreamer1.0-gl \
    gstreamer1.0-opencv \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-0 \
    libgstreamer1.0-dev \
RUN wget https://github.com/opencv/opencv-python/archive/refs/tags/82.tar.gz \
    && tar -zxvf 82.tar.gz \
    && mv 82 opencv-python \
# Build OpenCV from source with GStreamer enabled
WORKDIR /crossenv/opencv-python
ENV ENABLE_HEADLESS=1
ENV ENABLE_CONTRIB=0
ENV CMAKE_ARGS="-DWITH_GSTREAMER=ON -DBUILD_NEW_PYTHON_SUPPORT=ON -DBUILD_opencv_python3=ON -DHAVE_opencv_python3=ON"
ENV MAKEFLAGS="-j$(nproc)"
RUN python3 -m pip wheel . --verbose --wheel-dir=/wheels

FROM --platform=$TARGETPLATFORM python:3.11-bookworm
WORKDIR /orion
RUN apt-get update && apt-get install -y \
    gstreamer1.0-gl \
    gstreamer1.0-opencv \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-0 \
    libgstreamer1.0-dev \
COPY --from=opencv /wheels ./wheels
RUN pip3 install numpy==1.23.5 \
    && pip3 install ./wheels/opencv_python*.whl \
    && pip3 install --index-url https://wpilib.jfrog.io/artifactory/api/pypi/wpilib-python-release-2024/simple robotpy-wpimath==2024.3.2.1 \
    && pip3 install --index-url https://wpilib.jfrog.io/artifactory/api/pypi/wpilib-python-release-2024/simple pyntcore==2024.3.2.1 \
RUN rm -rf ./wheels
COPY . .
CMD python3 -m orion
