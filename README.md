# Orion

A vision and pose estimation system for FIRST Robotics Competition robots. Inspired by Team 6328's [Northstar](https://github.com/Mechanical-Advantage/RobotCode2024/tree/main/vision/northstar) 
and Team 1678's fork of it, [Polaris](https://github.com/frc1678/C2024-Public/tree/main/polaris). 

Currently only tested on an Orange Pi 5 running Armbian Bookworm 24.5.5, with an Arducam OV9281 camera

## Features
 - Tag detection using OpenCV's ArUco module
 - Multi-target and single-target pose estimation
 - Hardware video decoding using GStreamer
 - Configuration and output over NT4
 - MJPEG stream for debugging
 - Achieves 70-90 FPS with tested configuration

## Setup
1. OpenCV doesn't come with GStreamer support by default, so we need to build it from source. Run the following command:
    ```bash 
    source build-opencv.sh
    ```
   This will install OpenCV and GStreamer's dependencies and build an OpenCV Python wheel from source. This may take a 
   while (around 45 minutes to an hour on an Orange Pi 5).
2. Poetry is used to manage dependencies. If it's not installed, follow the instructions [here](https://python-poetry.org/docs/#installing-with-the-official-installer),
   and then run the following to set up a Poetry virtual environment and install all dependencies:
   ```bash
   poetry install
   ```
3. Make a copy of `./device-config/example-network-config.json` as `./device-config/network-config.json`, and set the
   fields as needed. The `device_id` field will be the name of the subtable of the `orion` table that this device will
   use, and should be unique. The `server_ip` field should be the IP address of the NT4 server. This will usually be the
   RoboRIO's IP (`10.TE.AM.2`). The `stream_port` will be the port that the MJPEG stream will be served on. This only
   needs to be changed if there is more than one instance running on the same device.
4. Use [CalibDB](https://calibdb.net) to calibrate your camera and export the calibration file using OpenCV formatting.
   Save this file to `./device-config/calibration.json`.
5. Run
    ```bash
    poetry run python -m orion
    ```
   to start the vision system.