import json

import cv2.aruco
import numpy as np
from wpimath.geometry import Pose3d, Rotation3d

from .config_types import CameraConfig, FiducialConfig, CameraCalibrationParams

fiducial_families = {
    'aruco_4x4_50': cv2.aruco.DICT_4X4_50,
    'aruco_4x4_100': cv2.aruco.DICT_4X4_100,
    'aruco_4x4_250': cv2.aruco.DICT_4X4_250,
    'aruco_4x4_1000': cv2.aruco.DICT_4X4_1000,
    'aruco_5x5_50': cv2.aruco.DICT_5X5_50,
    'aruco_5x5_100': cv2.aruco.DICT_5X5_100,
    'aruco_5x5_250': cv2.aruco.DICT_5X5_250,
    'aruco_5x5_1000': cv2.aruco.DICT_5X5_1000,
    'aruco_6x6_50': cv2.aruco.DICT_6X6_50,
    'aruco_6x6_100': cv2.aruco.DICT_6X6_100,
    'aruco_6x6_250': cv2.aruco.DICT_6X6_250,
    'aruco_6x6_1000': cv2.aruco.DICT_6X6_1000,
    'aruco_7x7_50': cv2.aruco.DICT_7X7_50,
    'aruco_7x7_100': cv2.aruco.DICT_7X7_100,
    'aruco_7x7_250': cv2.aruco.DICT_7X7_250,
    'aruco_7x7_1000': cv2.aruco.DICT_7X7_1000,
    'apriltag_16h5': cv2.aruco.DICT_APRILTAG_16h5,
    'apriltag_25h9': cv2.aruco.DICT_APRILTAG_25h9,
    'apriltag_36h10': cv2.aruco.DICT_APRILTAG_36h10,
    'apriltag_36h11': cv2.aruco.DICT_APRILTAG_36h11,
    'aruco_mip_36h12': cv2.aruco.DICT_ARUCO_MIP_36h12
}


def load_camera_calibration(calib_filename: str) -> CameraCalibrationParams:
    calib_file = cv2.FileStorage(calib_filename, cv2.FILE_STORAGE_READ)
    intrinsics_mat = calib_file.getNode("camera_matrix").mat()
    dist_coeffs = calib_file.getNode("distortion_coefficients").mat()
    calib_file.release()

    if type(intrinsics_mat) is not np.ndarray or type(dist_coeffs) is not np.ndarray:
        raise ValueError("Invalid calibration file")

    return CameraCalibrationParams(intrinsics_mat, dist_coeffs)


def load_camera_config(config_filename: str) -> CameraConfig:
    with open(config_filename, 'r') as f:
        cfg_data = json.loads(f.read())

    camera_id = cfg_data['id']
    resolution_width = cfg_data['img_width']
    resolution_height = cfg_data['img_height']
    auto_exposure = cfg_data['auto_exposure']
    exposure = cfg_data['exposure']
    gain = cfg_data['gain']

    return CameraConfig(camera_id, resolution_height, resolution_width, auto_exposure, exposure, gain)


def load_fiducial_config(config_filename: str) -> FiducialConfig:
    with open(config_filename, 'r') as f:
        cfg_data = json.loads(f.read())

    family = fiducial_families[cfg_data['tag_family']]
    size = cfg_data['tag_size']
    layout_data = cfg_data['tag_layout']
    layout = {}
    for tag_data in layout_data:
        tag_id = tag_data['id']
        translation_data = tag_data['position']
        rotation_data = tag_data['rotation']
        pose = Pose3d(translation_data[0],
                      translation_data[1],
                      translation_data[2],
                      Rotation3d(rotation_data["roll"], rotation_data["pitch"], rotation_data["yaw"]))
        layout[tag_id] = pose

    return FiducialConfig(family, size, layout)
