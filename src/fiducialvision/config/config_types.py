from dataclasses import dataclass
from typing import Dict, Union

import cv2
import numpy as np
import numpy.typing as npt
from wpimath.geometry import Pose3d


@dataclass
class NetworkConfig:
    device_id: str = "vision"
    server_ip: str = "10.15.40.2"
    stream_port: int = "8000"


@dataclass
class Calibration:
    intrinsics_matrix: Union[npt.NDArray[np.float64], None] = None
    distortion_coeffs: Union[npt.NDArray[np.float64], None] = None


@dataclass
class CameraConfig:
    id: Union[int, str] = 0
    resolution_height: int = 720
    resolution_width: int = 1280
    auto_exposure: float = 0.0
    exposure: float = 0.0
    gain: float = 0.0


@dataclass
class FiducialConfig:
    tag_family: int = cv2.aruco.DICT_APRILTAG_36h11
    tag_size_m: float = 0.1651
    tag_layout: Union[Dict[int, Pose3d], None] = None

