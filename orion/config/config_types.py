from dataclasses import dataclass
from typing import Dict, Union, Optional

import cv2
import numpy as np
import numpy.typing as npt
from wpimath.geometry import Pose3d


@dataclass
class NetworkConfig:
    device_id: str = "orion"
    server_ip: str = "10.15.40.2"
    stream_port: str = "8000"


@dataclass
class Calibration:
    intrinsics_matrix: Optional[npt.NDArray[np.float64]] = None
    distortion_coeffs: Optional[npt.NDArray[np.float64]] = None


@dataclass
class CameraConfig:
    id: Union[int, str] = 0
    resolution_height: int = 720
    resolution_width: int = 1280
    auto_exposure: int = 1
    exposure: int = 25
    brightness: int = 0
    gain: int = 20


@dataclass
class FiducialConfig:
    tag_family: int = cv2.aruco.DICT_APRILTAG_36h11
    tag_size_m: float = 0.1651
    tag_layout: Optional[Dict[int, Pose3d]] = None
