from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt

from ..geometry import Pose3d


@dataclass(frozen=True)
class CameraConfig:
    index: int
    camera_resolution_height: int
    camera_resolution_width: int
    intrinsics_matrix: npt.NDArray[np.float64]
    distortion_coefficients: npt.NDArray[np.float64]


@dataclass(frozen=True)
class FiducialConfig:
    tag_family: int
    tag_size_m: float
    tag_layout: Dict[int, Pose3d]

