from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import numpy.typing as npt
from wpimath.geometry import Pose3d


@dataclass(frozen=True)
class CameraCalibrationParams:
    intrinsics_matrix: npt.NDArray[np.float64]
    distortion_coefficients: npt.NDArray[np.float64]


@dataclass(frozen=True)
class CameraConfig:
    id: Union[int, str]
    resolution_height: int
    resolution_width: int
    auto_exposure: float
    exposure: float
    gain: float


@dataclass(frozen=True)
class FiducialConfig:
    tag_family: int
    tag_size_m: float
    tag_layout: Dict[int, Pose3d]

