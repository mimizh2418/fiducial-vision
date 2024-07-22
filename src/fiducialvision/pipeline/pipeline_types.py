from dataclasses import dataclass
from typing import List, Union

import numpy as np
from numpy import typing as npt
from wpimath.geometry import Pose3d


@dataclass(frozen=True)
class FiducialTagObservation:
    id: int
    corners: npt.NDArray[np.float64]


@dataclass(frozen=True)
class PoseEstimate:
    tag_ids: List[int]
    pose_0: Pose3d
    error_0: float
    pose_1: Union[Pose3d, None]
    error_1: Union[float, None]

