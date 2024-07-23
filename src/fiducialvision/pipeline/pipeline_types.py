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
    pose: Pose3d
    reproj_error: float
    pose_alternate: Union[Pose3d, None]
    reproj_error_alternate: Union[float, None]

