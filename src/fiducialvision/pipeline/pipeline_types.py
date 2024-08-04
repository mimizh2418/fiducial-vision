from dataclasses import dataclass
from typing import Union, Sequence, Optional

import cv2
import numpy as np
from numpy import typing as npt
from wpimath.geometry import Pose3d


@dataclass(frozen=True)
class CaptureFrame:
    image: cv2.Mat
    timestamp_ns: int
    resolution_height: int
    resolution_width: int


@dataclass(frozen=True)
class FiducialTagDetection:
    id: int
    corners: npt.NDArray[np.float64]


@dataclass
class CameraPoseEstimate:
    pose: Pose3d
    reproj_error: float
    pose_alternate: Optional[Pose3d]
    reproj_error_alternate: Optional[float]


@dataclass(frozen=True)
class PipelineResult:
    capture_timestamp_ns: int
    process_dt_ns: int
    processed_image: cv2.Mat

    detector_result: Sequence[FiducialTagDetection]
    pose_estimate: Union[CameraPoseEstimate, None]
