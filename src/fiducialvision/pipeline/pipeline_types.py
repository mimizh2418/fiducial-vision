from dataclasses import dataclass
from typing import Union, Sequence

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
class PoseEstimatorResult:
    pose: Pose3d
    reproj_error: float
    pose_alternate: Union[Pose3d, None]
    reproj_error_alternate: Union[float, None]


@dataclass(frozen=True)
class PipelineResult:
    capture_timestamp_ns: int
    process_dt_ns: int
    processed_image: cv2.Mat

    detector_result: Sequence[FiducialTagDetection]
    pose_estimate: Union[PoseEstimatorResult, None]
